import numpy as np
import torch


class SharedReplayBuffer:
    """
    Shared replay buffer for MAPPO training.

    Stores trajectories collected during rollout and provides generators
    that yield mini-batches for training.

    Parameters
    ----------
    num_agents : int
        Number of agents in the environment.
    obs_shape : tuple or list
        Shape of a single observation.
    action_shape : tuple or list
        Shape of a single action.
    buffer_size : int
        Maximum number of transitions stored in the buffer.
    device : torch.device, optional
        Torch device used for training, by default torch.device("cpu").

    Returns
    -------
    SharedReplayBuffer
        A replay buffer instance with preallocated numpy arrays for storing
        MAPPO rollout data.

    Examples
    --------
    >>> buffer = SharedReplayBuffer(
    ...     num_agents=2,
    ...     obs_shape=(128,),
    ...     action_shape=(1,),
    ...     buffer_size=256
    ... )
    >>> buffer.step
    0
    """

    def __init__(self, num_agents: int, obs_shape, action_shape, buffer_size: int,
                 device=torch.device("cpu")):
        """
        Initializes the shared replay buffer and allocates storage arrays.

        Parameters
        ----------
        self : SharedReplayBuffer
            The replay buffer instance.
        num_agents : int
            Number of agents in the environment.
        obs_shape : tuple or list
            Shape of a single observation.
        action_shape : tuple or list
            Shape of a single action.
        buffer_size : int
            Maximum number of rollout steps stored in the buffer.
        device : torch.device, optional
            Torch device used for training, by default torch.device("cpu").

        Returns
        -------
        None

        Examples
        --------
        >>> buffer = SharedReplayBuffer(2, (64,), (1,), 128)
        >>> buffer.buffer_size
        128
        >>> buffer.num_agents
        2
        """
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.device = device

        # Allocate buffers
        self.share_obs = np.zeros((buffer_size + 1, *obs_shape), dtype=np.float32)
        self.obs = np.zeros((buffer_size + 1, num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, *action_shape), dtype=np.float32)
        self.value_preds = np.zeros((buffer_size + 1, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((buffer_size + 1, num_agents, 1), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents, 1), dtype=np.float32)
        self.masks = np.ones((buffer_size + 1, num_agents, 1), dtype=np.float32)
        self.active_masks = np.ones((buffer_size + 1, num_agents, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((buffer_size, num_agents, 1), dtype=np.float32)
        self.rnn_states = np.zeros((buffer_size + 1, num_agents, 1), dtype=np.float32)  # Placeholder
        self.rnn_states_critic = np.zeros((buffer_size + 1, num_agents, 1), dtype=np.float32)  # Placeholder
        self.available_actions = np.ones((buffer_size + 1, num_agents, action_dim), dtype=np.float32)  # Placeholder

        self.step = 0

    def insert(self, share_obs, obs, actions, action_log_probs, value_preds,
               rewards, masks, active_masks, available_actions=None):
        """
        Inserts one transition into the replay buffer.

        Parameters
        ----------
        self : SharedReplayBuffer
            The replay buffer instance.
        share_obs : np.ndarray
            Shared or centralized observation stored for the current timestep.
        obs : np.ndarray
            Per-agent observations for the current timestep.
        actions : np.ndarray
            Actions taken by the agents at the current timestep.
        action_log_probs : np.ndarray
            Log-probabilities of the selected actions.
        value_preds : np.ndarray
            Critic value predictions for the current timestep.
        rewards : np.ndarray
            Rewards received after executing the actions.
        masks : np.ndarray
            Mask values indicating whether the episode continues at the next timestep.
        active_masks : np.ndarray
            Masks indicating whether an agent is active.
        available_actions : np.ndarray, optional
            Legal/available actions for the next timestep, by default None.

        Returns
        -------
        None

        Examples
        --------
        >>> buffer = SharedReplayBuffer(2, (4,), (1,), 8)
        >>> share_obs = np.zeros((4,), dtype=np.float32)
        >>> obs = np.zeros((2, 4), dtype=np.float32)
        >>> actions = np.zeros((2, 1), dtype=np.float32)
        >>> action_log_probs = np.zeros((2, 1), dtype=np.float32)
        >>> value_preds = np.zeros((2, 1), dtype=np.float32)
        >>> rewards = np.ones((2, 1), dtype=np.float32)
        >>> masks = np.ones((2, 1), dtype=np.float32)
        >>> active_masks = np.ones((2, 1), dtype=np.float32)
        >>> buffer.insert(share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, active_masks)
        >>> buffer.step
        1
        """
        self.share_obs[self.step] = share_obs
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks
        self.active_masks[self.step + 1] = active_masks
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions

        self.step = (self.step + 1) % self.buffer_size

    def compute_returns(self, next_value, gamma=0.99, gae_lambda=0.95):
        """
        Computes discounted returns and generalized advantage estimates (GAE).

        Parameters
        ----------
        self : SharedReplayBuffer
            The replay buffer instance.
        next_value : np.ndarray
            Value prediction at the end of the trajectory.
        gamma : float, optional
            Discount factor, by default 0.99.
        gae_lambda : float, optional
            Lambda parameter for generalized advantage estimation, by default 0.95.

        Returns
        -------
        None

        Examples
        --------
        >>> buffer = SharedReplayBuffer(2, (4,), (1,), 4)
        >>> next_value = np.zeros((2, 1), dtype=np.float32)
        >>> buffer.compute_returns(next_value)
        >>> buffer.returns.shape
        (5, 2, 1)
        """
        self.value_preds[-1] = next_value
        gae = 0
        advantages = np.zeros_like(self.rewards)

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = self.masks[step + 1]
                next_value_step = next_value
            else:
                next_non_terminal = self.masks[step + 1]
                next_value_step = self.value_preds[step + 1]

            delta = self.rewards[step] + gamma * next_value_step * next_non_terminal - self.value_preds[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[step] = gae

        self.returns[:-1] = advantages + self.value_preds[:-1]

    def feed_forward_generator(self, advantages, num_mini_batch):
        """
        Generates mini-batches for feed-forward policy optimization.

        Parameters
        ----------
        self : SharedReplayBuffer
            The replay buffer instance.
        advantages : np.ndarray
            Precomputed advantage estimates for all stored transitions.
        num_mini_batch : int
            Number of mini-batches to generate.

        Returns
        -------
        generator : generator
            A generator yielding tuples containing mini-batch tensors/arrays
            for feed-forward MAPPO training.

        Examples
        --------
        >>> buffer = SharedReplayBuffer(2, (4,), (1,), 8)
        >>> advantages = np.zeros((8, 2, 1), dtype=np.float32)
        >>> gen = buffer.feed_forward_generator(advantages, 2)
        >>> batch = next(gen)
        >>> len(batch)
        12
        """
        batch_size = self.buffer_size // num_mini_batch
        sampler = np.random.permutation(self.buffer_size)

        for batch_idx in range(num_mini_batch):
            batch_indices = sampler[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            share_obs_batch = self.share_obs[batch_indices]
            obs_batch = self.obs[batch_indices]
            rnn_states_batch = self.rnn_states[batch_indices]
            rnn_states_critic_batch = self.rnn_states_critic[batch_indices]
            actions_batch = self.actions[batch_indices]
            value_preds_batch = self.value_preds[batch_indices]
            return_batch = self.returns[batch_indices]
            masks_batch = self.masks[batch_indices]
            active_masks_batch = self.active_masks[batch_indices]
            old_action_log_probs_batch = self.action_log_probs[batch_indices]
            adv_targ = advantages[batch_indices]
            available_actions_batch = self.available_actions[batch_indices]

            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch
            )

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Generates mini-batches for recurrent network training.

        Currently this method falls back to the feed-forward generator and does
        not yet implement recurrent sequence chunking.

        Parameters
        ----------
        self : SharedReplayBuffer
            The replay buffer instance.
        advantages : np.ndarray
            Precomputed advantage estimates for all stored transitions.
        num_mini_batch : int
            Number of mini-batches to generate.
        data_chunk_length : int
            Length of sequence chunks intended for recurrent training.

        Returns
        -------
        generator : generator
            A generator yielding mini-batches. Currently identical to
            `feed_forward_generator`.

        Examples
        --------
        >>> buffer = SharedReplayBuffer(2, (4,), (1,), 8)
        >>> advantages = np.zeros((8, 2, 1), dtype=np.float32)
        >>> gen = buffer.recurrent_generator(advantages, 2, 4)
        >>> batch = next(gen)
        >>> len(batch)
        12
        """
        return self.feed_forward_generator(advantages, num_mini_batch)

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """
        Generates mini-batches for naive recurrent training.

        Currently this method falls back to the feed-forward generator and does
        not yet implement a dedicated naive recurrent batching strategy.

        Parameters
        ----------
        self : SharedReplayBuffer
            The replay buffer instance.
        advantages : np.ndarray
            Precomputed advantage estimates for all stored transitions.
        num_mini_batch : int
            Number of mini-batches to generate.

        Returns
        -------
        generator : generator
            A generator yielding mini-batches. Currently identical to
            `feed_forward_generator`.

        Examples
        --------
        >>> buffer = SharedReplayBuffer(2, (4,), (1,), 8)
        >>> advantages = np.zeros((8, 2, 1), dtype=np.float32)
        >>> gen = buffer.naive_recurrent_generator(advantages, 2)
        >>> batch = next(gen)
        >>> len(batch)
        12
        """
        return self.feed_forward_generator(advantages, num_mini_batch)

    def reset(self):
        """
        Resets the replay buffer to its initial empty state.

        Parameters
        ----------
        self : SharedReplayBuffer
            The replay buffer instance.

        Returns
        -------
        None

        Examples
        --------
        >>> buffer = SharedReplayBuffer(2, (4,), (1,), 8)
        >>> buffer.step = 5
        >>> buffer.reset()
        >>> buffer.step
        0
        """
        self.step = 0
        self.share_obs.fill(0.0)
        self.obs.fill(0.0)
        self.actions.fill(0.0)
        self.value_preds.fill(0.0)
        self.returns.fill(0.0)
        self.rewards.fill(0.0)
        self.masks.fill(1.0)
        self.active_masks.fill(1.0)
        self.action_log_probs.fill(0.0)