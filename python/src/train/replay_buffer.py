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
    action_dim : int, optional
        Number of discrete actions represented by the available-action mask.
        If omitted, it is inferred from `action_shape`.

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
                 device=torch.device("cpu"), action_dim=None):
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
        action_dim : int, optional
            Number of discrete actions available to each agent. This is used
            for the available_actions buffer. If None, it is inferred from
            action_shape as a fallback.

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

        # Determine action_dim for available_actions.
        #
        # Important:
        # - action_shape describes the stored selected action, e.g. (1,)
        # - action_dim describes the number of possible discrete actions, e.g. 5
        #
        # If action_dim is not explicitly provided, we infer a fallback from
        # action_shape to keep old constructor calls compatible.
        if action_dim is None:
            if isinstance(action_shape, (tuple, list)) and len(action_shape) > 0:
                action_dim = int(action_shape[0])
            else:
                action_dim = int(action_shape)

        self.action_dim = int(action_dim)

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
        self.available_actions = np.ones((buffer_size + 1, num_agents, self.action_dim), dtype=np.float32)  # Placeholder
        self.candidate_target_features = None
        self.candidate_target_mask = None    

        self.step = 0

    def insert(
        self,
        share_obs,
        obs,
        rnn_states,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        active_masks=None,
        available_actions=None,
        candidate_target_features=None,
        candidate_target_mask=None,
    ):
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
        if self.step >= self.buffer_size:
            raise RuntimeError(
                "Replay buffer is full. Call reset() before inserting "
                f"more than {self.buffer_size} transitions."
            )

        self.share_obs[self.step] = share_obs
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks
        self.active_masks[self.step + 1] = active_masks

        if available_actions is not None:
            available_actions = np.asarray(available_actions, dtype=np.float32)

            # If the buffer was created without explicit action_dim, it may have
            # inferred action_dim from action_shape, e.g. action_shape=(1,).
            # For discrete action spaces, available_actions can have a different
            # final dimension, e.g. 7. Resize once if needed.
            if available_actions.shape[-1] != self.available_actions.shape[-1]:
                self.action_dim = int(available_actions.shape[-1])
                new_available_actions = np.ones(
                    (self.buffer_size + 1, self.num_agents, self.action_dim),
                    dtype=np.float32,
                )

                steps_to_copy = min(
                    self.available_actions.shape[0],
                    new_available_actions.shape[0],
                )
                agents_to_copy = min(
                    self.available_actions.shape[1],
                    new_available_actions.shape[1],
                )
                actions_to_copy = min(
                    self.available_actions.shape[2],
                    new_available_actions.shape[2],
                )

                new_available_actions[
                    :steps_to_copy,
                    :agents_to_copy,
                    :actions_to_copy,
                ] = self.available_actions[
                    :steps_to_copy,
                    :agents_to_copy,
                    :actions_to_copy,
                ]

                self.available_actions = new_available_actions

            self.available_actions[self.step] = available_actions

            if candidate_target_features is not None:
                target_features = np.asarray(
                    candidate_target_features,
                    dtype=np.float32,
                )

                if target_features.shape[0] != self.num_agents:
                    raise ValueError(
                        "candidate_target_features must have num_agents "
                        f"as first dimension; got {target_features.shape}"
                    )

                expected_shape = (self.buffer_size, *target_features.shape)

                if self.candidate_target_features is None:
                    self.candidate_target_features = np.zeros(
                        expected_shape,
                        dtype=np.float32,
                    )
                elif self.candidate_target_features.shape[1:] != target_features.shape:
                    raise ValueError(
                        "candidate_target_features shape changed: "
                        f"expected {self.candidate_target_features.shape[1:]}, "
                        f"got {target_features.shape}"
                    )

                self.candidate_target_features[self.step] = target_features


            if candidate_target_mask is not None:
                target_mask = np.asarray(
                    candidate_target_mask,
                    dtype=np.bool_,
                )

                if target_mask.shape[0] != self.num_agents:
                    raise ValueError(
                        "candidate_target_mask must have num_agents "
                        f"as first dimension; got {target_mask.shape}"
                    )

                expected_shape = (self.buffer_size, *target_mask.shape)

                if self.candidate_target_mask is None:
                    self.candidate_target_mask = np.zeros(
                        expected_shape,
                        dtype=np.bool_,
                    )
                elif self.candidate_target_mask.shape[1:] != target_mask.shape:
                    raise ValueError(
                        "candidate_target_mask shape changed: "
                        f"expected {self.candidate_target_mask.shape[1:]}, "
                        f"got {target_mask.shape}"
                    )

                self.candidate_target_mask[self.step] = target_mask

        self.step += 1

    def compute_returns(
        self,
        next_value,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        valid_steps = self.step

        if valid_steps <= 0:
            raise RuntimeError(
                "Cannot compute returns for an empty buffer."
            )

        next_value = np.asarray(
            next_value,
            dtype=np.float32,
        )

        expected_shape = (
            self.num_agents,
            1,
        )

        if next_value.shape != expected_shape:
            raise ValueError(
                "Invalid next value shape: "
                f"expected {expected_shape}, "
                f"got {next_value.shape}."
            )

        self.value_preds[valid_steps] = next_value

        gae = np.zeros(
            (self.num_agents, 1),
            dtype=np.float32,
        )

        for step in reversed(range(valid_steps)):
            next_non_terminal = self.masks[step + 1]
            next_value_step = self.value_preds[step + 1]

            delta = (
                self.rewards[step]
                + gamma
                * next_value_step
                * next_non_terminal
                - self.value_preds[step]
            )

            gae = (
                delta
                + gamma
                * gae_lambda
                * next_non_terminal
                * gae
            )

            self.returns[step] = (
                gae + self.value_preds[step]
            )

        self.returns[valid_steps] = next_value

    def feed_forward_generator(
        self,
        advantages,
        num_mini_batch,
    ):
        """
        Generate PPO mini-batches flattened over time and agents.

        Input storage:
            obs:               [T, N, obs_dim]
            actions:           [T, N, action_width]
            available_actions: [T, N, action_dim]

        Generated batches:
            obs:               [B, obs_dim]
            actions:           [B, action_width]
            available_actions: [B, action_dim]
        """
        valid_steps = self.step

        if valid_steps <= 0:
            raise RuntimeError(
                "Cannot generate training batches from an empty buffer."
            )

        if num_mini_batch <= 0:
            raise ValueError(
                "num_mini_batch must be greater than zero."
            )

        advantages = np.asarray(
            advantages,
            dtype=np.float32,
        )[:valid_steps]

        expected_advantage_prefix = (
            valid_steps,
            self.num_agents,
        )

        if advantages.shape[:2] != expected_advantage_prefix:
            raise ValueError(
                "Invalid advantages shape: expected prefix "
                f"{expected_advantage_prefix}, got {advantages.shape}."
            )

        total_samples = valid_steps * self.num_agents

        if num_mini_batch > total_samples:
            raise ValueError(
                "num_mini_batch cannot exceed the number of "
                f"samples: batches={num_mini_batch}, "
                f"samples={total_samples}."
            )

        # share_obs ist einmal pro Zeitschritt gespeichert.
        # Jeder Agent desselben Schritts erhält denselben zentralen Zustand.
        share_obs_flat = np.repeat(
            self.share_obs[:valid_steps, None, ...],
            self.num_agents,
            axis=1,
        ).reshape(
            total_samples,
            *self.share_obs.shape[1:],
        )

        obs_flat = self.obs[:valid_steps].reshape(
            total_samples,
            *self.obs.shape[2:],
        )

        rnn_states_flat = self.rnn_states[:valid_steps].reshape(
            total_samples,
            *self.rnn_states.shape[2:],
        )

        rnn_states_critic_flat = (
            self.rnn_states_critic[:valid_steps].reshape(
                total_samples,
                *self.rnn_states_critic.shape[2:],
            )
        )

        actions_flat = self.actions[:valid_steps].reshape(
            total_samples,
            *self.actions.shape[2:],
        )

        value_preds_flat = self.value_preds[:valid_steps].reshape(
            total_samples,
            *self.value_preds.shape[2:],
        )

        returns_flat = self.returns[:valid_steps].reshape(
            total_samples,
            *self.returns.shape[2:],
        )

        masks_flat = self.masks[:valid_steps].reshape(
            total_samples,
            *self.masks.shape[2:],
        )

        active_masks_flat = self.active_masks[:valid_steps].reshape(
            total_samples,
            *self.active_masks.shape[2:],
        )

        old_action_log_probs_flat = (
            self.action_log_probs[:valid_steps].reshape(
                total_samples,
                *self.action_log_probs.shape[2:],
            )
        )

        advantages_flat = advantages.reshape(
            total_samples,
            *advantages.shape[2:],
        )

        available_actions_flat = (
            self.available_actions[:valid_steps].reshape(
                total_samples,
                self.action_dim,
            )
        )

        if self.candidate_target_features is None:
            candidate_target_features_flat = None
        else:
            candidate_target_features_flat = (
                self.candidate_target_features[:valid_steps].reshape(
                    total_samples,
                    *self.candidate_target_features.shape[2:],
                )
            )

        if self.candidate_target_mask is None:
            candidate_target_mask_flat = None
        else:
            candidate_target_mask_flat = (
                self.candidate_target_mask[:valid_steps].reshape(
                    total_samples,
                    *self.candidate_target_mask.shape[2:],
                )
            )

        permutation = np.random.permutation(total_samples)

        # array_split verliert keine Samples, falls total_samples nicht
        # glatt durch num_mini_batch teilbar ist.
        mini_batches = np.array_split(
            permutation,
            num_mini_batch,
        )

        for batch_indices in mini_batches:
            if batch_indices.size == 0:
                continue

            share_obs_batch = share_obs_flat[batch_indices]
            obs_batch = obs_flat[batch_indices]
            rnn_states_batch = rnn_states_flat[batch_indices]
            rnn_states_critic_batch = (
                rnn_states_critic_flat[batch_indices]
            )
            actions_batch = actions_flat[batch_indices]
            value_preds_batch = value_preds_flat[batch_indices]
            return_batch = returns_flat[batch_indices]
            masks_batch = masks_flat[batch_indices]
            active_masks_batch = active_masks_flat[batch_indices]
            old_action_log_probs_batch = (
                old_action_log_probs_flat[batch_indices]
            )
            adv_targ = advantages_flat[batch_indices]
            available_actions_batch = (
                available_actions_flat[batch_indices]
            )

            candidate_target_features_batch = (
                None
                if candidate_target_features_flat is None
                else candidate_target_features_flat[batch_indices]
            )

            candidate_target_mask_batch = (
                None
                if candidate_target_mask_flat is None
                else candidate_target_mask_flat[batch_indices]
            )

            # Fehler möglichst nah am Buffer erkennen.
            batch_size = len(batch_indices)

            if obs_batch.shape[0] != batch_size:
                raise RuntimeError(
                    f"Invalid obs batch shape: {obs_batch.shape}."
                )

            if actions_batch.shape != (
                batch_size,
                *self.actions.shape[2:],
            ):
                raise RuntimeError(
                    f"Invalid actions batch shape: {actions_batch.shape}."
                )

            if available_actions_batch.shape != (
                batch_size,
                self.action_dim,
            ):
                raise RuntimeError(
                    "Invalid available-actions batch shape: "
                    f"expected {(batch_size, self.action_dim)}, "
                    f"got {available_actions_batch.shape}."
                )

            if candidate_target_features_batch is not None:
                if candidate_target_features_batch.shape[0] != batch_size:
                    raise RuntimeError(
                        "Invalid target-feature batch shape: "
                        f"{candidate_target_features_batch.shape}."
                    )

            if candidate_target_mask_batch is not None:
                if candidate_target_mask_batch.shape[0] != batch_size:
                    raise RuntimeError(
                        "Invalid target-mask batch shape: "
                        f"{candidate_target_mask_batch.shape}."
                    )

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
                available_actions_batch,
                candidate_target_features_batch,
                candidate_target_mask_batch,
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
        self.rnn_states.fill(0.0)
        self.rnn_states_critic.fill(0.0)
        self.available_actions.fill(1.0)

        if self.candidate_target_features is not None:
            self.candidate_target_features.fill(0.0)

        if self.candidate_target_mask is not None:
            self.candidate_target_mask.fill(False)