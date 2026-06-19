import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    """
    Actor network for policy optimization.

    This network maps observations to action logits, which can be used to build
    a categorical policy distribution.

    Parameters
    ----------
    obs_dim : int
        Dimension of the input observation vector.
    action_dim : int
        Number of discrete actions.
    hidden_dim : int, optional
        Size of the hidden layers, by default 256.

    Returns
    -------
    Actor
        A neural network module representing the policy actor.

    Examples
    --------
    >>> actor = Actor(obs_dim=128, action_dim=7)
    >>> obs = torch.randn(4, 128)
    >>> logits = actor(obs)
    >>> logits.shape
    torch.Size([4, 7])
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initializes the actor network.

        Parameters
        ----------
        self : Actor
            The actor network instance.
        obs_dim : int
            Dimension of the input observation vector.
        action_dim : int
            Number of discrete actions.
        hidden_dim : int, optional
            Size of the hidden layers, by default 256.

        Returns
        -------
        None

        Examples
        --------
        >>> actor = Actor(64, 5)
        >>> isinstance(actor.net, nn.Sequential)
        True
        """
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        """
        Computes action logits from observations.

        Parameters
        ----------
        self : Actor
            The actor network instance.
        obs : torch.Tensor
            Input observation tensor of shape `(batch_size, obs_dim)`.

        Returns
        -------
        action_logits : torch.Tensor
            Tensor containing unnormalized action scores for each action.

        Examples
        --------
        >>> actor = Actor(32, 4)
        >>> obs = torch.randn(2, 32)
        >>> actor(obs).shape
        torch.Size([2, 4])
        """
        return self.net(obs)


class Critic(nn.Module):
    """
    Critic network for value estimation.

    This network maps observations to scalar value predictions used by the
    critic in actor-critic training.

    Parameters
    ----------
    obs_dim : int
        Dimension of the input observation vector.
    hidden_dim : int, optional
        Size of the hidden layers, by default 256.

    Returns
    -------
    Critic
        A neural network module representing the value function critic.

    Examples
    --------
    >>> critic = Critic(obs_dim=128)
    >>> obs = torch.randn(4, 128)
    >>> values = critic(obs)
    >>> values.shape
    torch.Size([4, 1])
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        """
        Initializes the critic network.

        Parameters
        ----------
        self : Critic
            The critic network instance.
        obs_dim : int
            Dimension of the input observation vector.
        hidden_dim : int, optional
            Size of the hidden layers, by default 256.

        Returns
        -------
        None

        Examples
        --------
        >>> critic = Critic(64)
        >>> isinstance(critic.net, nn.Sequential)
        True
        """
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.v_out = self.net  # for POPart compatibility

    def forward(self, obs):
        """
        Computes scalar value predictions from observations.

        Parameters
        ----------
        self : Critic
            The critic network instance.
        obs : torch.Tensor
            Input observation tensor of shape `(batch_size, obs_dim)`.

        Returns
        -------
        values : torch.Tensor
            Tensor containing scalar value estimates of shape `(batch_size, 1)`.

        Examples
        --------
        >>> critic = Critic(32)
        >>> obs = torch.randn(2, 32)
        >>> critic(obs).shape
        torch.Size([2, 1])
        """
        return self.net(obs)


class R_MAPPO_Policy:
    """
    MAPPO policy wrapper containing actor and critic networks.

    This class bundles together the policy actor, value critic, and their
    corresponding optimizers. It also provides helper methods for action
    evaluation and action sampling.

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation vector.
    action_dim : int
        Number of discrete actions.
    device : torch.device, optional
        Torch device on which the networks are allocated,
        by default torch.device("cpu").
    lr : float, optional
        Learning rate for both actor and critic optimizers,
        by default 5e-4.

    Returns
    -------
    R_MAPPO_Policy
        A policy wrapper containing actor, critic, and optimizers.

    Examples
    --------
    >>> policy = R_MAPPO_Policy(obs_dim=128, action_dim=7)
    >>> policy.obs_dim
    128
    >>> policy.action_dim
    7
    """

    def __init__(self, obs_dim: int, action_dim: int, device=torch.device("cpu"), lr: float = 5e-4):
        """
        Initializes the MAPPO policy wrapper.

        Parameters
        ----------
        self : R_MAPPO_Policy
            The policy wrapper instance.
        obs_dim : int
            Dimension of the observation vector.
        action_dim : int
            Number of discrete actions.
        device : torch.device, optional
            Torch device on which the networks are created,
            by default torch.device("cpu").
        lr : float, optional
            Learning rate for the Adam optimizers, by default 5e-4.

        Returns
        -------
        None

        Examples
        --------
        >>> policy = R_MAPPO_Policy(64, 5)
        >>> isinstance(policy.actor, Actor)
        True
        >>> isinstance(policy.critic, Critic)
        True
        """
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def evaluate_actions(self, share_obs_batch, obs_batch, rnn_states_batch,
                        rnn_states_critic_batch, actions_batch, masks_batch,
                        available_actions_batch, active_masks_batch):
        """
        Evaluates actions using the actor and critic networks.

        The critic predicts values from the shared observations, while the actor
        computes action logits from the local observations. A categorical action
        distribution is then used to compute log-probabilities and entropy for
        the provided actions.

        Parameters
        ----------
        self : R_MAPPO_Policy
            The policy wrapper instance.
        share_obs_batch : np.ndarray or torch.Tensor
            Shared observations used by the critic, typically for a centralized
            value function.
        obs_batch : np.ndarray or torch.Tensor
            Agent-specific observations used by the actor.
        rnn_states_batch : np.ndarray or torch.Tensor
            RNN states for the actor. Included for interface compatibility.
        rnn_states_critic_batch : np.ndarray or torch.Tensor
            RNN states for the critic. Included for interface compatibility.
        actions_batch : np.ndarray or torch.Tensor
            Actions whose log-probabilities should be evaluated.
        masks_batch : np.ndarray or torch.Tensor
            Masks for handling sequence boundaries in recurrent setups.
        available_actions_batch : np.ndarray or torch.Tensor
            Mask of currently legal actions. Included for interface compatibility.
        active_masks_batch : np.ndarray or torch.Tensor
            Mask indicating which agents are active. Included for interface compatibility.

        Returns
        -------
        values : torch.Tensor
            Critic value predictions.
        action_log_probs : torch.Tensor
            Log-probabilities of the provided actions.
        dist_entropy : torch.Tensor
            Mean entropy of the action distribution.

        Examples
        --------
        >>> policy = R_MAPPO_Policy(32, 4)
        >>> share_obs = torch.randn(3, 32)
        >>> obs = torch.randn(3, 32)
        >>> actions = torch.randint(0, 4, (3, 1))
        >>> values, log_probs, entropy = policy.evaluate_actions(
        ...     share_obs, obs, None, None, actions, None, None, None
        ... )
        >>> values.shape
        torch.Size([3, 1])
        >>> log_probs.shape
        torch.Size([3, 1])
        """
        # Convert numpy arrays to tensors
        if isinstance(share_obs_batch, np.ndarray):
            share_obs_batch = torch.FloatTensor(share_obs_batch).to(self.device)
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        if isinstance(actions_batch, np.ndarray):
            actions_batch = torch.LongTensor(actions_batch).to(self.device)

        # Get value predictions from critic
        values = self.critic(share_obs_batch)

        # Get action logits from actor
        action_logits = self.actor(obs_batch)

        # Create action distribution (simplified - use categorical)
        action_dist = torch.distributions.Categorical(logits=action_logits)

        # Evaluate given actions
        action_log_probs = action_dist.log_prob(actions_batch.squeeze(-1))
        dist_entropy = action_dist.entropy().mean()

        return values, action_log_probs.unsqueeze(-1), dist_entropy

    def get_action(self, obs):
        """
        Samples an action from the current policy.

        This method performs a forward pass through the actor network,
        constructs a categorical action distribution, and samples one action
        per input observation.

        Parameters
        ----------
        self : R_MAPPO_Policy
            The policy wrapper instance.
        obs : torch.Tensor
            Observation tensor used to sample actions.

        Returns
        -------
        action : np.ndarray
            Sampled action indices as a numpy array.
        action_log_prob : np.ndarray
            Log-probabilities of the sampled actions as a numpy array.

        Examples
        --------
        >>> policy = R_MAPPO_Policy(16, 3)
        >>> obs = torch.randn(2, 16)
        >>> action, log_prob = policy.get_action(obs)
        >>> action.shape
        (2,)
        """
        with torch.no_grad():
            action_logits = self.actor(obs)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
        return action.cpu().numpy(), action_log_prob.cpu().numpy()