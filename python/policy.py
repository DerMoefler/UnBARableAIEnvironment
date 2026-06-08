import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    """Actor network for policy"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """Critic network for value function"""
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.v_out = self  # for POPart compatibility
        
    def forward(self, obs):
        return self.net(obs)


class R_MAPPO_Policy:
    """
    MAPPO Policy wrapper with actor and critic networks.
    
    :param obs_dim: Observation dimension
    :param action_dim: Action dimension  
    :param device: torch device
    :param lr: Learning rate
    """
    def __init__(self, obs_dim: int, action_dim: int, device=torch.device("cpu"), lr: float = 5e-4):
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
        Evaluate actions using actor and critic networks.
        
        :param share_obs_batch: Shared observations (for centralized critic)
        :param obs_batch: Agent observations
        :param rnn_states_batch: RNN states for actor
        :param rnn_states_critic_batch: RNN states for critic
        :param actions_batch: Actions to evaluate
        :param masks_batch: Masks for RNN
        :param available_actions_batch: Available actions mask
        :param active_masks_batch: Active agents mask
        
        :return values: (torch.Tensor) value predictions
        :return action_log_probs: (torch.Tensor) action log probabilities
        :return dist_entropy: (torch.Tensor) action entropy
        """
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
        """Get action from policy"""
        with torch.no_grad():
            action_logits = self.actor(obs)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
        return action.cpu().numpy(), action_log_prob.cpu().numpy()
