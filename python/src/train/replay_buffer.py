import numpy as np
import torch


class SharedReplayBuffer:
    """
    Shared replay buffer for MAPPO training.
    Stores trajectories and provides generators for mini-batches.
    """
    def __init__(self, num_agents: int, obs_shape, action_shape, buffer_size: int, 
                 device=torch.device("cpu")):
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
        self.available_actions = np.ones((buffer_size + 1, num_agents, 1), dtype=np.float32)  # Placeholder
        
        self.step = 0
        
    def insert(self, share_obs, obs, actions, action_log_probs, value_preds, 
               rewards, masks, active_masks, available_actions=None):
        """Insert experience into buffer"""
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
        Compute returns and GAE advantages.
        
        :param next_value: Value at end of trajectory
        :param gamma: Discount factor
        :param gae_lambda: GAE lambda parameter
        """
        self.value_preds[-1] = next_value
        gae = 0
        advantages = np.zeros_like(self.rewards)
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.masks[step + 1]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - self.masks[step + 1]
                next_value_step = self.value_preds[step + 1]
            
            delta = self.rewards[step] + gamma * next_value_step * next_non_terminal - self.value_preds[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            
        self.returns[:-1] = advantages + self.value_preds[:-1]
        
    def feed_forward_generator(self, advantages, num_mini_batch):
        """
        Generate mini-batches for feed-forward networks.
        
        :param advantages: Computed advantages
        :param num_mini_batch: Number of mini-batches
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
            
            yield (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, 
                   actions_batch, value_preds_batch, return_batch, masks_batch, 
                   active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch)
    
    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """Generate mini-batches for recurrent networks (not implemented)"""
        return self.feed_forward_generator(advantages, num_mini_batch)
    
    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """Generate mini-batches for naive recurrent (not implemented)"""
        return self.feed_forward_generator(advantages, num_mini_batch)
    
    def reset(self):
        """Reset buffer"""
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
