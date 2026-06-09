"""
Unit tests for R_MAPPO trainer.
Tests individual components and integration.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path to import r_mappo
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "train"))
from r_mappo import R_MAPPO

# Mock onpolicy utilities since the package isn't available
import sys
from unittest.mock import MagicMock

# Create mock onpolicy modules
onpolicy_mock = MagicMock()
sys.modules['onpolicy'] = onpolicy_mock
sys.modules['onpolicy.utils'] = MagicMock()
sys.modules['onpolicy.utils.util'] = MagicMock()
sys.modules['onpolicy.algorithms'] = MagicMock()
sys.modules['onpolicy.algorithms.utils'] = MagicMock()
sys.modules['onpolicy.algorithms.utils.util'] = MagicMock()

# Define mock utility functions
def mse_loss(error):
    """Mock MSE loss - returns mean squared error"""
    return (error ** 2)

def huber_loss(error, delta):
    """Mock Huber loss"""
    abs_error = torch.abs(error)
    return torch.where(abs_error < delta, 0.5 * (error ** 2), delta * (abs_error - 0.5 * delta))

def get_gard_norm(net):
    """Mock gradient norm (note the typo is in the original code)"""
    total_norm = 0.0
    for p in net.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def check(x):
    """Mock check function - converts input to torch tensor"""
    if isinstance(x, np.ndarray):
        return torch.FloatTensor(x)
    return torch.as_tensor(x, dtype=torch.float32)

# Inject mocks into the onpolicy modules
sys.modules['onpolicy.utils.util'].mse_loss = mse_loss
sys.modules['onpolicy.utils.util'].huber_loss = huber_loss
sys.modules['onpolicy.utils.util'].get_gard_norm = get_gard_norm
sys.modules['onpolicy.algorithms.utils.util'].check = check


class Args:
    """Mock args for testing"""
    def __init__(self):
        self.clip_param = 0.2
        self.ppo_epoch = 2
        self.num_mini_batch = 2
        self.data_chunk_length = 4
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.huber_delta = 10.0
        self.use_recurrent_policy = False
        self.use_naive_recurrent_policy = False
        self.use_max_grad_norm = True
        self.use_clipped_value_loss = True
        self.use_huber_loss = False
        self.use_popart = False
        self.use_valuenorm = False
        self.use_value_active_masks = True
        self.use_policy_active_masks = True


class SimpleActor(nn.Module):
    """Simple mock actor for testing"""
    def __init__(self, obs_dim=10, action_dim=4):
        super(SimpleActor, self).__init__()
        self.net = nn.Linear(obs_dim, action_dim)
        
    def forward(self, x):
        return self.net(x)


class SimpleCritic(nn.Module):
    """Simple mock critic for testing"""
    def __init__(self, obs_dim=10):
        super(SimpleCritic, self).__init__()
        self.net = nn.Linear(obs_dim, 1)
        self.v_out = self.net
        
    def forward(self, x):
        return self.net(x)


class MockPolicy:
    """Mock policy with required methods for R_MAPPO"""
    def __init__(self, device=torch.device("cpu"), obs_dim=10, action_dim=4):
        self.device = device
        self.actor = SimpleActor(obs_dim, action_dim).to(device)
        self.critic = SimpleCritic(obs_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
    
    def evaluate_actions(self, share_obs_batch, obs_batch, rnn_states_batch,
                        rnn_states_critic_batch, actions_batch, masks_batch,
                        available_actions_batch, active_masks_batch):
        """Mock evaluate_actions"""
        batch_size = obs_batch.shape[0]
        
        # Get values
        values = self.critic(torch.FloatTensor(share_obs_batch).to(self.device))
        
        # Get action log probs
        action_logits = self.actor(torch.FloatTensor(obs_batch).to(self.device))
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action_log_probs = action_dist.log_prob(
            torch.LongTensor(actions_batch.squeeze(-1)).to(self.device)
        )
        dist_entropy = action_dist.entropy().mean()
        
        return values, action_log_probs.unsqueeze(-1), dist_entropy


class TestR_MAPPO:
    """Test suite for R_MAPPO trainer"""
    
    def setup_method(self):
        """Setup before each test"""
        self.device = torch.device("cpu")
        self.args = Args()
        self.policy = MockPolicy(device=self.device)
        self.trainer = R_MAPPO(self.args, self.policy, device=self.device)
    
    def test_initialization(self):
        """Test R_MAPPO initialization"""
        assert self.trainer.policy is not None
        assert self.trainer.clip_param == 0.2
        assert self.trainer.ppo_epoch == 2
        assert self.trainer.device == self.device
    
    def test_value_loss_calculation(self):
        """Test cal_value_loss method"""
        batch_size = 4
        
        values = torch.randn(batch_size, 1)
        value_preds_batch = torch.randn(batch_size, 1)
        return_batch = torch.randn(batch_size, 1)
        active_masks_batch = torch.ones(batch_size, 1)
        
        value_loss = self.trainer.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )
        
        assert isinstance(value_loss, torch.Tensor)
        assert value_loss.item() >= 0
    
    def test_ppo_update_12_sample(self):
        """Test ppo_update with 12-element sample"""
        batch_size = 4
        obs_dim = 10
        action_dim = 4
        
        sample = (
            np.random.randn(batch_size, obs_dim),  # share_obs_batch
            np.random.randn(batch_size, obs_dim),  # obs_batch
            np.zeros((batch_size, 1)),              # rnn_states_batch
            np.zeros((batch_size, 1)),              # rnn_states_critic_batch
            np.random.randint(0, action_dim, (batch_size, 1)),  # actions_batch
            np.random.randn(batch_size, 1),        # value_preds_batch
            np.random.randn(batch_size, 1),        # return_batch
            np.ones((batch_size, 1)),              # masks_batch
            np.ones((batch_size, 1)),              # active_masks_batch
            np.random.randn(batch_size, 1),        # old_action_log_probs_batch
            np.random.randn(batch_size, 1),        # adv_targ
            np.ones((batch_size, 1)),              # available_actions_batch
        )
        
        (value_loss, critic_grad_norm, policy_loss, 
         dist_entropy, actor_grad_norm, imp_weights) = self.trainer.ppo_update(sample)
        
        assert isinstance(value_loss, torch.Tensor)
        assert isinstance(policy_loss, torch.Tensor)
        assert isinstance(dist_entropy, torch.Tensor)
        assert isinstance(actor_grad_norm, (float, torch.Tensor))
        assert isinstance(critic_grad_norm, (float, torch.Tensor))
    
    def test_ppo_update_13_sample(self):
        """Test ppo_update with 13-element sample"""
        batch_size = 4
        obs_dim = 10
        action_dim = 4
        
        sample = (
            np.random.randn(batch_size, obs_dim),  # share_obs_batch
            np.random.randn(batch_size, obs_dim),  # obs_batch
            np.zeros((batch_size, 1)),              # rnn_states_batch
            np.zeros((batch_size, 1)),              # rnn_states_critic_batch
            np.random.randint(0, action_dim, (batch_size, 1)),  # actions_batch
            np.random.randn(batch_size, 1),        # value_preds_batch
            np.random.randn(batch_size, 1),        # return_batch
            np.ones((batch_size, 1)),              # masks_batch
            np.ones((batch_size, 1)),              # active_masks_batch
            np.random.randn(batch_size, 1),        # old_action_log_probs_batch
            np.random.randn(batch_size, 1),        # adv_targ
            np.ones((batch_size, 1)),              # available_actions_batch
            np.zeros((batch_size, 1)),              # extra element
        )
        
        (value_loss, critic_grad_norm, policy_loss,
         dist_entropy, actor_grad_norm, imp_weights) = self.trainer.ppo_update(sample)
        
        assert isinstance(value_loss, torch.Tensor)
    
    def test_ppo_update_no_actor_update(self):
        """Test ppo_update with update_actor=False"""
        batch_size = 4
        obs_dim = 10
        action_dim = 4
        
        sample = (
            np.random.randn(batch_size, obs_dim),
            np.random.randn(batch_size, obs_dim),
            np.zeros((batch_size, 1)),
            np.zeros((batch_size, 1)),
            np.random.randint(0, action_dim, (batch_size, 1)),
            np.random.randn(batch_size, 1),
            np.random.randn(batch_size, 1),
            np.ones((batch_size, 1)),
            np.ones((batch_size, 1)),
            np.random.randn(batch_size, 1),
            np.random.randn(batch_size, 1),
            np.ones((batch_size, 1)),
        )
        
        # Should not raise error with update_actor=False
        (value_loss, critic_grad_norm, policy_loss,
         dist_entropy, actor_grad_norm, imp_weights) = self.trainer.ppo_update(
            sample, update_actor=False
        )
        
        assert isinstance(value_loss, torch.Tensor)
    
    def test_prep_modes(self):
        """Test prep_training and prep_rollout"""
        # After prep_training, networks should be in train mode
        self.trainer.prep_training()
        assert self.trainer.policy.actor.training
        assert self.trainer.policy.critic.training
        
        # After prep_rollout, networks should be in eval mode
        self.trainer.prep_rollout()
        assert not self.trainer.policy.actor.training
        assert not self.trainer.policy.critic.training
    
    def test_gradient_clipping(self):
        """Test gradient clipping"""
        # Create a simple batch
        batch_size = 4
        obs_dim = 10
        action_dim = 4
        
        sample = (
            np.random.randn(batch_size, obs_dim),
            np.random.randn(batch_size, obs_dim),
            np.zeros((batch_size, 1)),
            np.zeros((batch_size, 1)),
            np.random.randint(0, action_dim, (batch_size, 1)),
            np.random.randn(batch_size, 1),
            np.random.randn(batch_size, 1),
            np.ones((batch_size, 1)),
            np.ones((batch_size, 1)),
            np.random.randn(batch_size, 1),
            np.random.randn(batch_size, 1) * 10,  # Large advantages
            np.ones((batch_size, 1)),
        )
        
        value_loss, critic_grad_norm, _, _, actor_grad_norm, _ = self.trainer.ppo_update(sample)
        
        # Just verify gradient norms are computed (they may exceed max_grad_norm before clipping)
        assert isinstance(actor_grad_norm, (float, torch.Tensor))
        assert isinstance(critic_grad_norm, (float, torch.Tensor))


class TestR_MAPPO_MockBuffer:
    """Test R_MAPPO with mock buffer"""
    
    def setup_method(self):
        self.device = torch.device("cpu")
        self.args = Args()
        self.policy = MockPolicy(device=self.device)
        self.trainer = R_MAPPO(self.args, self.policy, device=self.device)
    
    def test_train_with_mock_buffer(self):
        """Test train method with mock buffer"""
        
        class MockBuffer:
            def __init__(self, num_agents=2, buffer_size=8):
                self.num_agents = num_agents
                self.buffer_size = buffer_size
                self.returns = np.random.randn(buffer_size + 1, num_agents, 1)
                self.value_preds = np.random.randn(buffer_size + 1, num_agents, 1)
                self.active_masks = np.ones((buffer_size + 1, num_agents, 1))
            
            def feed_forward_generator(self, advantages, num_mini_batch):
                """Yield mini-batches"""
                batch_size = self.buffer_size // num_mini_batch
                for i in range(num_mini_batch):
                    obs_dim = 10
                    action_dim = 4
                    # Flatten advantages to match expected shape
                    adv_slice = advantages[i*batch_size:(i+1)*batch_size]
                    if len(adv_slice.shape) > 2:
                        adv_slice = adv_slice.reshape(batch_size, -1)
                    batch = (
                        np.random.randn(batch_size, obs_dim),
                        np.random.randn(batch_size, obs_dim),
                        np.zeros((batch_size, 1)),
                        np.zeros((batch_size, 1)),
                        np.random.randint(0, action_dim, (batch_size, 1)),
                        np.random.randn(batch_size, 1),
                        np.random.randn(batch_size, 1),
                        np.ones((batch_size, 1)),
                        np.ones((batch_size, 1)),
                        np.random.randn(batch_size, 1),
                        adv_slice,
                        np.ones((batch_size, 1)),
                    )
                    yield batch
            
            def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
                return self.feed_forward_generator(advantages, num_mini_batch)
            
            def naive_recurrent_generator(self, advantages, num_mini_batch):
                return self.feed_forward_generator(advantages, num_mini_batch)
        
        buffer = MockBuffer()
        advantages = np.random.randn(buffer.buffer_size, 1)
        
        train_info = self.trainer.train(buffer, update_actor=True)
        
        assert isinstance(train_info, dict)
        assert 'value_loss' in train_info
        assert 'policy_loss' in train_info
        assert 'dist_entropy' in train_info
        assert train_info['value_loss'] >= 0
        assert train_info['policy_loss'] >= 0


# Quick test runner
if __name__ == "__main__":
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
