"""
Quick integration test for R_MAPPO.
Tests R_MAPPO with simulated data without requiring the full environment.
"""

import numpy as np
import torch
from r_mappo import R_MAPPO
from replay_buffer import SharedReplayBuffer
from policy import R_MAPPO_Policy


class Args:
    """Training arguments"""
    def __init__(self):
        self.clip_param = 0.2
        self.ppo_epoch = 5
        self.num_mini_batch = 4
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


def test_r_mappo_basic():
    """Test basic R_MAPPO functionality"""
    print("=" * 60)
    print("TEST 1: Basic R_MAPPO Initialization")
    print("=" * 60)
    
    device = torch.device("cpu")
    args = Args()
    
    # Create policy
    obs_dim = 128
    action_dim = 7
    policy = R_MAPPO_Policy(obs_dim, action_dim, device=device)
    
    # Create trainer
    trainer = R_MAPPO(args, policy, device=device)
    
    print("✓ R_MAPPO initialized successfully")
    print(f"  - Actor: {policy.actor}")
    print(f"  - Critic: {policy.critic}")
    print(f"  - Device: {device}")
    print()


def test_value_loss():
    """Test value loss calculation"""
    print("=" * 60)
    print("TEST 2: Value Loss Calculation")
    print("=" * 60)
    
    device = torch.device("cpu")
    args = Args()
    policy = R_MAPPO_Policy(64, 4, device=device)
    trainer = R_MAPPO(args, policy, device=device)
    
    # Create sample batch
    batch_size = 8
    values = torch.randn(batch_size, 1)
    value_preds_batch = torch.randn(batch_size, 1)
    return_batch = torch.randn(batch_size, 1)
    active_masks_batch = torch.ones(batch_size, 1)
    
    # Calculate loss
    value_loss = trainer.cal_value_loss(
        values, value_preds_batch, return_batch, active_masks_batch
    )
    
    print(f"✓ Value loss calculated: {value_loss.item():.6f}")
    print()


def test_ppo_update():
    """Test single PPO update step"""
    print("=" * 60)
    print("TEST 3: Single PPO Update Step")
    print("=" * 60)
    
    device = torch.device("cpu")
    args = Args()
    policy = R_MAPPO_Policy(64, 4, device=device)
    trainer = R_MAPPO(args, policy, device=device)
    
    # Create sample batch
    batch_size = 8
    obs_dim = 64
    action_dim = 4
    
    sample = (
        np.random.randn(batch_size, obs_dim),      # share_obs_batch
        np.random.randn(batch_size, obs_dim),      # obs_batch
        np.zeros((batch_size, 1)),                  # rnn_states_batch
        np.zeros((batch_size, 1)),                  # rnn_states_critic_batch
        np.random.randint(0, action_dim, (batch_size, 1)),  # actions_batch
        np.random.randn(batch_size, 1),            # value_preds_batch
        np.random.randn(batch_size, 1),            # return_batch
        np.ones((batch_size, 1)),                  # masks_batch
        np.ones((batch_size, 1)),                  # active_masks_batch
        np.random.randn(batch_size, 1),            # old_action_log_probs_batch
        np.random.randn(batch_size, 1),            # adv_targ
        np.ones((batch_size, 1)),                  # available_actions_batch
    )
    
    # Perform update
    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = trainer.ppo_update(sample)
    
    print("✓ PPO update completed successfully")
    print(f"  - Value Loss:      {value_loss.item():.6f}")
    print(f"  - Policy Loss:     {policy_loss.item():.6f}")
    print(f"  - Entropy:         {dist_entropy.item():.6f}")
    print(f"  - Actor Grad Norm: {actor_grad_norm:.6f}")
    print(f"  - Critic Grad Norm:{critic_grad_norm:.6f}")
    print(f"  - Importance Weights Mean: {imp_weights.mean().item():.6f}")
    print()


def test_training_loop():
    """Test full training loop with replay buffer"""
    print("=" * 60)
    print("TEST 4: Full Training Loop (2 epochs)")
    print("=" * 60)
    
    device = torch.device("cpu")
    args = Args()
    args.ppo_epoch = 2  # Quick test
    
    # Setup
    num_agents = 2
    obs_dim = 64
    action_dim = 4
    buffer_size = 16
    
    policy = R_MAPPO_Policy(obs_dim, action_dim, device=device)
    trainer = R_MAPPO(args, policy, device=device)
    buffer = SharedReplayBuffer(
        num_agents=num_agents,
        obs_shape=(obs_dim,),
        action_shape=(1,),
        buffer_size=buffer_size,
        device=device
    )
    
    # Populate buffer with random data
    print("Populating buffer with random transitions...")
    for step in range(buffer_size):
        share_obs = np.random.randn(obs_dim).astype(np.float32)
        obs = np.random.randn(num_agents, obs_dim).astype(np.float32)
        actions = np.random.randint(0, action_dim, (num_agents, 1)).astype(np.float32)
        action_log_probs = np.random.randn(num_agents, 1).astype(np.float32)
        values = np.random.randn(1, 1).astype(np.float32)
        rewards = np.random.randn(num_agents, 1).astype(np.float32)
        masks = np.ones((num_agents, 1))
        active_masks = np.ones((num_agents, 1))
        
        buffer.insert(
            share_obs=share_obs,
            obs=obs,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards,
            masks=masks,
            active_masks=active_masks,
        )
    
    print("✓ Buffer populated")
    
    # Compute returns
    print("Computing returns and advantages...")
    with torch.no_grad():
        next_value = torch.randn(1, 1).numpy()
    buffer.compute_returns(next_value, gamma=0.99)
    print("✓ Returns computed")
    
    # Advantages
    advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
    
    # Train
    print("Running training loop...")
    trainer.prep_training()
    train_info = trainer.train(buffer, update_actor=True)
    
    print("✓ Training completed")
    print(f"\nTraining Statistics (averaged over {args.ppo_epoch} epochs):")
    for key, val in train_info.items():
        print(f"  - {key:20s}: {val:.6f}")
    print()


def test_network_modes():
    """Test prep_training and prep_rollout"""
    print("=" * 60)
    print("TEST 5: Network Mode Switching")
    print("=" * 60)
    
    device = torch.device("cpu")
    args = Args()
    policy = R_MAPPO_Policy(64, 4, device=device)
    trainer = R_MAPPO(args, policy, device=device)
    
    # Test rollout mode
    trainer.prep_rollout()
    assert not policy.actor.training, "Actor should be in eval mode"
    assert not policy.critic.training, "Critic should be in eval mode"
    print("✓ Rollout mode: networks in eval mode")
    
    # Test training mode
    trainer.prep_training()
    assert policy.actor.training, "Actor should be in train mode"
    assert policy.critic.training, "Critic should be in train mode"
    print("✓ Training mode: networks in train mode")
    print()


def test_gradient_clipping():
    """Test gradient clipping"""
    print("=" * 60)
    print("TEST 6: Gradient Clipping")
    print("=" * 60)
    
    device = torch.device("cpu")
    args = Args()
    args.use_max_grad_norm = True
    
    policy = R_MAPPO_Policy(64, 4, device=device)
    trainer = R_MAPPO(args, policy, device=device)
    
    # Create sample with large values to trigger large gradients
    batch_size = 8
    obs_dim = 64
    action_dim = 4
    
    sample = (
        np.random.randn(batch_size, obs_dim) * 10,  # Large values
        np.random.randn(batch_size, obs_dim) * 10,
        np.zeros((batch_size, 1)),
        np.zeros((batch_size, 1)),
        np.random.randint(0, action_dim, (batch_size, 1)),
        np.random.randn(batch_size, 1) * 100,
        np.random.randn(batch_size, 1) * 100,
        np.ones((batch_size, 1)),
        np.ones((batch_size, 1)),
        np.random.randn(batch_size, 1),
        np.random.randn(batch_size, 1) * 100,  # Large advantages
        np.ones((batch_size, 1)),
    )
    
    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = trainer.ppo_update(sample)
    
    # Check gradient norms are clipped
    max_grad_norm = trainer.max_grad_norm
    assert actor_grad_norm <= max_grad_norm + 1e-5, f"Actor grad norm {actor_grad_norm} exceeds limit {max_grad_norm}"
    assert critic_grad_norm <= max_grad_norm + 1e-5, f"Critic grad norm {critic_grad_norm} exceeds limit {max_grad_norm}"
    
    print(f"✓ Gradient clipping working")
    print(f"  - Max grad norm limit: {max_grad_norm}")
    print(f"  - Actor grad norm:     {actor_grad_norm:.6f}")
    print(f"  - Critic grad norm:    {critic_grad_norm:.6f}")
    print()


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "R_MAPPO Integration Tests" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        test_r_mappo_basic()
        test_value_loss()
        test_ppo_update()
        test_network_modes()
        test_gradient_clipping()
        test_training_loop()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
