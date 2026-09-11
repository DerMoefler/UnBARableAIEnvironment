"""
Quick integration test for R_MAPPO.
Tests R_MAPPO with simulated data without requiring the full environment.

This version explicitly tests:
- basic initialization
- value loss
- single-agent style PPO update
- network mode switching
- gradient clipping path
- multi-agent PPO update
- multi-agent full training loop
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "train"))

import numpy as np
import torch
from unittest.mock import MagicMock

# -------------------------------------------------------------------
# Mock onpolicy utilities since the package isn't available
# -------------------------------------------------------------------
onpolicy_mock = MagicMock()
sys.modules["onpolicy"] = onpolicy_mock
sys.modules["onpolicy.utils"] = MagicMock()
sys.modules["onpolicy.utils.util"] = MagicMock()
sys.modules["onpolicy.algorithms"] = MagicMock()
sys.modules["onpolicy.algorithms.utils"] = MagicMock()
sys.modules["onpolicy.algorithms.utils.util"] = MagicMock()


# -------------------------------------------------------------------
# Mock utility functions
# -------------------------------------------------------------------
def mse_loss(error):
    """Mock MSE loss - returns elementwise squared error."""
    return error ** 2


def huber_loss(error, delta):
    """Mock Huber loss."""
    abs_error = torch.abs(error)
    return torch.where(
        abs_error < delta,
        0.5 * (error ** 2),
        delta * (abs_error - 0.5 * delta),
    )


def get_gard_norm(net):
    """Mock gradient norm (typo preserved from original dependency name)."""
    total_norm = 0.0
    params = net if hasattr(net, "__iter__") else net.parameters()
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def check(x):
    """Mock check function - converts input to torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.FloatTensor(x)
    return torch.as_tensor(x, dtype=torch.float32)


# Inject mocks into the onpolicy modules
sys.modules["onpolicy.utils.util"].mse_loss = mse_loss
sys.modules["onpolicy.utils.util"].huber_loss = huber_loss
sys.modules["onpolicy.utils.util"].get_gard_norm = get_gard_norm
sys.modules["onpolicy.algorithms.utils.util"].check = check


# -------------------------------------------------------------------
# Imports from your project
# -------------------------------------------------------------------
from r_mappo import R_MAPPO
from replay_buffer import SharedReplayBuffer
from policy import R_MAPPO_Policy


# -------------------------------------------------------------------
# Test config
# -------------------------------------------------------------------
class Args:
    """Training arguments."""

    def __init__(self):
        """
        Initializes the quick-test training configuration.

        Parameters
        ----------
        self : Args
            The configuration instance.

        Returns
        -------
        None
        """
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


# -------------------------------------------------------------------
# Optional helpers
# -------------------------------------------------------------------
def print_sample_shapes(sample, prefix="sample"):
    """Print shapes of a PPO sample tuple for debugging."""
    names = [
        "share_obs_batch",
        "obs_batch",
        "rnn_states_batch",
        "rnn_states_critic_batch",
        "actions_batch",
        "value_preds_batch",
        "return_batch",
        "masks_batch",
        "active_masks_batch",
        "old_action_log_probs_batch",
        "adv_targ",
        "available_actions_batch",
    ]

    print(f"{prefix} shapes:")
    for name, item in zip(names, sample):
        if item is None:
            print(f"  {name}: None")
        else:
            shape = getattr(item, "shape", None)
            print(f"  {name}: {shape}")


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
def test_r_mappo_basic():
    """Test basic R_MAPPO functionality."""
    print("=" * 60)
    print("TEST 1: Basic R_MAPPO Initialization")
    print("=" * 60)

    device = torch.device("cpu")
    args = Args()

    obs_dim = 128
    action_dim = 5  # 0:north, 1:south, 2:east, 3:west, 4:attack
    policy = R_MAPPO_Policy(obs_dim, action_dim, device=device)
    trainer = R_MAPPO(args, policy, device=device)

    print("✓ R_MAPPO initialized successfully")
    print(f"  - Actor: {policy.actor}")
    print(f"  - Critic: {policy.critic}")
    print(f"  - Device: {device}")
    print()


def test_value_loss():
    """Test value loss calculation."""
    print("=" * 60)
    print("TEST 2: Value Loss Calculation")
    print("=" * 60)

    device = torch.device("cpu")
    args = Args()
    policy = R_MAPPO_Policy(64, 4, device=device)
    trainer = R_MAPPO(args, policy, device=device)

    batch_size = 8
    values = torch.randn(batch_size, 1)
    value_preds_batch = torch.randn(batch_size, 1)
    return_batch = torch.randn(batch_size, 1)
    active_masks_batch = torch.ones(batch_size, 1)

    value_loss = trainer.cal_value_loss(
        values, value_preds_batch, return_batch, active_masks_batch
    )

    print(f"✓ Value loss calculated: {value_loss.item():.6f}")
    print()


def test_ppo_update():
    """Test single PPO update step."""
    print("=" * 60)
    print("TEST 3: Single PPO Update Step")
    print("=" * 60)

    device = torch.device("cpu")
    args = Args()
    policy = R_MAPPO_Policy(64, 5, device=device)  # 0:north, 1:south, 2:east, 3:west, 4:attack
    trainer = R_MAPPO(args, policy, device=device)

    batch_size = 8
    obs_dim = 64
    action_dim = 5

    sample = (
        np.random.randn(batch_size, obs_dim).astype(np.float32),      # share_obs_batch
        np.random.randn(batch_size, obs_dim).astype(np.float32),      # obs_batch
        np.zeros((batch_size, 1), dtype=np.float32),                  # rnn_states_batch
        np.zeros((batch_size, 1), dtype=np.float32),                  # rnn_states_critic_batch
        np.column_stack((
            np.random.randint(0, action_dim, batch_size),
            np.random.randint(0, 3, batch_size),
        )).astype(np.float32),  # actions_batch: [action_type, target_id]
        np.random.randn(batch_size, 1).astype(np.float32),            # value_preds_batch
        np.random.randn(batch_size, 1).astype(np.float32),            # return_batch
        np.ones((batch_size, 1), dtype=np.float32),                   # masks_batch
        np.ones((batch_size, 1), dtype=np.float32),                   # active_masks_batch
        np.random.randn(batch_size, 1).astype(np.float32),            # old_action_log_probs_batch
        np.random.randn(batch_size, 1).astype(np.float32),            # adv_targ
        np.ones((batch_size, 1), dtype=np.float32),                   # available_actions_batch
    )

    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, decoded_actions = trainer.ppo_update(sample)

    print("✓ PPO update completed successfully")
    print(f"  - Value Loss:      {value_loss.item():.6f}")
    print(f"  - Policy Loss:     {policy_loss.item():.6f}")
    print(f"  - Entropy:         {dist_entropy.item():.6f}")
    print(f"  - Actor Grad Norm: {float(actor_grad_norm):.6f}")
    print(f"  - Critic Grad Norm:{float(critic_grad_norm):.6f}")
    print(f"  - Importance Weights Mean: {imp_weights.mean().item():.6f}")
    print(f"  - Decoded Actions: {len(decoded_actions)} actions")
    print()


def test_network_modes():
    """Test prep_training and prep_rollout."""
    print("=" * 60)
    print("TEST 4: Network Mode Switching")
    print("=" * 60)

    device = torch.device("cpu")
    args = Args()
    policy = R_MAPPO_Policy(64, 4, device=device)
    trainer = R_MAPPO(args, policy, device=device)

    trainer.prep_rollout()
    assert not policy.actor.training, "Actor should be in eval mode"
    assert not policy.critic.training, "Critic should be in eval mode"
    print("✓ Rollout mode: networks in eval mode")

    trainer.prep_training()
    assert policy.actor.training, "Actor should be in train mode"
    assert policy.critic.training, "Critic should be in train mode"
    print("✓ Training mode: networks in train mode")
    print()


def test_gradient_clipping():
    """
    Test gradient clipping execution path.

    Important:
    torch.nn.utils.clip_grad_norm_ usually returns the gradient norm
    BEFORE clipping, not after clipping. So we only verify the path executes
    and norms are reported.
    """
    print("=" * 60)
    print("TEST 5: Gradient Clipping")
    print("=" * 60)

    device = torch.device("cpu")
    args = Args()
    args.use_max_grad_norm = True

    policy = R_MAPPO_Policy(64, 4, device=device)
    trainer = R_MAPPO(args, policy, device=device)

    batch_size = 8
    obs_dim = 64
    action_dim = 4

    sample = (
        (np.random.randn(batch_size, obs_dim) * 10).astype(np.float32),
        (np.random.randn(batch_size, obs_dim) * 10).astype(np.float32),
        np.zeros((batch_size, 1), dtype=np.float32),
        np.zeros((batch_size, 1), dtype=np.float32),
        np.column_stack((
            np.random.randint(0, action_dim, batch_size),
            np.random.randint(0, 3, batch_size),
        )).astype(np.float32),
        (np.random.randn(batch_size, 1) * 100).astype(np.float32),
        (np.random.randn(batch_size, 1) * 100).astype(np.float32),
        np.ones((batch_size, 1), dtype=np.float32),
        np.ones((batch_size, 1), dtype=np.float32),
        np.random.randn(batch_size, 1).astype(np.float32),
        (np.random.randn(batch_size, 1) * 100).astype(np.float32),
        np.ones((batch_size, 1), dtype=np.float32),
    )

    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, decoded_actions = trainer.ppo_update(sample)

    assert isinstance(actor_grad_norm, (float, torch.Tensor)), "Actor grad norm should be computed"
    assert isinstance(critic_grad_norm, (float, torch.Tensor)), "Critic grad norm should be computed"

    print("✓ Gradient clipping path executed")
    print(f"  - Max grad norm limit: {trainer.max_grad_norm}")
    print(f"  - Actor grad norm (reported):  {float(actor_grad_norm):.6f}")
    print(f"  - Critic grad norm (reported): {float(critic_grad_norm):.6f}")
    print(f"  - Decoded Actions: {len(decoded_actions)} actions")
    print()


def test_multi_agent_ppo_update():
    """Test a single PPO update with true multi-agent shaped inputs."""
    print("=" * 60)
    print("TEST 6: Multi-Agent PPO Update Step")
    print("=" * 60)

    device = torch.device("cpu")
    args = Args()
    policy = R_MAPPO_Policy(64, 5, device=device)  # 0:north, 1:south, 2:east, 3:west, 4:attack
    trainer = R_MAPPO(args, policy, device=device)

    batch_size = 8
    num_agents = 2
    obs_dim = 64
    action_dim = 5

    # share_obs is global per timestep: (B, D)
    # obs is per-agent: (B, A, D)
    sample = (
        np.random.randn(batch_size, obs_dim).astype(np.float32),               # share_obs_batch
        np.random.randn(batch_size, num_agents, obs_dim).astype(np.float32),   # obs_batch
        np.zeros((batch_size, num_agents, 1), dtype=np.float32),               # rnn_states_batch
        np.zeros((batch_size, num_agents, 1), dtype=np.float32),               # rnn_states_critic_batch
        np.stack((
            np.random.randint(0, action_dim, (batch_size, num_agents)),
            np.random.randint(0, 3, (batch_size, num_agents)),
        ), axis=-1).astype(np.float32),  # actions_batch: [action_type, target_id]
        np.random.randn(batch_size, num_agents, 1).astype(np.float32),         # value_preds_batch
        np.random.randn(batch_size, num_agents, 1).astype(np.float32),         # return_batch
        np.ones((batch_size, num_agents, 1), dtype=np.float32),                # masks_batch
        np.ones((batch_size, num_agents, 1), dtype=np.float32),                # active_masks_batch
        np.random.randn(batch_size, num_agents, 1).astype(np.float32),         # old_action_log_probs_batch
        np.random.randn(batch_size, num_agents, 1).astype(np.float32),         # adv_targ
        np.ones((batch_size, num_agents, 1), dtype=np.float32),                # available_actions_batch
    )

    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, decoded_actions = trainer.ppo_update(sample)

    print("✓ Multi-agent PPO update completed successfully")
    print(f"  - Value Loss:      {value_loss.item():.6f}")
    print(f"  - Policy Loss:     {policy_loss.item():.6f}")
    print(f"  - Entropy:         {dist_entropy.item():.6f}")
    print(f"  - Actor Grad Norm: {float(actor_grad_norm):.6f}")
    print(f"  - Critic Grad Norm:{float(critic_grad_norm):.6f}")
    print(f"  - Importance Weights Mean: {imp_weights.mean().item():.6f}")
    print(f"  - Decoded Actions: {len(decoded_actions)} actions")
    print()


def test_training_loop_multi_agent():
    """Test full multi-agent training loop with replay buffer."""
    print("=" * 60)
    print("TEST 7: Full Multi-Agent Training Loop (2 agents, 2 epochs)")
    print("=" * 60)

    device = torch.device("cpu")
    args = Args()
    args.ppo_epoch = 2  # Quick test

    num_agents = 2
    obs_dim = 64
    action_dim = 5  # 0:north, 1:south, 2:east, 3:west, 4:attack
    buffer_size = 16

    policy = R_MAPPO_Policy(obs_dim, action_dim, device=device)
    trainer = R_MAPPO(args, policy, device=device)
    buffer = SharedReplayBuffer(
        num_agents=num_agents,
        obs_shape=(obs_dim,),
        action_shape=(2,),
        action_dim=action_dim,
        buffer_size=buffer_size,
        device=device,
    )

    print("Expected buffer insert shapes:")
    print(f"  share_obs:        {buffer.share_obs[0].shape}")
    print(f"  obs:              {buffer.obs[0].shape}")
    print(f"  actions:          {buffer.actions[0].shape}")
    print(f"  action_log_probs: {buffer.action_log_probs[0].shape}")
    print(f"  value_preds:      {buffer.value_preds[0].shape}")
    print(f"  rewards:          {buffer.rewards[0].shape}")
    print(f"  masks:            {buffer.masks[0].shape}")
    print(f"  active_masks:     {buffer.active_masks[0].shape}")

    print("Populating buffer with random transitions...")
    for step in range(buffer_size):
        share_obs = np.random.randn(*buffer.share_obs[0].shape).astype(np.float32)
        obs = np.random.randn(*buffer.obs[0].shape).astype(np.float32)

        actions = np.column_stack((
            np.random.randint(0, action_dim, buffer.actions[0].shape[0]),
            np.random.randint(0, 3, buffer.actions[0].shape[0]),
        )).astype(np.float32)

        action_log_probs = np.random.randn(*buffer.action_log_probs[0].shape).astype(np.float32)
        value_preds = np.random.randn(*buffer.value_preds[0].shape).astype(np.float32)
        rewards = np.random.randn(*buffer.rewards[0].shape).astype(np.float32)
        masks = np.ones(buffer.masks[0].shape, dtype=np.float32)
        active_masks = np.ones(buffer.active_masks[0].shape, dtype=np.float32)

        # Optional sanity assertions
        assert share_obs.shape == buffer.share_obs[0].shape
        assert obs.shape == buffer.obs[0].shape
        assert actions.shape == buffer.actions[0].shape
        assert action_log_probs.shape == buffer.action_log_probs[0].shape
        assert value_preds.shape == buffer.value_preds[0].shape
        assert rewards.shape == buffer.rewards[0].shape
        assert masks.shape == buffer.masks[0].shape
        assert active_masks.shape == buffer.active_masks[0].shape

        buffer.insert(
            share_obs=share_obs,
            obs=obs,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
            masks=masks,
            active_masks=active_masks,
        )

    print("✓ Buffer populated")

    print("Computing returns...")
    with torch.no_grad():
        next_value = torch.randn(*buffer.value_preds[0].shape).numpy().astype(np.float32)
    buffer.compute_returns(next_value, gamma=0.99)
    print("✓ Returns computed")

    print("Buffer internal shapes after population:")
    print(f"  returns:      {buffer.returns.shape}")
    print(f"  value_preds:  {buffer.value_preds.shape}")

    print("Running multi-agent training loop...")
    trainer.prep_training()
    train_info = trainer.train(buffer, update_actor=True)

    print("✓ Multi-agent training completed")
    print(f"\nTraining Statistics (averaged over {args.ppo_epoch} epochs):")
    for key, val in train_info.items():
        if isinstance(val, (int, float)):
            print(f"  - {key:20s}: {val:.6f}")
        elif isinstance(val, list):
            print(f"  - {key:20s}: {len(val)} actions decoded")
    
    # Print the most recent decoded actions
    recent_actions = train_info["actions"][-10:]
    print("\nLast 10 Taken Actions:")
    start_index = len(train_info["actions"]) - len(recent_actions)
    for i, action in enumerate(recent_actions, start=start_index):
        print(f"  [{i}] {describe_action(action)}")
    
    # Calculate action statistics
    action_counts = {
        "move_right": 0,
        "move_left": 0,
        "move_up": 0,
        "move_down": 0,
        "attack": 0,
    }

    for action in train_info["actions"]:
        if hasattr(action, "action_id"):
            action_name = {
                1: "move_right",
                2: "move_left",
                3: "move_up",
                4: "move_down",
                5: "attack",
            }.get(int(action.action_id), "attack")
        else:
            action_name = action["action"]
        action_counts[action_name] += 1

    total_actions = len(train_info["actions"])
    print("\nAction Distribution:")
    for action_name, count in action_counts.items():
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"  - {action_name:15s}: {count:3d} ({percentage:5.1f}%)")
    print()


def describe_action(action):
    """Return a readable summary for a pybind Action object or fallback compatibility object."""
    if hasattr(action, "action_id"):
        return {
            "unit_id": int(getattr(action, "unit_id", 0)),
            "team_id": int(getattr(action, "team_id", 0)),
            "ally_team_id": int(getattr(action, "ally_team_id", 0)),
            "action_id": int(getattr(action, "action_id", 0)),
            "target_unit_id": int(getattr(action, "target_unit_id", 0)),
        }

    if isinstance(action, dict):
        return {
            "unit_id": int(action.get("unit_id", 0)),
            "team_id": int(action.get("team_id", 0)),
            "ally_team_id": int(action.get("ally_team_id", 0)),
            "action_id": int(action.get("action_id", action.get("action", 0))),
            "target_unit_id": int(action.get("target_unit_id", 0)),
        }

    return {"value": str(action)}


# -------------------------------------------------------------------
# Main runner
# -------------------------------------------------------------------
def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "R_MAPPO Multi-Agent Integration Tests" + " " * 11 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        test_r_mappo_basic()
        test_value_loss()
        test_ppo_update()
        test_network_modes()
        test_gradient_clipping()
        test_multi_agent_ppo_update()
        test_training_loop_multi_agent()

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