# R_MAPPO Integration Guide

## Overview

This guide explains how to integrate the **R_MAPPO** (Recurrent Multi-Agent PPO) trainer with your BAR environment for multi-agent reinforcement learning.

## Components

### 1. **Environment** (`bar_environment.py`)
- Your custom BAR game environment
- Provides `reset()`, `step()`, and observation methods
- Connects to the game engine via `engine_session.py`

### 2. **Policy** (`policy.py`)
- **Actor Network**: Maps observations to action logits
- **Critic Network**: Maps observations to value predictions
- **Optimizers**: Adam optimizers for both networks
- **evaluate_actions()**: Required method for R_MAPPO to compute log probabilities and entropy

### 3. **Replay Buffer** (`replay_buffer.py`)
- **SharedReplayBuffer**: Stores trajectories from environment rollouts
- **Methods**:
  - `insert()`: Add experience to buffer
  - `compute_returns()`: Calculate returns and advantages using GAE
  - `feed_forward_generator()`: Yield mini-batches for training

### 4. **Trainer** (`r_mappo.py`)
- **R_MAPPO**: Main training algorithm
- **Methods**:
  - `ppo_update()`: Single PPO update step
  - `train()`: Full training iteration over mini-batches
  - `prep_training()`: Set networks to training mode
  - `prep_rollout()`: Set networks to evaluation mode

### 5. **Training Script** (`train.py`)
- Orchestrates the entire training pipeline
- Manages the training loop: rollout → compute returns → training update

## Integration Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. ROLLOUT PHASE (Collect Experience)              │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ • Set policy to eval mode: prep_rollout()          │   │
│  │ • Get obs from environment                         │   │
│  │ • Sample actions from policy.actor()               │   │
│  │ • Get values from policy.critic()                  │   │
│  │ • Step environment                                 │   │
│  │ • Store in replay buffer                           │   │
│  │ • Repeat until buffer is full or episode ends      │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 2. COMPUTE RETURNS & ADVANTAGES                    │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ • buffer.compute_returns()                         │   │
│  │ • Uses GAE (Generalized Advantage Estimation)      │   │
│  │ • Normalizes advantages                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. TRAINING PHASE (Update Policy)                  │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ • Set policy to train mode: prep_training()        │   │
│  │ • Loop over PPO epochs                             │   │
│  │   - Generate mini-batches from buffer              │   │
│  │   - For each batch:                                │   │
│  │     · trainer.ppo_update()                         │   │
│  │     · Update actor (policy loss)                   │   │
│  │     · Update critic (value loss)                   │   │
│  │     · Accumulate gradients and losses              │   │
│  │   - Average losses over mini-batches               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 4. RESET & REPEAT                                  │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ • Reset replay buffer                              │   │
│  │ • Next episode                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Key Data Flow

### Rollout Phase
```
Observation (obs_dim)
      ↓
Policy.actor() → action logits
      ↓
Sample action + compute log_prob
      ↓
Policy.critic() → value prediction
      ↓
Environment.step(action)
      ↓
New observation, reward, done
      ↓
Buffer.insert(obs, action, log_prob, value, reward, ...)
```

### Training Phase
```
Buffer.feed_forward_generator() → mini-batch samples
      ↓
R_MAPPO.ppo_update(sample)
      ↓
Policy.evaluate_actions() → recompute values, log_probs, entropy
      ↓
Compute policy loss and value loss
      ↓
Backward pass + optimization step
      ↓
Accumulate losses for logging
```

## Configuration

Edit `Args` class in `train.py` to adjust hyperparameters:

```python
class Args:
    clip_param = 0.2              # PPO clipping range
    ppo_epoch = 10                # Number of PPO epochs per update
    num_mini_batch = 4            # Number of mini-batches
    value_loss_coef = 1.0         # Weight for value loss
    entropy_coef = 0.01           # Weight for entropy regularization
    max_grad_norm = 0.5           # Gradient clipping
    huber_delta = 10.0            # Huber loss delta
```

## Running Training

```bash
# Basic training
python train.py

# With custom parameters
python train.py --num-episodes 100 --num-mini-batch 8 --lr 1e-3

# On GPU
python train.py --device cuda
```

## Customization Points

### 1. Observation Dimension
In `train.py`, set to match your environment:
```python
obs_dim = 128  # TODO: Set to your actual observation dimension
```

### 2. Action Dimension
In `train.py`, set to match your environment's action space:
```python
action_dim = 7  # e.g., 7 actions: move_up, move_down, ..., stay
```

### 3. Custom Networks
Modify `Actor` and `Critic` in `policy.py` to use different architectures:
```python
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        # Add recurrent layers, attention, etc.
```

### 4. Observation Normalization
Add preprocessing in `train.py` before passing to policy:
```python
obs_normalized = (obs - obs_mean) / (obs_std + 1e-8)
```

## Next Steps

1. **Adjust observation/action dimensions** to match your environment
2. **Test environment integration** with `test_bar_environment.py`
3. **Run training** with `python train.py`
4. **Monitor training metrics** (losses, rewards) for debugging
5. **Tune hyperparameters** for your specific problem

## Troubleshooting

**Issue**: "No shared_memory_reader_factory configured"
- Solution: Pass a reader factory to `EngineSessionConfig()` in the environment initialization

**Issue**: Shape mismatch errors
- Solution: Verify `obs_dim` and `action_dim` match your environment's observation/action spaces

**Issue**: Poor training performance
- Solution: Adjust learning rate, entropy coefficient, or PPO epochs in `Args`

## References

- MAPPO Paper: https://arxiv.org/abs/2103.01955
- On-Policy Repository: https://github.com/marlbenchmark/on-policy
- PPO Paper: https://arxiv.org/abs/1707.06347
