# Training

This folder contains the policy, rollout buffer, and MAPPO trainer used by
`UnBARableAIEnvironment`. The implementation is designed for multi-agent
reinforcement learning with one policy shared across agents.

## Components

- `policy.py`
  - `Actor`: maps an observation to discrete-action logits.
  - `Critic`: estimates the value of a shared observation.
  - `R_MAPPO_Policy`: combines the actor and critic, samples actions, and
    evaluates stored actions.
- `replay_buffer.py`
  - `SharedReplayBuffer`: stores one rollout and computes generalized
    advantage estimates (GAE) and returns.
- `r_mappo.py`
  - `R_MAPPO`: performs PPO-style actor and critic updates from the buffer.
  - `ValueNorm`: optional running normalization for value targets.
- `train.py`
  - command-line training loop for the simulated BAR environment.
  - accepts several environment result formats and normalizes their shapes
    before inserting data into the buffer.

## Training flow

One rollout follows this sequence:

```text
env.reset()
    -> initial observations and info
policy samples actions
    -> values, actions, action log-probabilities
env.step(actions)
    -> next observations, reward, terminated, truncated, info
buffer.insert(...)
    -> transition is stored
buffer.compute_returns(next_value)
    -> GAE advantages and value targets
trainer.train(buffer)
    -> actor and critic optimization
buffer.reset()
    -> storage is cleared for the next rollout
```

The critic uses a shared observation (`share_obs`), while the actor uses one
local observation per agent (`obs`). This is the centralized-value,
decentralized-policy pattern used by MAPPO.

## Environment contract

The preferred Gymnasium-style interface is:

```python
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(action)
```

The training loop also accepts these extended step formats:

```python
(obs, share_obs, reward, terminated, truncated, info)
(obs, share_obs, reward, terminated, truncated, info, available_actions)
```

`terminated` and `truncated` both indicate that a rollout should stop:

```python
done = terminated or truncated
```

Their meanings are different:

- `terminated=True`: the game reached a natural terminal state, such as a
  win or loss.
- `truncated=True`: the episode was stopped by an artificial limit, such as a
  step or frame limit.

The current rollout loop uses their logical OR to stop collection and sets
`masks` to zero at that boundary. This means the current implementation does
not yet preserve value bootstrapping for truncation separately from natural
termination. Environment implementations should include a final observation
when an episode ends.

## Data shapes

The default training configuration uses three agents, 32 observation features,
and seven action types. The policy samples a second target value as well, so
stored actions have two components.

| Data | Shape | Meaning |
| --- | --- | --- |
| Single observation | `(obs_dim,)` | Features for one agent or one shared state |
| Agent observations | `(num_agents, obs_dim)` | Input to the actor |
| Shared observation | `(obs_dim,)` | Input to the centralized critic |
| Stored actions | `(num_agents, 2)` | Action type and target value |
| Action log-probabilities | `(num_agents, 1)` | Log-probability of each sampled action |
| Value predictions | `(num_agents, 1)` | Critic output repeated per agent |
| Rewards | `(num_agents, 1)` | Reward received after the action |
| Continuation masks | `(num_agents, 1)` | Used by GAE to decide whether to bootstrap |
| Active-agent masks | `(num_agents, 1)` | Excludes inactive agents from losses |
| Available actions | `(num_agents, action_dim)` | Legal-action indicators, when supplied |
| RNN states | `(num_agents, recurrent_n, hidden_size)` | Recurrent-policy state storage |

The replay buffer allocates observations, values, returns, masks, and recurrent
states with `buffer_size + 1` entries. The extra entry stores the next state or
bootstrap value associated with the final transition.

## Policy behavior

`R_MAPPO_Policy` contains two categorical distributions:

1. `actor` samples the action type from `action_dim` choices.
2. `target_actor` samples a target from `target_dim` choices. `target_dim` is
   `3` by default.

`get_action(obs)` returns:

```python
actions, action_log_probs
# actions: (batch_size, 2)
# action_log_probs: (batch_size,)
```

`evaluate_actions(...)` returns values, action log-probabilities, and mean
entropy for the actions stored in a rollout. The critic is evaluated on
`share_obs`; the actor is evaluated on `obs`.

## Replay buffer

`SharedReplayBuffer.insert(...)` stores one transition at the current buffer
index and advances the index circularly. The buffer stores:

- current observations and shared observations,
- selected actions and their log-probabilities,
- critic predictions,
- rewards,
- continuation and active-agent masks,
- optional available-action arrays.

`compute_returns(next_value, gamma, gae_lambda)` calculates GAE backwards
through the rollout. Its continuation mask is used in both the TD residual and
the recursive GAE calculation:

```text
delta_t = reward_t + gamma * value_(t+1) * mask_(t+1) - value_t
```

Use `reset()` after a training update to clear the stored rollout and restore
masks to one.

## MAPPO trainer

`R_MAPPO` reads its hyperparameters from an attribute-based configuration
object. The current `TrainerArgs` in `train.py` provides:

- PPO clipping: `clip_param`, `ppo_epoch`, `num_mini_batch`
- Optimization: `value_loss_coef`, `entropy_coef`, `max_grad_norm`
- Value loss options: clipped value loss, Huber loss, POPArt, or `ValueNorm`
- Mask options: policy and value active masks
- Recurrent options: recurrent policy and sequence chunk length

`use_popart` and `use_valuenorm` cannot both be enabled.

The trainer exposes `prep_rollout()` before collection, `prep_training()` before
updates, and `train(buffer, update_actor=True)` for the actual optimization.

## Running the current trainer

From the `python` directory, install dependencies first:

```console
uv sync
```

Then run the simulated training loop:

```console
uv run python src/train/train.py
```

Useful options include:

```console
uv run python src/train/train.py \
  --num-episodes 10 \
  --num-mini-batch 4 \
  --buffer-size 128 \
  --num-agents 3 \
  --obs-dim 32 \
  --action-dim 7 \
  --lr 0.0001 \
  --gamma 0.99 \
  --device cpu \
  --debug-shapes
```

Available command-line options are defined in `train.py`, including
`--max-steps`, `--seed`, `--debug-env`, and `--debug-shapes`.

## Current limitations

This folder is still partly experimental:

- `train.py` imports a simulated environment through fallback import paths;
  verify the selected environment before running a real BAR experiment.
- The environment adapters accept malformed or differently shaped arrays by
  padding, truncating, or tiling them. This helps compatibility but can hide
  integration errors.
- `available_actions` is carried through the rollout interfaces, but the
  current policy implementation does not apply it to its categorical logits.
- Recurrent-state arrays are currently placeholder-sized in the replay buffer;
  use the recurrent generator only with matching state shapes.
- `terminated` and `truncated` are currently combined into one continuation
  mask in the rollout loop. Natural termination and time-limit truncation
  should be separated before using this code for production training.
- The BAR environment currently contains placeholder observations and action
  handling. See `src/environment/bar_environment.py` for the live integration
  status.

## Related tests

Training and integration tests are under `python/tests/`, including:

- `test_r_mappo.py`: policy and trainer behavior.
- `test_r_mappo_bar_environment.py`: environment observations with MAPPO data.
- `test_bar_3v3_training.py`: simulated end-to-end rollout and update checks.

Run the Python tests from the `python` directory with:

```console
uv run pytest
```
