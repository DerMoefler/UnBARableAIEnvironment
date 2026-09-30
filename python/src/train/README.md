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
- `reward.py`
  - `RewardConfig`: configures BAR-specific reward shaping and controlled team.
  - `RewardCalculator`: tracks reward baselines and computes per-step rewards.
- `obs_r_mappo.py`
  - Converts BAR's unit dictionary into fixed-width, per-agent MAPPO observations.
- `train.py`
  - command-line MAPPO training loop, using the simulated BAR environment by
    default.
  - converts dictionary observations with `build_r_mappo_observations()` and
    normalizes input shapes before inserting data into the buffer.

For a custom live-environment reward, pass a configured calculator to
`BAR_Environment`:

```python
from src.environment.bar_environment import BAR_Environment
from src.train.reward import RewardCalculator, RewardConfig

env = BAR_Environment(
    reward_calculator=RewardCalculator(
        RewardConfig(training_team_id=1, win=8.0)
    )
)
```

## Training flow

One rollout follows this sequence:

```text
env.reset()
    -> initial observations and info
policy samples actions
    -> values, actions, action log-probabilities
env.step(actions)
  -> next observations, reward, terminated, truncated, info
     dictionary observations are converted by obs_r_mappo.py
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

The default training configuration uses three agents, 53 observation features,
and five action types. The policy samples a second target value as well, so
stored actions have two components.

For the current BAR action space, the first component is the action type and the
second component is a target selector:

- `0`: move north
- `1`: move south
- `2`: move east
- `3`: move west
- `4`: attack

When the selected action is `attack`, the second component chooses which target
unit or target class to attack. For movement actions, the second component is
ignored.

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

### How `train.py` initializes R-MAPPO

When `main()` starts, it parses the command-line arguments, applies the random
seed to Python, NumPy, and PyTorch, and selects the PyTorch device. The defaults
that define the model and rollout are:

| Argument | Default | Purpose |
| --- | ---: | --- |
| `--num-agents` | `3` | Number of controlled agents sharing one policy |
| `--obs-dim` | `53` | Number of features in each agent and shared observation |
| `--action-dim` | `5` | Number of action-type choices produced by the actor: north, south, east, west, attack |
| `--lr` | `0.0001` | Learning rate for the actor and critic optimizers |
| `--buffer-size` | `128` | Maximum rollout length before an update |
| `--num-mini-batch` | `4` | Number of minibatches per PPO epoch |
| `--gamma` | `0.99` | Discount factor used when computing returns |
| `--device` | `cpu` | Device used for policy inference and updates |
| `--seed` | `42` | Seed for Python, NumPy, and PyTorch random generators |

#### BAR observation layout

`build_r_mappo_observations()` in `obs_r_mappo.py` takes the dictionary returned
by `BAR_Environment.create_observation_dictionary()` and creates a `float32`
matrix with one row per unit on `training_team_id`. Rows are ordered by stable
`unit_id`; the ID is used for ordering but is not itself a policy feature.

The default layout is `8 + (2 + 3) * 9 = 53` values per controlled unit:

| Indices | Slot | Features |
| --- | --- | --- |
| `0-7` | Own unit | `unit_def_id`, health fraction, scaled world `pos_x/y/z`, scaled `los_radius`, `is_dead`, `being_built` |
| `8-16` | Nearest ally | Present mask, `unit_def_id`, health fraction, scaled relative `x/y/z`, scaled `los_radius`, `is_dead`, `being_built` |
| `17-25` | Second-nearest ally | Same nine features |
| `26-34` | Nearest enemy | Same nine features |
| `35-43` | Second-nearest enemy | Same nine features |
| `44-52` | Third-nearest enemy | Same nine features |

Health fraction is clipped to `[0, 1]`; position and sight values are divided
by `1000`. Ally slots use matching `ally_team_id`; enemy slots use a different
`ally_team_id`. Neighbors are sorted by 3D distance, with `unit_id` as a tie
breaker. Missing slots are zero-filled, so their present mask is zero. The
converter does not filter by visibility; it uses the units present in the input
dictionary. `train.py` defaults to `obs_dim=53` and uses the mean of agent
observations as `share_obs` when the environment supplies no separate shared
state.

`BAR_Environment.step()` returns the dictionary on non-terminal steps. Its
`reset()` still returns a placeholder observation, so the initial live-engine
state is not yet converted through this pipeline.

## Policy behavior

`R_MAPPO_Policy` contains two categorical distributions:

1. `actor` samples the action type from `action_dim` choices.
   The default action set is: north, south, east, west, and attack.
2. `target_actor` samples a target from `target_dim` choices. When the action
   type is `attack`, the second action component selects the target unit or
   target class. In the engine adapter, this is mapped to the `target_unit_id`
   field of the final action message.

The action tensor therefore has the form:

```python
actions = [action_type, target_index]
# action_type in {0, 1, 2, 3, 4}
# target_index is ignored unless action_type == 4
```

For example:

```python
[4, 2]  # attack; target_unit_id = 2
[0, 0]  # move north; target_index is ignored
```

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

The stored action layout matches the engine action contract: one row per agent,
with a two-value action tuple `(action_type, target_index)`. The engine
adapter then converts this into an action object with `unit_id` and
`target_unit_id` fields for the actual game command.

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
  --obs-dim 53 \
  --action-dim 5 \
  --training-team-id 0 \
  --lr 0.0001 \
  --gamma 0.99 \
  --device cpu \
  --debug-shapes
```

Available command-line options are defined in `train.py`, including
`--max-steps`, `--training-team-id`, `--seed`, `--debug-env`, and
`--debug-shapes`.

## Current limitations

This folder is still partly experimental:

- `train.py` creates the simulated environment by default. The live BAR
  environment is not selected by this runner yet.
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
- `BAR_Environment.reset()` still returns a placeholder, and `step()` does not
  yet send MAPPO actions to the engine. Its shared-memory ownership/read path
  also needs to be confirmed before relying on live observations.

## Related tests

Training and integration tests are under `python/tests/`, including:

- `test_r_mappo.py`: policy and trainer behavior.
- `test_r_mappo_bar_environment.py`: environment observations with MAPPO data.
- `test_bar_3v3_training.py`: simulated end-to-end rollout and update checks.

Run the Python tests from the `python` directory with:

```console
uv run pytest
```
