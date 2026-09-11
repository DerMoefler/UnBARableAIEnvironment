# Python Tests

```console
cd /home/lennart/repos/UnBARableAIEnvironment/python
uv sync
```

## `test_r_mappo.py`

### How to run

```console
uv run 'test_r_mappo.py'
```

### What it tests

This is the main unit-test suite for `R_MAPPO`. It checks:

- trainer initialization and configuration values,
- value-loss calculation,
- conversion of policy output to the `bar_ai.Action` contract,
- PPO updates with 12- and 13-element samples,
- critic-only updates with `update_actor=False`,
- training and rollout mode switches,
- gradient clipping execution,
- the full `R_MAPPO.train()` path using a mock replay buffer.

The test supplies mock `onpolicy` utility functions because the external
`onpolicy` package is not required by this repository.

### How to interpret output

A successful run ends with a pytest summary similar to:

```text
... passed
```

Failures involving `bar_ai` usually mean the native Python extension was not
built or is not importable in the active uv environment. Failures involving
shapes or `ppo_update` indicate a policy/trainer interface mismatch.

## `quick_test_r_mappo.py`

### How to run

This is a standalone smoke test that runs seven checks in sequence:

```console
uv run 'quick_test_r_mappo.py'
```

### What it tests

The script uses random NumPy data and mocked `onpolicy` utilities to exercise:

1. basic `R_MAPPO` initialization,
2. value-loss calculation,
3. one PPO update,
4. rollout/training network mode switching,
5. gradient clipping,
6. a multi-agent PPO update,
7. replay-buffer population, return calculation, and a complete multi-agent
   training update.

This test does not launch the BAR engine and does not validate gameplay.

### How to interpret output

The standalone runner prints sections labelled `TEST 1` through `TEST 7`.
Each section should contain a check mark and finish with:

```text
============================================================
✓ ALL TESTS PASSED!
============================================================
```

Losses, entropy, gradient norms, and importance weights are diagnostic values;
they are not expected to match a fixed number because the test uses random
inputs. An exception or a missing check mark identifies the failing training
path. In particular, shape errors usually mean that the policy and replay
buffer disagree about agent or action dimensions.

## `test_r_mappo_bar_environment.py`

### How to run

```console
uv run 'test_r_mappo_bar_environment.py'
```

### What it tests

This test replaces the gRPC server, BAR units, and session with dummy objects,
then checks the Python integration between `BAR_Environment` and MAPPO. It
covers:

- construction of 32-feature observations for three dummy units,
- finite observation values and selected feature positions,
- conversion to two training-agent observations,
- policy action sampling,
- replay-buffer insertion,
- return calculation,
- a multi-agent `R_MAPPO.train()` update,
- finite training metrics and decoded actions.

No live BAR process or shared-memory connection is required.

### How to interpret output

The test prints observation and buffer shapes. The important successful lines
include:

```text
✓ BAR_Environment dummy observations received
✓ Buffer populated
✓ Returns computed
```

Pytest should finish with one passed test. Failures in observation assertions
usually indicate a change to `BAR_Environment.get_obs()` or the observation
feature layout. Failures in training metrics indicate a policy, buffer, or
shape-contract regression.

## `test_bar_3v3_training.py`

### How to run

This is a Python-only simulated 3v3 training run:

```console
uv run 'test_bar_3v3_training.py'
```

A short, verbose run is useful while debugging:

```console
uv run 'test_bar_3v3_training.py' \
  --num-episodes 1 \
  --buffer-size 32 \
  --max-steps 16 \
  --debug-env \
  --debug-shapes
```

The script accepts additional options such as `--num-agents`, `--obs-dim`,
`--action-dim`, `--device`, `--seed`, `--num-mini-batch`, and
`--exploration-eps`. It is primarily a standalone script rather than a normal
pytest test.

### What it tests

The script runs a simulated BAR world entirely in Python. It checks the full
training path:

- simulated unit spawning and observations,
- multi-agent action sampling,
- ally and enemy policy steps,
- damage and death rewards,
- natural termination on team death,
- truncation at `--max-steps`,
- `bad_transition` information for time-limit transitions,
- replay-buffer insertion and GAE return calculation,
- MAPPO updates and action decoding,
- episode statistics and action distributions.

It does not validate the real Recoil engine, gRPC synchronization, or native
shared memory.

### How to interpret output

A successful run prints episode summaries, training statistics, and an action
distribution, then exits with status `0`. With `--debug-env`, lines such as
these show the simulated episode state:

```text
DEBUG ENV STEP=... ally_alive=... enemy_alive=... won=... lost=... timeout=...
```

Interpret the flags as follows:

- `won=True`: all enemy units died while at least one ally survived.
- `lost=True`: all ally units died while at least one enemy survived.
- `timeout=True`: `max_steps` was reached before either team died.
- `terminated=True`: the natural game-ending condition was reached.
- `truncated=True`: the step limit ended the episode.

Losses or non-finite training metrics point to a simulated rollout, buffer, or
MAPPO update problem. Unexpected action distributions are diagnostic rather
than failures by themselves because actions are sampled stochastically.

## `test_pybind.py`

### How to run

```console
uv run 'test_pybind.py'
```

### What it tests

The script verifies that the compiled `bar_ai` extension:

- imports successfully,
- exports `UnitData` and `Action`,
- allows all expected `UnitData` fields to be assigned and read,
- preserves the expected values and basic types,
- allows all expected `Action` fields to be assigned and read.

It does not test engine callbacks or shared-memory behavior.

### How to interpret output

A successful run ends with:

```text
🎉 bar_ai UnitData- und Action-Test erfolgreich abgeschlossen!
```

The script exits with status `1` and prints the failing field when the module
cannot be imported, a class is missing, or a value round-trip fails. Import
errors normally mean that the extension has not been built, is not on the
Python path, or was built for a different Python version.

## `test_env_reset_step.py`

### How to run

This test starts a real `BAR_Environment`, launches an engine session, waits
for the first gRPC update, then repeatedly calls `step()`:

```console
uv run 'test_env_reset_step.py'
```

Use this only when the BAR/Recoil executable, start script, shared-memory
reader, gRPC server, map, and game data are installed and configured.

### What it tests

It is a live smoke test for:

- environment construction,
- engine-session startup,
- initial `reset()` observation and info,
- repeated gRPC update synchronization,
- the five-value `step()` result,
- reward and episode-end reporting,
- cleanup through `env.close()`.

The action `10` is explicitly a placeholder and is not a validated gameplay
action.

### How to interpret output

The reset section should print:

```text
=== RESET TEST PASSED ===
```

Each successful environment step prints `=== STEP TEST PASSED ===` and shows
`reward`, `terminated`, and `truncated`. The test ends when one of the flags
becomes true and then prints whether the episode ended naturally or was
truncated.

A gRPC timeout means the engine did not produce an update within the expected
window. A process exit, segmentation fault, or shared-memory error is an
engine/native integration failure, not a normal `terminated` result. Always
inspect the engine `infolog.txt` and stdout for those failures.

## `test_bar_environment.py`

### How to run

```console
uv run 'test_bar_environment.py'
```

### What it tests

This is a minimal observation smoke script. It constructs `BAR_Environment` and
calls `get_obs_agent(agent_id=0)` to print one agent observation.

### How to interpret output

A successful run prints a NumPy-like observation array. In the current source,
the script does not call `env.reset()` or assign a session before requesting the
observation. Therefore it may fail with:

```text
RuntimeError: Engine session is not running. Call reset() first.
```

That failure means the script is incomplete as a standalone test, not that the
observation encoder itself is necessarily broken. Use
`test_env_reset_step.py` for the complete live reset/step path, or update this
smoke script to initialize an environment session first.

