# UnBARableAIEnvironment: Python
This folder contains the Python contents of the repo.

## Installation
The python dependencies are managed using uv. To get started, first run:
```console
uv sync
```
## gRPC
To generate the gRPC stubs, run:
```console
mkdir generated
uv run python -m grpc_tools.protoc \
  -I ../cpp/proto \
  --python_out=generated \
  --grpc_python_out=generated \
  ../cpp/proto/UnBARableAI.proto
```

## Pybind
Our C++ **UnBARableAI Core** library exposes python bindings which are automatically built using uv. If something changes in the library though but you already have the package installed (having previously run *uv sync*), these changes will not be picked up as uv decides there is not reason for rebuilding. Therefore, make sure to always run:

```console
uv sync --reinstall
```

Pybind is also capable of generating **stubs** for type information. Since that would be two commands (one for uv, one for stub generation), we created a script for that :) Simply run:

``` console
./scripts/rebuild-bindings.sh
```

### Pyright
[Pyright](https://github.com/microsoft/pyright) is a static type checker for python. It's probably already embedded, but you can also invoke it manually on the command line using:

``` console
uv run pyright --project ../pyrightconfig.json
```

This gives you a listing of all the errors which you may or may not find useful :)

## train.py
Command to start the train.py script, from the python directory. Capslock signals the epceted data type.
```console
uv run train.py --num-episodes INT --num-mini-batch INT --buffer-size INT --num-agents INT --lr FLOAT --gamma FLOAT --device STR --obs-dim INT --action-dim INT --debug-shapes BOOL --use-centralized-v BOOL --use-wandb BOOL
```

## Pytest
to run pytest like [environment test](./tests/test_env.py), run:
```console
uv run pytest -v tests/test_env.py
```
usefull flags:
- `-v` makes the test output more detailed and shows each test individually
- `-x` stops the entire test run after the first failed test.
- `-s` Disables output capturing and displays `print()`, `stdout`, and `stderr` directly in the terminal.
- `--durations=0` shows how long each test ran
- `--log-cli-level=INFO`all logging.info gets displayed in terminal

for more flag options run:
```console
uv run pytest -h
```
