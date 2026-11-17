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
If you want to use the bar_ai module just run:

```console
uv sync
```

and it should work

otherwise try:

```console
uv sync --reinstall
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
