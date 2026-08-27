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
  -I ../proto \
  --python_out=generated \
  --grpc_python_out=generated \
  ../proto/UnBARableAI.proto
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

## train.py
Command to start the train.py script, from the python directory. Capslock signals the epceted data type.
```console
uv run train.py --num-episodes INT --num-mini-batch INT --buffer-size INT --num-agents INT --lr FLOAT --gamma FLOAT --device STR --obs-dim INT --action-dim INT --debug-shapes BOOL --use-centralized-v BOOL --use-wandb BOOL
```