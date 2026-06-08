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

if it does not work run:

```console
uv build --wheel
uv sync
```

if you changed something in the cpp files and want to rebuild run:

```console
uv sync --reinstall
```