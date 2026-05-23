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