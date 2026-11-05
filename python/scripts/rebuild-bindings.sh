#!/usr/bin/env sh

set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
python_dir="${repository_root}/python"
typings_dir="${python_dir}/typings"

echo "Rebuilding and reinstalling bar_ai..."
uv sync --project "${python_dir}" --reinstall

echo "Regenerating Python type stubs..."
rm -rf "${typings_dir}/bar_ai" "${typings_dir}/bar_ai.pyi"
mkdir -p "${typings_dir}"

uv run --project "${python_dir}" pybind11-stubgen bar_ai -o "${typings_dir}"

echo "bar_ai rebuilt and type stubs generated"
