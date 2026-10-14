#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi
uv venv --python 3.11 --allow-existing .venv
uv pip install --python .venv/bin/python -e '.[dev]'
# Explicit ONNX-only install, following upstream's lightweight model instructions.
uv pip install --python .venv/bin/python --no-deps kittenml==0.9.4
if [ ! -f .env ]; then cp .env.example .env; chmod 600 .env; fi
echo "Ready. Fill DEEPSEEK_API_KEY in .env, then run ./scripts/run.sh"
