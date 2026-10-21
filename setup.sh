#!/usr/bin/env bash
# Creates the venv, installs dependencies, generates .env and downloads the model.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
# NVIDIA GPU: torch with CUDA 12.8 (supports Blackwell/RTX 50xx, ~6 GB). No GPU: CPU torch (~200 MB).
if command -v nvidia-smi >/dev/null || [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128 -q
else
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu -q
fi
.venv/bin/pip install -r requirements.txt -q
if [ ! -f .env ]; then
  cp .env.example .env
  sed -i "s|^QILLQA_TOKEN=.*|QILLQA_TOKEN=$(.venv/bin/python -c 'import secrets;print(secrets.token_urlsafe(32))')|" .env
  echo "Generated .env with a new token."
fi
mkdir -p pages var
echo "Downloading the model (once)..."
.venv/bin/python -c "
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(dotenv_values('.env').get('QILLQA_MODEL') or 'Qwen/Qwen3-Embedding-0.6B')
print('prompts:', list(m.prompts))
"
echo "OK. Register the MCP:"
echo "  claude mcp add -s user qillqa -- $PWD/.venv/bin/python $PWD/server.py"
