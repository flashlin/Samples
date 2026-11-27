#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d "$ROOT_DIR/.venv" ] || [ ! -d "$ROOT_DIR/ComfyUI" ]; then
    bash "$ROOT_DIR/setup.sh"
fi

source "$ROOT_DIR/lib/load_hf_token.sh"
export PATH="$ROOT_DIR/bin:$ROOT_DIR/.venv/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1

ENTRY_SCRIPT="$1"
shift
exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/$ENTRY_SCRIPT" "$@"
