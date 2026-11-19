#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/server"
uv sync
exec uv run caption-server-setup "$@"
