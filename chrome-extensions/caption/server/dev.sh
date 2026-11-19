#!/usr/bin/env bash
set -euo pipefail

server_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$server_dir"
export CAPTION_DEV_EXTENSION_DIR="$server_dir/../dist"
exec uv run caption-server "$@"
