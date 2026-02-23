#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p ./documents

bun install --frozen-lockfile 2>/dev/null || bun install

PID=$(lsof -ti :8181 2>/dev/null) && kill "$PID" 2>/dev/null && sleep 1 || true

npx tsx src/qmd.ts collection add ./documents || true
npx tsx src/qmd.ts update
npx tsx src/qmd.ts embed
exec npx tsx src/qmd.ts mcp --http --port 8181 "$@"
