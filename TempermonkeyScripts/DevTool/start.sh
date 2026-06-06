#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_DIR="$SCRIPT_DIR/PageProbeGateway"

if [ ! -f "$GATEWAY_DIR/dist/server.js" ]; then
  echo "Gateway is not built yet. Run ./build.sh first." >&2
  exit 1
fi

cd "$GATEWAY_DIR"

if [ -n "${DEVTOOL_TOKEN:-}" ]; then
  GATEWAY_TOKEN="$DEVTOOL_TOKEN"
  echo "==> Using token from DEVTOOL_TOKEN environment variable"
elif [ -f "$GATEWAY_DIR/.data/token" ]; then
  GATEWAY_TOKEN="$(tr -d '[:space:]' < "$GATEWAY_DIR/.data/token")"
  echo "==> Using token from .data/token"
else
  GATEWAY_TOKEN=""
  echo "==> No token found; gateway will generate one in .data/token on start"
fi

if [ -n "$GATEWAY_TOKEN" ]; then
  echo ""
  echo "Gateway token: $GATEWAY_TOKEN"
  echo ""
  echo "MCP client config (Streamable HTTP, token already carried):"
  echo "  URL:    http://127.0.0.1:17890/mcp"
  echo "  Header: Authorization: Bearer $GATEWAY_TOKEN"
  echo ""
  echo "Claude Code:"
  echo "  claude mcp add --transport http page-probe http://127.0.0.1:17890/mcp \\"
  echo "    --header \"Authorization: Bearer $GATEWAY_TOKEN\""
  echo ""
fi

echo "==> Starting PageProbeGateway on http://127.0.0.1:17890"
exec bun dist/server.js
