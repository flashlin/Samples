#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Installing workspace dependencies"
pnpm install --frozen-lockfile

TOKEN_FILE="$SCRIPT_DIR/PageProbeGateway/.data/token"
if [ ! -f "$TOKEN_FILE" ]; then
  echo "==> Generating gateway token"
  mkdir -p "$(dirname "$TOKEN_FILE")"
  openssl rand -hex 32 > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi
GATEWAY_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"

echo "==> Injecting gateway token into the extension build"
printf 'WXT_GATEWAY_TOKEN=%s\n' "$GATEWAY_TOKEN" > "$SCRIPT_DIR/PageProbe/.env.local"

echo "==> Building all workspace packages (Protocol -> Gateway -> Extension)"
pnpm --recursive run build

EXTENSION_OUTPUT="$SCRIPT_DIR/PageProbe/.output/chrome-mv3"
GATEWAY_OUTPUT="$SCRIPT_DIR/PageProbeGateway/dist"

echo ""
echo "==> Build complete"
echo ""
echo "PageProbe extension (Load unpacked in Chrome):"
echo "  $EXTENSION_OUTPUT"
echo ""
echo "PageProbeGateway server output:"
echo "  $GATEWAY_OUTPUT  (entry: server.js)"
echo ""
echo "Gateway token (already baked into the extension):"
echo "  $GATEWAY_TOKEN"
echo ""
echo "Next steps:"
echo "  1. Open chrome://extensions, enable Developer mode."
echo "  2. Load unpacked -> $EXTENSION_OUTPUT"
echo "  3. Start the gateway with ./start.sh"
echo "  4. In PageProbe options, add Allowed origins (whitelist) and Save."
echo "     The token is pre-filled and the extension connects automatically."
