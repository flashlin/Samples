#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_PATH="$SCRIPT_DIR/build/go-ocr.app"
BIN_PATH="$APP_PATH/Contents/MacOS/go-ocr"
LOG_PATH="$HOME/Library/Logs/go-ocr.log"

needs_rebuild() {
    [[ ! -x "$BIN_PATH" ]] && return 0
    local newer
    newer=$(find "$SCRIPT_DIR/src" -type f \( -name "*.go" -o -name "go.mod" -o -name "go.sum" \) -newer "$BIN_PATH" 2>/dev/null | head -1)
    [[ -n "$newer" ]]
}

echo "==> Stopping any running instance"
pkill -f "go-ocr/build/go-ocr.app" 2>/dev/null || true
pkill -f "/Applications/go-ocr.app" 2>/dev/null || true
sleep 1

if needs_rebuild; then
    echo "==> Sources changed, building"
    "$SCRIPT_DIR/build.sh"
else
    echo "==> Using existing build"
fi

echo "==> Starting go-ocr (via macOS open, proper bundle launch)"
echo "    bundle: $APP_PATH"
echo "    log:    $LOG_PATH"
echo ""

open "$APP_PATH"
sleep 1
if pgrep -f "go-ocr/build/go-ocr.app/Contents/MacOS/go-ocr" >/dev/null; then
    echo "Launched. tail -f the log to follow:"
    echo "    tail -f $LOG_PATH"
else
    echo "WARNING: process not detected. Check log:"
    echo "    cat $LOG_PATH"
fi
