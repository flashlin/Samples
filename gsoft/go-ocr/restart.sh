#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# go-ocr restart script
#
# Restarts the installed app to reload config.json (e.g. after changing the
# OCR model). Does not rebuild — use build.sh / install.sh for code changes.
#==============================================================================

APP_PATH="/Applications/go-ocr.app"
BIN_PATH="$APP_PATH/Contents/MacOS/go-ocr"
CONFIG_PATH="$HOME/Library/Application Support/go-ocr/config.json"
LOG_PATH="$HOME/Library/Logs/go-ocr.log"

if [[ ! -d "$APP_PATH" ]]; then
    echo "ERROR: $APP_PATH not found. Run ./install.sh first." >&2
    exit 1
fi

echo "==> Stopping running instance"
pkill -f "$APP_PATH" 2>/dev/null || true
sleep 1

echo "==> Starting go-ocr"
open "$APP_PATH"
sleep 1

if pgrep -f "$BIN_PATH" >/dev/null; then
    echo "Restarted."
    if [[ -f "$CONFIG_PATH" ]]; then
        echo "    config: $CONFIG_PATH"
    fi
    echo "    log:    tail -f $LOG_PATH"
else
    echo "WARNING: process not detected. Check log:" >&2
    echo "    cat $LOG_PATH" >&2
    exit 1
fi
