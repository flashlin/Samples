#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# go-ocr install script
#
# Usage:
#   ./install.sh              build + install to /Applications
#   ./install.sh --autostart  also register LaunchAgent (start at login)
#   ./install.sh --uninstall  remove from /Applications and LaunchAgent
#==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="go-ocr"
BUNDLE_ID="com.flash.go-ocr"
APP_PATH="$SCRIPT_DIR/build/${APP_NAME}.app"
INSTALL_DEST="/Applications/${APP_NAME}.app"
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
LAUNCH_AGENT_PLIST="$LAUNCH_AGENT_DIR/${BUNDLE_ID}.plist"

MODE="install"
if [[ $# -gt 0 ]]; then
    case "$1" in
        --autostart)  MODE="autostart" ;;
        --uninstall)  MODE="uninstall" ;;
        *) echo "Usage: $0 [--autostart | --uninstall]" >&2; exit 1 ;;
    esac
fi

#------------------------------------------------------------------------------
uninstall() {
    echo "==> Stopping running instance"
    pkill -f "${APP_NAME}.app" 2>/dev/null || true
    sleep 1

    if [[ -f "$LAUNCH_AGENT_PLIST" ]]; then
        echo "==> Removing LaunchAgent"
        launchctl unload "$LAUNCH_AGENT_PLIST" 2>/dev/null || true
        rm -f "$LAUNCH_AGENT_PLIST"
    fi

    if [[ -d "$INSTALL_DEST" ]]; then
        echo "==> Removing $INSTALL_DEST"
        rm -rf "$INSTALL_DEST"
    fi

    echo "Uninstalled."
}

#------------------------------------------------------------------------------
install_app() {
    echo "==> Building"
    "$SCRIPT_DIR/build.sh"

    echo "==> Stopping running instance"
    pkill -f "${APP_NAME}.app" 2>/dev/null || true
    sleep 1

    echo "==> Installing to /Applications"
    rm -rf "$INSTALL_DEST"
    cp -R "$APP_PATH" "$INSTALL_DEST"
    echo "    -> $INSTALL_DEST"
}

#------------------------------------------------------------------------------
register_autostart() {
    mkdir -p "$LAUNCH_AGENT_DIR"
    cat > "$LAUNCH_AGENT_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>              <string>${BUNDLE_ID}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DEST}/Contents/MacOS/${APP_NAME}</string>
    </array>
    <key>RunAtLoad</key>          <true/>
    <key>KeepAlive</key>          <false/>
    <key>StandardOutPath</key>    <string>/dev/null</string>
    <key>StandardErrorPath</key>  <string>/dev/null</string>
</dict>
</plist>
EOF
    launchctl load "$LAUNCH_AGENT_PLIST" 2>/dev/null || true
    echo "==> LaunchAgent registered: $LAUNCH_AGENT_PLIST"
    echo "    go-ocr will start automatically at login."
}

#------------------------------------------------------------------------------
case "$MODE" in
    uninstall)
        uninstall
        ;;
    autostart)
        install_app
        register_autostart
        echo ""
        echo "==> Launching"
        open "$INSTALL_DEST"
        echo ""
        echo "Done. go-ocr is installed and will start at every login."
        ;;
    install)
        install_app
        echo ""
        echo "==> Launching"
        open "$INSTALL_DEST"
        echo ""
        echo "Done. To also start at login: ./install.sh --autostart"
        echo "      To uninstall:           ./install.sh --uninstall"
        ;;
esac
