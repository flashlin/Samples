#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# go-ocr build script
# Compiles the Go binary and packages it as a macOS .app bundle with
# LSUIElement=true (menubar-only, no Dock icon).
#==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="go-ocr"
BUNDLE_ID="com.flash.go-ocr"
VERSION="0.1.0"
MIN_MACOS="11.0"

SRC_DIR="$SCRIPT_DIR/src"
ASSETS_DIR="$SCRIPT_DIR/assets"
BUILD_DIR="$SCRIPT_DIR/build"
BIN_PATH="$BUILD_DIR/$APP_NAME"
APP_PATH="$BUILD_DIR/${APP_NAME}.app"
CONTENTS="$APP_PATH/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RES_DIR="$CONTENTS/Resources"

#------------------------------------------------------------------------------
# Pre-flight
#------------------------------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: build.sh only supports macOS" >&2
    exit 1
fi

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Error: source directory not found: $SRC_DIR" >&2
    exit 1
fi

if [[ ! -f "$SRC_DIR/go.mod" ]]; then
    echo "Error: src/go.mod not found. Run 'go mod init ocr' inside src/ first." >&2
    exit 1
fi

#------------------------------------------------------------------------------
# Step 1: compile go binary
#------------------------------------------------------------------------------
echo "==> Building Go binary"
mkdir -p "$BUILD_DIR"
(
    cd "$SRC_DIR"
    CGO_ENABLED=1 go build \
        -ldflags "-s -w -X main.version=$VERSION" \
        -o "$BIN_PATH" \
        .
)
echo "    -> $BIN_PATH"

#------------------------------------------------------------------------------
# Step 2: prepare .app bundle skeleton
#------------------------------------------------------------------------------
echo "==> Assembling .app bundle"
rm -rf "$APP_PATH"
mkdir -p "$MACOS_DIR" "$RES_DIR"

cp "$BIN_PATH" "$MACOS_DIR/$APP_NAME"
chmod +x "$MACOS_DIR/$APP_NAME"

#------------------------------------------------------------------------------
# Step 3: handle icon (convert .png -> .icns when needed)
#------------------------------------------------------------------------------
ICON_ICNS="$ASSETS_DIR/icon.icns"
ICON_PNG="$ASSETS_DIR/icon.png"

if [[ -f "$ICON_ICNS" ]]; then
    cp "$ICON_ICNS" "$RES_DIR/icon.icns"
    echo "    -> using existing icon.icns"
elif [[ -f "$ICON_PNG" ]]; then
    echo "    -> converting icon.png to icon.icns"
    ICONSET="$BUILD_DIR/icon.iconset"
    rm -rf "$ICONSET"
    mkdir -p "$ICONSET"
    for size in 16 32 64 128 256 512; do
        sips -z "$size" "$size" "$ICON_PNG" \
            --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
        double=$((size * 2))
        sips -z "$double" "$double" "$ICON_PNG" \
            --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
    done
    iconutil -c icns "$ICONSET" -o "$RES_DIR/icon.icns"
    rm -rf "$ICONSET"
else
    echo "    -> WARNING: no icon found at assets/icon.{icns,png}, bundle will use default icon"
fi

#------------------------------------------------------------------------------
# Step 4: write Info.plist
#------------------------------------------------------------------------------
echo "==> Writing Info.plist (LSUIElement=true)"
cat > "$CONTENTS/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>              <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>       <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>        <string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key>           <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundlePackageType</key>       <string>APPL</string>
    <key>CFBundleExecutable</key>        <string>${APP_NAME}</string>
    <key>CFBundleIconFile</key>          <string>icon.icns</string>
    <key>LSMinimumSystemVersion</key>    <string>${MIN_MACOS}</string>
    <key>LSUIElement</key>               <true/>
    <key>NSHighResolutionCapable</key>   <true/>
    <key>NSCameraUsageDescription</key>
    <string>${APP_NAME} does not use the camera.</string>
</dict>
</plist>
EOF

#------------------------------------------------------------------------------
# Step 5: codesign (prefer stable self-signed cert if installed)
#------------------------------------------------------------------------------
CERT_NAME="go-ocr-codesign"
if security find-identity -p codesigning 2>/dev/null | grep -q "$CERT_NAME"; then
    echo "==> Codesigning with stable identity '$CERT_NAME' (TCC perms persist)"
    codesign --force --deep --sign "$CERT_NAME" "$APP_PATH" 2>&1 | sed 's/^/    /'
else
    echo "==> Ad-hoc codesigning (TCC perms reset on every rebuild)"
    echo "    Run ./setup-cert.sh once for a stable identity."
    codesign --force --deep --sign - "$APP_PATH" 2>&1 | sed 's/^/    /'
fi

#------------------------------------------------------------------------------
# Step 6: report
#------------------------------------------------------------------------------
SIZE=$(du -sh "$APP_PATH" | awk '{print $1}')
echo ""
echo "Build complete."
echo "  Bundle: $APP_PATH"
echo "  Size:   $SIZE"
echo ""
echo "Run locally:"
echo "  open $APP_PATH"
echo ""
echo "Install to /Applications:"
echo "  rm -rf /Applications/${APP_NAME}.app && cp -R $APP_PATH /Applications/"
