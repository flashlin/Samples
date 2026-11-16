#!/usr/bin/env bash
# ==============================================================================
# build-ios.sh - 編譯 iOS 執行檔 (.app / .ipa)
# 用法:
#   ./build-ios.sh                # 預設編譯 iOS Simulator Debug .app
#   ./build-ios.sh simulator      # 編譯 iOS Simulator Debug .app
#   ./build-ios.sh release        # 編譯 iOS 實機 Release (免憑證簽章 --no-codesign)
#   ./build-ios.sh ipa            # 編譯 iOS Archive / IPA
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env.sdk" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env.sdk"
fi
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:$HOME/flutter/bin:$HOME/development/flutter/bin:$HOME/fvm/default/bin"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "❌ 錯誤: iOS 編譯僅支援在 macOS 環境上執行。"
  exit 1
fi

if ! command -v flutter >/dev/null 2>&1; then
  echo "⚠️  偵測到本機尚未安裝 Flutter SDK，正在執行 ./setup.sh 自動安裝開發環境..."
  "$PROJECT_DIR/setup.sh"
  if [ -f "$PROJECT_DIR/.env.sdk" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env.sdk"
  fi
fi

if [ ! -f "ios/Runner.xcodeproj/project.pbxproj" ]; then
  echo "🔧 初始化 iOS 原生專案結構..."
  flutter create . --platforms=android,ios --project-name=go_youtu --org=com.example.goyoutu
fi

BUILD_TYPE="${1:-simulator}"
OUTPUT_DIR="$PROJECT_DIR/dist/ios"
mkdir -p "$OUTPUT_DIR"

echo "📦 正在同步套件 (flutter pub get)..."
flutter pub get

case "$BUILD_TYPE" in
  simulator|debug)
    echo "🔨 正在編譯 iOS Simulator Debug 應用程式..."
    flutter build ios --debug --simulator
    rm -rf "$OUTPUT_DIR/Runner-Simulator.app"
    cp -R build/ios/iphonesimulator/Runner.app "$OUTPUT_DIR/Runner-Simulator.app"
    echo "✅ 編譯完成！輸出檔案位置:"
    echo "   - build/ios/iphonesimulator/Runner.app"
    echo "   - $OUTPUT_DIR/Runner-Simulator.app"
    ;;
  release)
    echo "🔨 正在編譯 iOS Release 應用程式 (--no-codesign)..."
    flutter build ios --release --no-codesign
    rm -rf "$OUTPUT_DIR/Runner-Release.app"
    cp -R build/ios/iphoneos/Runner.app "$OUTPUT_DIR/Runner-Release.app"
    echo "✅ 編譯完成！輸出檔案位置:"
    echo "   - build/ios/iphoneos/Runner.app"
    echo "   - $OUTPUT_DIR/Runner-Release.app"
    ;;
  ipa)
    echo "🔨 正在編譯 iOS IPA..."
    flutter build ipa --release
    echo "✅ 編譯完成！輸出目錄: build/ios/ipa/"
    ;;
  *)
    echo "用法: ./build-ios.sh [simulator|debug|release|ipa]"
    exit 1
    ;;
esac
