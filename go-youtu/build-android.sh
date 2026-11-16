#!/usr/bin/env bash
# ==============================================================================
# build-android.sh - 編譯 Android 執行檔 (APK / AppBundle)
# 用法:
#   ./build-android.sh            # 預設編譯 Debug APK (先以 Android Debug 為主)
#   ./build-android.sh debug      # 編譯 Debug APK
#   ./build-android.sh release    # 編譯 Release APK
#   ./build-android.sh aab        # 編譯 Release AppBundle (.aab)
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env.sdk" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env.sdk"
fi
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:$HOME/flutter/bin:$HOME/development/flutter/bin:$HOME/fvm/default/bin"

if ! command -v flutter >/dev/null 2>&1; then
  echo "⚠️  偵測到本機尚未安裝 Flutter SDK，正在執行 ./setup.sh 自動安裝開發環境..."
  "$PROJECT_DIR/setup.sh"
  if [ -f "$PROJECT_DIR/.env.sdk" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env.sdk"
  fi
fi

if [ ! -f "android/gradlew" ]; then
  echo "🔧 初始化 Android 原生專案結構..."
  flutter create . --platforms=android,ios --project-name=go_youtu --org=com.example.goyoutu
fi

BUILD_TYPE="${1:-debug}"
OUTPUT_DIR="$PROJECT_DIR/dist/android"
mkdir -p "$OUTPUT_DIR"

echo "📦 正在同步套件 (flutter pub get)..."
flutter pub get

case "$BUILD_TYPE" in
  debug)
    echo "🔨 正在編譯 Android Debug APK..."
    flutter build apk --debug
    cp -f build/app/outputs/flutter-apk/app-debug.apk "$OUTPUT_DIR/go-youtu-debug.apk"
    echo "✅ 編譯完成！輸出檔案位置:"
    echo "   - build/app/outputs/flutter-apk/app-debug.apk"
    echo "   - $OUTPUT_DIR/go-youtu-debug.apk"
    ;;
  release)
    echo "🔨 正在編譯 Android Release APK..."
    flutter build apk --release
    cp -f build/app/outputs/flutter-apk/app-release.apk "$OUTPUT_DIR/go-youtu-release.apk"
    echo "✅ 編譯完成！輸出檔案位置:"
    echo "   - build/app/outputs/flutter-apk/app-release.apk"
    echo "   - $OUTPUT_DIR/go-youtu-release.apk"
    ;;
  aab)
    echo "🔨 正在編譯 Android Release AppBundle (.aab)..."
    flutter build appbundle --release
    cp -f build/app/outputs/bundle/release/app-release.aab "$OUTPUT_DIR/go-youtu-release.aab"
    echo "✅ 編譯完成！輸出檔案位置:"
    echo "   - build/app/outputs/bundle/release/app-release.aab"
    echo "   - $OUTPUT_DIR/go-youtu-release.aab"
    ;;
  *)
    echo "用法: ./build-android.sh [debug|release|aab]"
    exit 1
    ;;
esac
