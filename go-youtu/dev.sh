#!/usr/bin/env bash
# ==============================================================================
# dev.sh - 啟動與執行 Go-Youtu Flutter 專案 (預設自動啟動 Android 模擬器並以 Debug 執行)
# 用法:
#   ./dev.sh              # 自動檢查/啟動 Android 模擬器並執行 flutter run --debug
#   ./dev.sh ios          # 自動啟動 iOS Simulator 並執行 flutter run --debug
#   ./dev.sh test         # 執行靜態分析 (flutter analyze) 與單元測試 (flutter test)
#   ./dev.sh clean        # 清理專案建置快取
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

DEFAULT_AVD_NAME="GoYoutu_Pixel_API34"

# 載入 setup.sh 產生的 SDK 環境變數設定
if [ -f "$PROJECT_DIR/.env.sdk" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env.sdk"
fi
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:/opt/homebrew/bin:/usr/local/bin:$HOME/flutter/bin:$HOME/development/flutter/bin:$PATH"

check_flutter() {
  if ! command -v flutter >/dev/null 2>&1; then
    echo "⚠️  偵測到本機尚未安裝 Flutter SDK，正在執行 ./setup.sh 自動安裝開發環境..."
    "$PROJECT_DIR/setup.sh"
    if [ -f "$PROJECT_DIR/.env.sdk" ]; then
      # shellcheck disable=SC1091
      source "$PROJECT_DIR/.env.sdk"
    fi
  fi
}

ensure_platform_scaffold() {
  if [ ! -f "android/gradlew" ] || [ ! -f "ios/Runner.xcodeproj/project.pbxproj" ]; then
    echo "🔧 偵測到原生平台專案檔尚未完整生成，正在執行 flutter create 補齊平台設定..."
    flutter create . --platforms=android,ios --project-name=go_youtu --org=com.example.goyoutu
  fi
}

# 檢查並確保至少有一台 Android 實機或模擬器正在執行；若無則自動安裝/建立並啟動模擬器
ensure_android_device_running() {
  local running_device
  running_device="$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device" {print $1; exit}' || true)"
  if [ -n "$running_device" ]; then
    echo "✅ 偵測到已連線的 Android 裝置/模擬器: $running_device"
    return 0
  fi

  echo "ℹ️  目前沒有正在執行的 Android 裝置，準備啟動 Android 模擬器..."

  # 若尚未安裝 Android Emulator 執行檔，呼叫 setup.sh 補齊
  if [ ! -x "$ANDROID_HOME/emulator/emulator" ]; then
    echo "📦 偵測到尚未安裝 Android Emulator，正在透過 ./setup.sh 自動安裝模擬器與系統映像檔..."
    "$PROJECT_DIR/setup.sh"
    if [ -f "$PROJECT_DIR/.env.sdk" ]; then
      # shellcheck disable=SC1091
      source "$PROJECT_DIR/.env.sdk"
    fi
  fi

  local avd_name
  avd_name="$("$ANDROID_HOME/emulator/emulator" -list-avds 2>/dev/null | head -n 1 || true)"

  # 若還沒有任何 AVD，呼叫 setup.sh 建立預設 AVD
  if [ -z "$avd_name" ]; then
    echo "📱 尚未建立 Android 虛擬裝置 (AVD)，正在透過 ./setup.sh 自動建立 $DEFAULT_AVD_NAME..."
    "$PROJECT_DIR/setup.sh"
    avd_name="$("$ANDROID_HOME/emulator/emulator" -list-avds 2>/dev/null | head -n 1 || true)"
  fi

  if [ -z "$avd_name" ]; then
    echo "❌ 無法取得可用的 Android AVD，請檢查 Android SDK 設定。"
    exit 1
  fi

  # 確保該 AVD 啟用主機實體鍵盤輸入 (hw.keyboard=yes)
  local avd_cfg="$HOME/.android/avd/${avd_name}.avd/config.ini"
  if [ -f "$avd_cfg" ]; then
    sed -i '' 's/^hw\.keyboard=no/hw.keyboard=yes/' "$avd_cfg"
    grep -q "^hw\.keyboard=yes" "$avd_cfg" || echo "hw.keyboard=yes" >> "$avd_cfg"
  fi

  echo "🚀 正在背景啟動 Android 模擬器: $avd_name ..."
  nohup "$ANDROID_HOME/emulator/emulator" -avd "$avd_name" -netdelay none -netspeed full >/tmp/go_youtu_emulator.log 2>&1 &

  echo "⏳ 正在等待 Android 模擬器開機完成 (首次開機約需 20~60 秒)..."
  adb wait-for-device

  # 等待 Android 系統完成 boot_completed
  local boot_completed=""
  local elapsed=0
  while [[ "$boot_completed" != "1" ]]; do
    boot_completed="$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true)"
    if [[ "$boot_completed" == "1" ]]; then
      break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    if (( elapsed % 10 == 0 )); then
      echo "   ... 模擬器開機中 (已等待 ${elapsed} 秒)"
    fi
    if (( elapsed >= 180 )); then
      echo "⚠️  等待模擬器開機逾時 (180 秒)，嘗試直接繼續執行..."
      break
    fi
  done

  # 額外等待 2 秒確保 Package Manager 服務就緒
  sleep 2
  echo "✅ Android 模擬器 ($avd_name) 已開機就緒！"
}

MODE="${1:-android}"

check_flutter
ensure_platform_scaffold

echo "📦 正在安裝與同步 Dart/Flutter 套件 (flutter pub get)..."
flutter pub get

case "$MODE" in
  android)
    ensure_android_device_running
    ANDROID_DEVICE="$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')"
    echo "🚀 正在 Android 裝置 ($ANDROID_DEVICE) 上啟動 Go-Youtu Debug 模式..."
    flutter run --debug -d "$ANDROID_DEVICE"
    ;;
  ios)
    echo "🍎 正在啟動 iOS Simulator 並執行 Debug 模式..."
    open -a Simulator || true
    flutter run --debug -d ios
    ;;
  test)
    echo "🔍 執行 Clean Code 靜態分析 (flutter analyze)..."
    flutter analyze
    echo "🧪 執行單元測試 (flutter test)..."
    flutter test
    ;;
  clean)
    echo "🧹 清理建置快取..."
    flutter clean
    flutter pub get
    ;;
  *)
    echo "用法: ./dev.sh [android|ios|test|clean]"
    exit 1
    ;;
esac
