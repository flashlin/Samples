#!/usr/bin/env bash
# ==============================================================================
# setup.sh - 自動檢查並安裝 Go-Youtu 所需的 Dart / Flutter、Android (含模擬器) 與 iOS 開發環境
# 支援組件:
#   1. Homebrew 檢查
#   2. OpenJDK 17 (Android Gradle 與 sdkmanager / avdmanager 必備)
#   3. Flutter & Dart SDK (透過 Homebrew Cask 或官方 Git Stable 分支安裝)
#   4. Android SDK Command-line Tools、platform-tools (adb)、build-tools、Android 34
#   5. Android Emulator 與系統映像檔 (arm64-v8a / x86_64) + 自動建立預設 AVD 虛擬手機
#   6. CocoaPods (iOS 原生套件管理必備)
#   7. 自動寫入專案 .env.sdk 與 ~/.zshrc，並初始化專案平台檔案與執行 flutter doctor
#
# 用法:
#   ./setup.sh              # 安裝完整開發環境 (Flutter + Android SDK + Emulator AVD + iOS)
#   ./setup.sh --check      # 僅檢查目前環境狀態，不執行安裝
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

ENV_FILE="$PROJECT_DIR/.env.sdk"
DEFAULT_AVD_NAME="GoYoutu_Pixel_API34"
CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=true
fi

# 載入既有路徑
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/development/flutter/bin:$HOME/flutter/bin:$PATH"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

echo "=================================================================="
echo "🛠️  Go-Youtu 開發環境與 SDK / 模擬器 自動檢查與安裝工具 (setup.sh)"
echo "=================================================================="

# 1. 檢查 Homebrew
if ! command -v brew >/dev/null 2>&1; then
  echo "❌ 找不到 Homebrew (brew)。請先安裝 Homebrew: https://brew.sh"
  exit 1
fi
BREW_PREFIX="$(brew --prefix)"
echo "✅ Homebrew 已就緒: $BREW_PREFIX"

# 2. 檢查與安裝 OpenJDK 17
check_java() {
  if [ -d "$BREW_PREFIX/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" ]; then
    export JAVA_HOME="$BREW_PREFIX/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    export PATH="$JAVA_HOME/bin:$PATH"
  fi
  java -version >/dev/null 2>&1
}

if check_java; then
  echo "✅ Java JDK 已安裝: $(java -version 2>&1 | head -n 1)"
else
  if [ "$CHECK_ONLY" = true ]; then
    echo "⚠️  尚未安裝可用的 Java Runtime (OpenJDK 17)"
  else
    echo "📦 正在透過 Homebrew 安裝 OpenJDK 17..."
    brew install openjdk@17
    export JAVA_HOME="$BREW_PREFIX/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "✅ OpenJDK 17 安裝完成: $JAVA_HOME"
  fi
fi

# 3. 檢查與安裝 Flutter & Dart SDK
check_flutter() {
  if command -v flutter >/dev/null 2>&1; then
    return 0
  fi
  for candidate in "$HOME/development/flutter/bin" "$HOME/flutter/bin" "$BREW_PREFIX/Caskroom/flutter/latest/flutter/bin" "$BREW_PREFIX/share/flutter/bin"; do
    if [ -x "$candidate/flutter" ]; then
      export PATH="$candidate:$PATH"
      return 0
    fi
  done
  return 1
}

if check_flutter; then
  echo "✅ Flutter SDK 已安裝: $(command -v flutter)"
else
  if [ "$CHECK_ONLY" = true ]; then
    echo "⚠️  尚未安裝 Flutter / Dart SDK"
  else
    echo "📦 正在安裝 Flutter SDK..."
    if brew install --cask flutter; then
      echo "✅ 透過 Homebrew Cask 安裝 Flutter 成功"
    else
      echo "ℹ️  改用官方 Git Stable 分支安裝 Flutter 至 $HOME/development/flutter..."
      mkdir -p "$HOME/development"
      if [ ! -d "$HOME/development/flutter/.git" ]; then
        git clone https://github.com/flutter/flutter.git -b stable --depth 1 "$HOME/development/flutter"
      fi
      export PATH="$HOME/development/flutter/bin:$PATH"
    fi
    flutter --version
  fi
fi

# 4. 檢查與安裝 Android SDK、Emulator 與 System Image
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
export ANDROID_SDK_ROOT="$ANDROID_HOME"

ARCH_NAME="$(uname -m)"
if [[ "$ARCH_NAME" == "arm64" ]]; then
  SYS_IMAGE_PKG="system-images;android-34;google_apis;arm64-v8a"
  SYS_IMAGE_REL_DIR="system-images/android-34/google_apis/arm64-v8a"
else
  SYS_IMAGE_PKG="system-images;android-34;google_apis;x86_64"
  SYS_IMAGE_REL_DIR="system-images/android-34/google_apis/x86_64"
fi

# 確保 $ANDROID_HOME/cmdline-tools/latest 是實體目錄而非 symlink，
# 否則 avdmanager 會因為解析 symlink (pwd -P) 而找不到 $ANDROID_HOME/system-images
ensure_real_cmdline_tools_dir() {
  local brew_cmdline="$BREW_PREFIX/share/android-commandlinetools/cmdline-tools/latest"
  local target_cmdline="$ANDROID_HOME/cmdline-tools/latest"

  if [ -L "$target_cmdline" ]; then
    rm -f "$target_cmdline"
  fi
  if [ ! -d "$target_cmdline" ] && [ -d "$brew_cmdline" ]; then
    mkdir -p "$ANDROID_HOME/cmdline-tools"
    cp -R "$brew_cmdline" "$target_cmdline"
  fi
}

# 優先使用 $ANDROID_HOME/cmdline-tools/latest/bin 下的工具，確保與 $ANDROID_HOME 綁定一致
find_sdkmanager() {
  for candidate in \
    "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" \
    "$BREW_PREFIX/share/android-commandlinetools/cmdline-tools/latest/bin/sdkmanager"; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  if command -v sdkmanager >/dev/null 2>&1; then
    command -v sdkmanager
    return 0
  fi
  return 1
}

find_avdmanager() {
  for candidate in \
    "$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager" \
    "$BREW_PREFIX/share/android-commandlinetools/cmdline-tools/latest/bin/avdmanager"; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  if command -v avdmanager >/dev/null 2>&1; then
    command -v avdmanager
    return 0
  fi
  return 1
}

if [ "$CHECK_ONLY" = false ]; then
  ensure_real_cmdline_tools_dir
fi

SDKMANAGER_BIN="$(find_sdkmanager || true)"

if [ -n "$SDKMANAGER_BIN" ] && \
   [ -d "$ANDROID_HOME/platform-tools" ] && \
   [ -x "$ANDROID_HOME/emulator/emulator" ] && \
   [ -d "$ANDROID_HOME/$SYS_IMAGE_REL_DIR" ]; then
  echo "✅ Android SDK、Emulator 與系統映像檔已安裝: $ANDROID_HOME"
else
  if [ "$CHECK_ONLY" = true ]; then
    echo "⚠️  尚未完整安裝 Android SDK 或 Android Emulator ($ANDROID_HOME)"
  else
    if [ -z "$SDKMANAGER_BIN" ]; then
      echo "📦 正在安裝 Android Command-line Tools..."
      brew install --cask android-commandlinetools
    fi
    ensure_real_cmdline_tools_dir

    SDKMANAGER_BIN="$(find_sdkmanager)"
    echo "📦 正在透過 sdkmanager 安裝 Android platform-tools、Android 34、build-tools 與 Emulator 系統映像檔 ($SYS_IMAGE_PKG)..."
    yes | "$SDKMANAGER_BIN" --sdk_root="$ANDROID_HOME" --licenses >/dev/null 2>&1 || true
    "$SDKMANAGER_BIN" --sdk_root="$ANDROID_HOME" \
      "platform-tools" \
      "emulator" \
      "platforms;android-34" \
      "build-tools;34.0.0" \
      "$SYS_IMAGE_PKG"

    flutter config --android-sdk "$ANDROID_HOME" || true
    yes | flutter doctor --android-licenses >/dev/null 2>&1 || true
    echo "✅ Android SDK 與 Emulator 組件安裝完成"
  fi
fi

# 將 $ANDROID_HOME/cmdline-tools/latest/bin 排在 /opt/homebrew/bin 之前
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"

# 5. 檢查與自動建立預設 Android 模擬器 (AVD)
AVDMANAGER_BIN="$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager"
if [ ! -x "$AVDMANAGER_BIN" ]; then
  AVDMANAGER_BIN="$(find_avdmanager || true)"
fi

EXISTING_AVDS="$("$ANDROID_HOME/emulator/emulator" -list-avds 2>/dev/null || true)"

if [ -n "$EXISTING_AVDS" ]; then
  echo "✅ 已存在 Android 虛擬裝置 (AVD): $(echo "$EXISTING_AVDS" | tr '\n' ' ')"
else
  if [ "$CHECK_ONLY" = true ]; then
    echo "⚠️  尚未建立任何 Android 虛擬裝置 (AVD)"
  elif [ -n "$AVDMANAGER_BIN" ]; then
    echo "📱 正在建立預設 Android 模擬器 AVD ($DEFAULT_AVD_NAME)..."
    echo "no" | "$AVDMANAGER_BIN" create avd \
      --name "$DEFAULT_AVD_NAME" \
      --package "$SYS_IMAGE_PKG" \
      --device "pixel_6" \
      --force
    echo "✅ 預設 Android 模擬器建立完成: $DEFAULT_AVD_NAME"
  fi
fi

# 確保所有 AVD 皆啟用實體鍵盤輸入 (hw.keyboard=yes)，讓 Mac 主機鍵盤可直接輸入
for avd_cfg in "$HOME"/.android/avd/*.avd/config.ini; do
  if [ -f "$avd_cfg" ]; then
    sed -i '' 's/^hw\.keyboard=no/hw.keyboard=yes/' "$avd_cfg"
    grep -q "^hw\.keyboard=yes" "$avd_cfg" || echo "hw.keyboard=yes" >> "$avd_cfg"
  fi
done

# 6. 檢查與安裝 CocoaPods (供 iOS 建置使用)
if command -v pod >/dev/null 2>&1; then
  echo "✅ CocoaPods 已安裝"
else
  if [ "$CHECK_ONLY" = true ]; then
    echo "⚠️  尚未安裝 CocoaPods (iOS 編譯需要)"
  else
    echo "📦 正在透過 Homebrew 安裝 CocoaPods..."
    brew install cocoapods
    echo "✅ CocoaPods 安裝完成"
  fi
fi

if [ "$CHECK_ONLY" = true ]; then
  echo "=================================================================="
  echo "ℹ️  環境檢查結束。若要安裝缺少的組件，請直接執行: ./setup.sh"
  echo "=================================================================="
  exit 0
fi

# 7. 產生專案專屬環境變數檔 (.env.sdk) 並更新 ~/.zshrc
FLUTTER_BIN_DIR="$(dirname "$(command -v flutter)")"
JAVA_HOME_VAL="${JAVA_HOME:-$BREW_PREFIX/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home}"

cat > "$ENV_FILE" <<EOF
# Auto-generated by setup.sh
export JAVA_HOME="$JAVA_HOME_VAL"
export ANDROID_HOME="$ANDROID_HOME"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$FLUTTER_BIN_DIR:\$JAVA_HOME/bin:\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/emulator:\$ANDROID_HOME/platform-tools:/opt/homebrew/bin:/usr/local/bin:\$PATH"
EOF

if [ -f "$HOME/.zshrc" ] && ! grep -q "go-youtu/.env.sdk" "$HOME/.zshrc" 2>/dev/null; then
  {
    echo ""
    echo "# Go-Youtu Flutter & Android SDK Environment"
    echo "[ -f \"$ENV_FILE\" ] && source \"$ENV_FILE\""
  } >> "$HOME/.zshrc"
  echo "✅ 已將 SDK 環境變數設定加入 ~/.zshrc"
fi

# 8. 補齊專案 Android / iOS 原生目錄並下載 pub 套件
if [ ! -f "android/gradlew" ] || [ ! -f "ios/Runner.xcodeproj/project.pbxproj" ]; then
  echo "🔧 正在初始化專案 Android 與 iOS 原生平台檔案..."
  flutter create . --platforms=android,ios --project-name=go_youtu --org=com.example.goyoutu
fi

echo "📦 正在安裝專案 Dart/Flutter 依賴套件 (flutter pub get)..."
flutter pub get

echo "🩺 執行 flutter doctor 檢查環境狀態..."
flutter doctor

echo "=================================================================="
echo "🎉 開發環境安裝與專案初始化完成！"
echo "   - 啟動開發 (自動開啟 Android 模擬器並執行): ./dev.sh"
echo "   - 編譯 Android APK:                        ./build-android.sh"
echo "   - 編譯 iOS App:                            ./build-ios.sh"
echo "=================================================================="
