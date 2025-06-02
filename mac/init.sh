#!/bin/bash

echo "🚀 開始安裝 zsh + fzf + atuin + autosuggestions + syntax-highlighting..."

# 安裝 Homebrew（如果尚未安裝）
if ! command -v brew &> /dev/null; then
  echo "🍺 安裝 Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

eval "$(/opt/homebrew/bin/brew shellenv)"

# 安裝 zsh（macOS 通常已內建）
brew install zsh

# 設為預設 shell
if [[ "$SHELL" != *zsh ]]; then
  echo "⚙️ 將 zsh 設為預設 shell..."
  chsh -s "$(which zsh)"
fi

# 安裝 fzf 並啟用 key bindings 和自動補全
brew install fzf
"$(brew --prefix)"/opt/fzf/install --all --no-bash --no-fish

# 安裝 atuin
brew install atuin

# 加入 atuin 初始化到 zshrc（避免重複加入）
if ! grep -q "atuin init zsh" ~/.zshrc; then
  echo 'eval "$(atuin init zsh)"' >> ~/.zshrc
fi

# 安裝 zsh-autosuggestions
brew install zsh-autosuggestions

# 安裝 zsh-syntax-highlighting
brew install zsh-syntax-highlighting

# 加入 plugins 到 zshrc（避免重複）
if ! grep -q "zsh-autosuggestions" ~/.zshrc; then
  echo 'source /opt/homebrew/share/zsh-autosuggestions/zsh-autosuggestions.zsh' >> ~/.zshrc
fi

if ! grep -q "zsh-syntax-highlighting" ~/.zshrc; then
  echo 'source /opt/homebrew/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh' >> ~/.zshrc
fi

# 匯入歷史記錄到 atuin
atuin import auto

# 檢查是否已安裝 Hammerspoon，若未安裝則自動下載並解壓縮安裝
HAMMERSPOON_APP="/Applications/Hammerspoon.app"
if [ ! -d "$HAMMERSPOON_APP" ]; then
  echo "⚠️ 尚未偵測到 Hammerspoon，開始下載並安裝..."
  TMP_ZIP="/tmp/Hammerspoon-1.0.0.zip"
  curl -L -o "$TMP_ZIP" "https://github.com/Hammerspoon/hammerspoon/releases/download/1.0.0/Hammerspoon-1.0.0.zip"
  unzip -q "$TMP_ZIP" -d /Applications/
  rm "$TMP_ZIP"
  echo "✅ Hammerspoon 已安裝完成！"
fi

### ✅ 檢查 Raycast ###
echo ""
echo "🔍 檢查是否已安裝 Raycast..."
if [ ! -d "/Applications/Raycast.app" ]; then
  echo "⚠️ 尚未偵測到 Raycast"
  echo "👉 你可以從以下網址下載並安裝 Raycast："
  echo "   🔗 https://www.raycast.com/download"
  echo "在搜尋欄位當中輸入”Settings” 或者”General”，找到 Raycast settings 的 General 就可以進入設定畫面"
fi

echo "✅ 安裝完成！請重新開啟 Terminal 或執行 'exec zsh' 以啟用所有功能。"
