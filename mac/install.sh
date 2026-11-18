#!/bin/bash

# 詢問使用者家目錄路徑（用於 install.py 中的 alias 路徑）
DEFAULT_BASE_PATH="/Users/flash/"
read -r -p "請輸入使用者家目錄路徑 [預設: $DEFAULT_BASE_PATH]: " BASE_PATH
BASE_PATH="${BASE_PATH:-$DEFAULT_BASE_PATH}"
# 確保結尾有 /
[[ "$BASE_PATH" != */ ]] && BASE_PATH="$BASE_PATH/"
echo "使用路徑: $BASE_PATH"

# 檢查 miniconda 是否已安裝
if [ ! -d ~/miniconda3 ]; then
    echo "正在安裝 Miniconda..."
    mkdir -p ~/miniconda3
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh

    source ~/miniconda3/bin/activate
    conda init --all
fi

# 檢查 nvm 是否已安裝
if [ ! -d "$HOME/.nvm" ]; then
    echo "正在安裝 nvm..."
    PROFILE=/dev/null bash -c 'curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash'
fi

# 檢查當下的 shell 並添加 nvm 設定
CURRENT_SHELL=$(basename "$SHELL")
case "$CURRENT_SHELL" in
    zsh)
        SHELL_RC="$HOME/.zshrc"
        ;;
    bash)
        if [ -f "$HOME/.bashrc" ]; then
            SHELL_RC="$HOME/.bashrc"
        else
            SHELL_RC="$HOME/.bash_profile"
        fi
        ;;
    *)
        SHELL_RC="$HOME/.profile"
        ;;
esac

touch "$SHELL_RC"

if ! grep -q 'export NVM_DIR="\$HOME/.nvm"' "$SHELL_RC"; then
    echo "正在將 nvm 設定加入 $SHELL_RC..."
    cat << 'EOF' >> "$SHELL_RC"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
EOF
fi

# 載入 nvm 並檢查是否已安裝 Node.js，若無則安裝最新 LTS 版本
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

if [ "$(nvm current)" = "none" ]; then
    echo "正在安裝 Node.js LTS..."
    nvm install --lts
fi

brew install zsh-autosuggestions
brew install ripgrep
brew install fd
brew install uv

# 檢查 zoxide 是否已安裝
if ! command -v zoxide &> /dev/null; then
    echo "正在安裝 zoxide..."
    brew install zoxide
fi

# 檢查並設置 ~/.zshrc 以啟用 zoxide (使用 j 作為快速指令)
ZSHRC="$HOME/.zshrc"
touch "$ZSHRC"
if ! grep -q 'zoxide init' "$ZSHRC"; then
    echo "正在將 zoxide 設定加入 $ZSHRC..."
    cat << 'EOF' >> "$ZSHRC"

# zoxide (j)
eval "$(zoxide init zsh --cmd j)"
EOF
fi

uv run ./install.py "$BASE_PATH"

echo "Install done"
echo "請執行 'source ~/.zshrc' 或重新開啟終端機以使設定生效"
