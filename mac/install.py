import os
from pathlib import Path

# 所有模板內容
templates_content = '''
# ===== 自定義函數區塊開始 =====

# nvm 自動切換版本
autoload -U add-zsh-hook

load-nvmrc() {
  local nvmrc_path
  nvmrc_path="$(nvm_find_nvmrc)"

  if [ -n "$nvmrc_path" ]; then
    local nvmrc_node_version
    nvmrc_node_version=$(nvm version "$(cat "${nvmrc_path}")")

    if [ "$nvmrc_node_version" = "N/A" ]; then
      nvm install
    elif [ "$nvmrc_node_version" != "$(nvm version)" ]; then
      nvm use
    fi
  elif [ -n "$(PWD=$OLDPWD nvm_find_nvmrc)" ] && [ "$(nvm version)" != "$(nvm version default)" ]; then
    echo "Reverting to nvm default version"
    nvm use default
  fi
}

add-zsh-hook chpwd load-nvmrc
load-nvmrc

# 目錄列表指令
ld() {
  setopt NULL_GLOB  # 設定空匹配返回空而不是錯誤
  if [ $# -eq 0 ]; then
    dirs=(*/)
    if [ ${#dirs[@]} -eq 0 ]; then
      echo "\033[37m當前目錄下沒有子目錄\033[0m"  # 使用白色顯示
    else
      ls -d --color=auto */
    fi
  else
    pattern="$*"
    dirs=(*/)
    if [ ${#dirs[@]} -eq 0 ]; then
      echo "\033[37m當前目錄下沒有子目錄\033[0m"  # 使用白色顯示
    else
      ls -d */ | grep --color=auto -E "$pattern"
    fi
  fi
  unsetopt NULL_GLOB  # 恢復原始設定
}

# 檔案搜尋函數
ff() {
  if [ $# -lt 2 ]; then
    echo "用法: ff 'regex' '*.ext'"
    return 1
  fi

  pattern="$1"
  shift

  find . -type f \( $(printf "! -name %s " "$@") -o -false \) -prune -o -type f \( $(printf "-name %s -o " "$@") -false \) -print \\
    | xargs grep --color=always -n -E "$pattern" 2>/dev/null
}

# Rider 快速開啟指令
ro() { open -a "Rider" "${1:-.}"; }

# Python 快速執行指令
py() {
  if [ $# -eq 0 ]; then
    echo "用法: py <檔案路徑> [參數...]"
    return 1
  fi
  python "$@"
}

# ===== 自定義函數區塊結束 =====
'''

# 取得 .zshrc 的完整路徑
zshrc_path = os.path.expanduser('~/.zshrc')

# 檢查檔案是否存在
if not os.path.exists(zshrc_path):
    print(f"找不到 {zshrc_path} 檔案")
    exit(1)

# 讀取現有的內容
with open(zshrc_path, 'r', encoding='utf-8') as file:
    current_content = file.read()

# 定義區塊的起始和結束標記
start_marker = "# ===== 自定義函數區塊開始 ====="
end_marker = "# ===== 自定義函數區塊結束 ====="

# 檢查是否已經存在區塊
if start_marker in current_content and end_marker in current_content:
    print("找到現有的函數區塊，正在更新...")
    # 找到區塊的起始和結束位置
    start_pos = current_content.find(start_marker)
    end_pos = current_content.find(end_marker) + len(end_marker)
    # 移除舊的區塊
    new_content = current_content[:start_pos] + current_content[end_pos:]
    # 在檔案末尾添加新的區塊
    with open(zshrc_path, 'w', encoding='utf-8') as file:
        file.write(new_content + templates_content)
    print("已更新函數區塊")
else:
    print("未找到函數區塊，正在添加...")
    # 在檔案末尾添加新的區塊
    with open(zshrc_path, 'a', encoding='utf-8') as file:
        file.write(templates_content)
    print("已添加函數區塊")

print("完成所有更新，請執行 'source ~/.zshrc' 來套用新的設定")
