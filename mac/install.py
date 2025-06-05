import os
from pathlib import Path

# 要添加的 nvm 內容
content_to_add = '''
# place this after nvm initialization!
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
'''

# 要添加的目錄列表指令
dir_list_content = '''
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
'''

# 要添加的檔案搜尋函數
file_find_content = '''
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
'''

# 要添加的 Rider 快速開啟指令
rider_content = '''
# Rider 快速開啟指令
ro() { open -a "Rider" "${1:-.}"; }
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

# 檢查是否已經包含要添加的內容
changes_made = False

if content_to_add.strip() not in current_content:
    print("未找到 nvm 設定，正在添加...")
    with open(zshrc_path, 'a', encoding='utf-8') as file:
        file.write(content_to_add)
    changes_made = True
    print("已成功添加 nvm 設定到 .zshrc")
else:
    print("nvm 設定已存在於 .zshrc 中")

if "ld() {" not in current_content:
    print("未找到目錄列表指令，正在添加...")
    with open(zshrc_path, 'a', encoding='utf-8') as file:
        file.write(dir_list_content)
    changes_made = True
    print("已成功添加目錄列表指令到 .zshrc")
else:
    print("目錄列表指令已存在於 .zshrc 中")

if "ff() {" not in current_content:
    print("未找到檔案搜尋函數，正在添加...")
    with open(zshrc_path, 'a', encoding='utf-8') as file:
        file.write(file_find_content)
    changes_made = True
    print("已成功添加檔案搜尋函數到 .zshrc")
else:
    print("檔案搜尋函數已存在於 .zshrc 中")

if "ro() {" not in current_content:
    print("未找到 Rider 快速開啟指令，正在添加...")
    with open(zshrc_path, 'a', encoding='utf-8') as file:
        file.write(rider_content)
    changes_made = True
    print("已成功添加 Rider 快速開啟指令到 .zshrc")
else:
    print("Rider 快速開啟指令已存在於 .zshrc 中")

if changes_made:
    print("完成所有更新，請執行 'source ~/.zshrc' 來套用新的設定")
