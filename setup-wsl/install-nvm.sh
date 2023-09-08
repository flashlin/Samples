#!/bin/bash
# set exit when exception
set -e

# 加载 nvm 配置
if ! grep -q "export NVM_DIR" ~/.bashrc; then
    echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.bashrc
    echo 'source "$NVM_DIR/nvm.sh"' >> ~/.bashrc
    source ~/.bashrc 
fi    
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

echo "checking nvm..."
exists=$(command -v nvm | tr -d '\n') # 去除換行
if [ -z "$exists" ]; then # 檢查是否為空的
    echo "nvm command not found. Installing nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
    export NVM_DIR="$HOME/.nvm" >> ~/.bashrc
    source ~/.bashrc 
    echo "please relogin bash shell to use nvm!"
fi

echo "done"