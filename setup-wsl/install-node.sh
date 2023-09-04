#!/bin/bash
# set exit when exception
set -e

exists=$(command -v nvm)
if [ -n "$exists" ]; then #檢查是否為空
    echo "nvm command not found. Installing nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
    source ~/.bashrc 
    echo "nvm installed successfully."
fi

echo "install nodejs"
nvm install 14.14.0
nvm use 14.14.0

echo "reload env..."
source ~/.bashrc
echo "done"