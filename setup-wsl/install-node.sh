#!/bin/bash
# set exit when exception
set -e

echo ""
echo "# install nvm..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

if grep -q "^NVM_DIR" "~/.zshrc"; then
  echo "NVM exists"
else
  echo ""
  echo "# setup nvm env..."
  content=$(cat ./nvm.zshrc)
  echo "$content" | sudo tee --append ~/.zshrc
fi

source ~/.zshrc
nvm install 14.14.0
nvm use 14.14.0

echo "reload env..."
source ~/.bashrc

echo "done"
