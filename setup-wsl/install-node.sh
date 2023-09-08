#!/bin/bash
# set exit when exception
set -e

source ~/.nvm/nvm.sh

nvm install 14.14.0
nvm use 14.14.0

npm install -g @fsouza/prettierd

echo ""
echo "install node done"