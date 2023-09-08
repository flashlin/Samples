#!/bin/bash
# set exit when exception
set -e

echo "install nodejs"
source ~/.nvm/nvm.sh
nvm install 14.14.0
nvm use 14.14.0

echo "install prettierd"
npm install -g @fsouza/prettierd

echo "done. please re-login."