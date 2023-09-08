#!/bin/bash
# set exit when exception
set -e
source ~/.nvm/nvm.sh

desired_version="14.14.0"
installed_version=$(nvm ls | grep "$desired_version")
if [[ -z "$installed_version" ]]; then
    echo "install node 14.14.0"
    nvm install 14.14.0
fi
nvm use 14.14.0

npm install -g @fsouza/prettierd
echo "done"