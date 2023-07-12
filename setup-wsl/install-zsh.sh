#!/bin/bash
# set exit when exception
set -e

if grep -q "^[interop]" "/etc/wsl.conf"; then
  echo "exists"
else
  content = $(cat ./wsl.conf)
  echo "$content" | sudo tee --append /etc/wsl.conf
fi

if ! command -v zsh &> /dev/null; then
    sudo apt update
    sudo apt install zsh
    chsh -s $(which zsh)
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
fi