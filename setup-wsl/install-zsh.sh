#!/bin/bash
# set exit when exception
set -e

# if grep -q "^[interop]" "/etc/wsl.conf"; then
#   echo "exists"
# else
#   echo ""
#   echo "# speed wsl..."
#   content=$(cat ./wsl.conf)
#   echo "$content" | sudo tee --append /etc/wsl.conf
# fi

# if ! command -v zsh &> /dev/null; then
#     sudo apt update
#     sudo apt install zsh
#     chsh -s $(which zsh)
#     sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
#     exit
# fi

# 安裝 zim
if command -v zim &> /dev/null; then
  echo "Zim installed"
else
  echo "install Zim"
  curl -fsSL https://raw.githubusercontent.com/zimfw/install/master/install.zsh | zsh
  exit
fi

# 
if grep -q "^zmodule romkatv/powerlevel10k" "~/.zimrc"; then
  echo "exists"
else
  echo "install Powerlevel10k"
  echo "zmodule romkatv/powerlevel10k" >> ~/.zimrc
  zimfw install
  exit
fi

p10k configure