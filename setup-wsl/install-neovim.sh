#!/bin/bash
# set exit when exception
set -e

source ./common.sh

if ! command -v unzip &> /dev/null; then
    echo ""
    echo "# update apt"
    sudo apt update
    sudo apt upgrade -y
    sudo apt install -y unzip
fi

if [ ! -f "/usr/local/bin/win32yank.exe" ]; then
    echo ""
    echo "# download win32yank"
    curl -sLo/tmp/win32yank.zip https://github.com/equalsraf/win32yank/releases/download/v0.1.1/win32yank-x64.zip
    unzip -p /tmp/win32yank.zip win32yank.exe > /tmp/win32yank.exe
    chmod +x /tmp/win32yank.exe
    sudo mv /tmp/win32yank.exe /usr/local/bin/
fi

if ! command -v rg &> /dev/null; then
    echo "install ripgrep"
    sudo apt-get install -y ripgrep
fi

if ! isFileExists /usr/share/autojump/autojump.sh; then
    echo "install autojump"
    sudo apt-get install -y autojump
    echo '. /usr/share/autojump/autojump.sh' >> ~/.bashrc
fi

# 0.4.0 version
# if ! command -v nvim &> /dev/null; then
#     echo ""
#     echo "install neovim..."
#     sudo sudo add-apt-repository ppa:neovim-ppa/stable
#     sudo sudo apt update
#     sudo sudo apt install neovim
#     pip install pynvim
# fi

if ! command -v nvim &> /dev/null; then
    echo ""
    echo "install neovim..."
    wget https://github.com/neovim/neovim/releases/download/v0.9.1/nvim-linux64.tar.gz
    sudo tar xzvf nvim-linux64.tar.gz
    sudo mv nvim-linux64 /usr/local/nvim
    rm ./nvim-linux64.tar.gz
    # 軟連接
    sudo ln -s /usr/local/nvim/bin/nvim /usr/bin/nvim
    nvim -v
fi


if [ ! -d "$HOME/.local/share/nvim/site/pack/packer" ]; then
    echo "install packer.nvim that plugin manager"
    git clone --depth 1 https://github.com/wbthomason/packer.nvim ~/.local/share/nvim/site/pack/packer/start/packer.nvim
fi

# if [ ! -d "$HOME/.config/nvim/packer/start" ]; then
#     echo "install Packer Manager"
#     ./create-path.sh ~/.config/nvim/packer/start
#     git clone --depth 1 https://github.com/wbthomason/packer.nvim ~/.config/nvim/packer/start/packer.nvim
# fi


echo "clean ~/.config/nvim"
if [ -d ~/.config/nvim/ ]; then
  rm -rf ~/.config/nvim/
fi

echo "create ~/.config/nvim"
./create-path.sh ~/.config/nvim/lua/

echo "copy init.vim to ~/.config/nvim/init.vim"
cp -Rf ./neovim-lua/* ~/.config/nvim

#nvim -c 'PackerInstall' -c 'qa!'
echo ""
echo "# please run nvim, then input :PackerInstall"
