#!/bin/bash
# set exit when exception
set -e

if ! command -v unzip &> /dev/null; then
    echo ""
    echo "# update apt"
    sudo apt update
    sudo apt upgrade
    sudo apt install unzip
fi

if [ ! -f "/usr/local/bin/win32yank.exe" ]; then
    echo ""
    echo "# download win32yank"
    curl -sLo/tmp/win32yank.zip https://github.com/equalsraf/win32yank/releases/download/v0.1.1/win32yank-x64.zip
    unzip -p /tmp/win32yank.zip win32yank.exe > /tmp/win32yank.exe
    chmod +x /tmp/win32yank.exe
    sudo mv /tmp/win32yank.exe /usr/local/bin/
fi    

# install vim-plug
# curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

if ! command -v nvim &> /dev/null; then
    echo ""
    echo "# install neovim..."
    sudo sudo add-apt-repository ppa:neovim-ppa/stable
    sudo sudo apt update
    sudo sudo apt install neovim
    pip install pynvim
fi


echo "install packer.nvim that plugin manager"
if [ -d ~/.config/nvim/site/pack/packer/ ]; then
    git clone --depth 1 https://github.com/wbthomason/packer.nvim \
        ~/.local/share/nvim/site/pack/packer/start/packer.nvim
fi


echo "clean ~/.config/nvim"
if [ -d ~/.config/nvim/ ]; then
  rm -rf ~/.config/nvim/
fi

echo "create ~/.config"
if [ ! -d ~/.config/ ]; then
    mkdir ~/.config/
fi
if [ ! -d ~/.config/nvim/ ]; then
    mkdir ~/.config/nvim/
fi
if [ ! -d ~/.config/nvim/lua ]; then
    mkdir ~/.config/nvim/lua/
fi

echo "~/.config/nvim/lua/plugins.lua"


echo "copy init.vim to ~/.config/nvim/init.vim"
cp -Rf ./neovim-data/* ~/.config/nvim

nvim -c 'PlugInstall' -c 'qa!'
echo ""
echo "# please run nvim, then input :PlugInstall"
