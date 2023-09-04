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
curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim


if ! command -v nvim &> /dev/null; then
    echo ""
    echo "# install neovim..."
    sudo add-apt-repository ppa:neovim-ppa/stable
    sudo apt update
    sudo apt install neovim
fi

echo "# copy init.vim to ~/.config/nvim/init.vim"
cp -Rf ./neovim-data/* ~/.config/nvim

echo ""
echo "# please run nvim, then input :PlugInstall"
