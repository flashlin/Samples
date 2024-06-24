#!/bin/bash
# 啟動 errexit , script will stop when exception
set -e
sudo rm -rf /opt/nvim

#./download.sh https://github.com/neovim/neovim/releases/download/stable/nvim-linux64.tar.gz
./download.sh https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz
#tar xzvf nvim-linux64.tar.gz
#sudo cp nvim-linux64/bin/nvim /usr/local/bin -r
sudo tar -C /opt -xzf nvim-linux64.tar.gz

if grep -q 'export PATH="$PATH:/opt/nvim-linux64/bin"' ~/.bashrc; then
    echo "nvim bin already in PATH. No changes made."
else
    echo 'export PATH="$PATH:/opt/nvim-linux64/bin"' >> ~/.bashrc
    echo "Added nvim bin to PATH in ~/.bashrc"
fi

if grep -q 'export VIMRUNTIME=/opt/nvim-linux64/share/nvim/runtime' ~/.bashrc; then
    echo "VIMRUNTIME in bashrc. No changes made."
else
    echo 'export VIMRUNTIME=/opt/nvim-linux64/share/nvim/runtime' >> ~/.bashrc
    echo "Added VIMRUNTIME in ~/.bashrc"
fi

#rm ./nvim-linux64.tar.gz
#pip3 install pynvim

mkdir -p ~/.config
cp -r neovim-files/* ~/.config/nvim/
echo "done"