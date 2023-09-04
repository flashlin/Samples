#!/bin/bash
# 啟動 errexit , script will stop when exception
set -e
./download.sh https://github.com/neovim/neovim/releases/download/stable/nvim-linux64.tar.gz
tar xzvf nvim-linux64.tar.gz
sudo cp nvim-linux64/bin/nvim /usr/local/bin -r
rm ./nvim-linux64.tar.gz
pip3 install pynvim
echo "done"