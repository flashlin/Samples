#!/bin/bash
# 確認是否有足夠參數數量
if [ $# -ne 1 ]; then
   echo "Usage: $0 <url>"
   exit 1
fi

file="$1"
# https://github.com/neovim/neovim/releases/download/stable/nvim.appimage

wget -c $file