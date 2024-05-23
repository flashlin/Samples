#!/bin/bash
# 確認是否有足夠參數數量
if [ $# -ne 1 ]; then
   echo "Usage: $0 <url>"
   exit 1
fi

file_url="$1"
# https://github.com/neovim/neovim/releases/download/stable/nvim.appimage

file=$(basename "$file_url")
if [ -f "$file" ]; then
  echo "$file already exists, skipping download."
  exit 0
fi

wget -c $file_url