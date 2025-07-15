#!/bin/bash

# 檢查 miniconda 是否已安裝
if [ ! -d ~/miniconda3 ]; then
    echo "正在安裝 Miniconda..."
    mkdir -p ~/miniconda3
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh

    source ~/miniconda3/bin/activate
    conda init --all
fi

brew install zsh-autosuggestions
uv run ./install.py

echo "Install done"