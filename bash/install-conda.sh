#!/bin/bash
set -e

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

echo "export PATH=/home/flash/miniconda3/bin:$PATH"