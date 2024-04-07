#!/bin/bash
set -e

if [ ! -f "Miniconda3-latest-Linux-x86_64.sh" ]; then
   echo "Downloading"
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi

./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

echo 'export PATH=/home/flash/miniconda3/bin:$PATH' >> ~/.bashrc
