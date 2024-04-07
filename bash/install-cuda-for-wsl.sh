#!/bin/bash
#set -e

if [ ! -f "cuda-keyring_1.1-1_all.deb" ]; then
   echo "Downloading"
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
fi
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

content='
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
'

# 检查~/.bashrc文件是否已经存在内容，如果不存在，则将内容添加到文件末尾
if ! grep -qxF "$content" ~/.bashrc; then
    echo "$content" >> ~/.bashrc
    echo "内容已成功添加到 ~/.bashrc 文件。"
else
    echo "内容已經存在於 ~/.bashrc 文件中。"
fi

nvidia-smi
nvcc -V

