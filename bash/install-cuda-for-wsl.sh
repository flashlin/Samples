#!/bin/bash
#set -e
sudo apt-get install -y gcc

# Install CUDA
if [ ! -f "cuda-keyring_1.1-1_all.deb" ]; then
   echo "Downloading"
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
fi
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4


content='
export PATH=/usr/local/cuda-12.4/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-12.4
'

# 检查~/.bashrc文件是否已经存在内容，如果不存在，则将内容添加到文件末尾
if ! grep -qxF "$content" ~/.bashrc; then
    echo "$content" >> ~/.bashrc
    echo "内容已成功添加到 ~/.bashrc 文件。"
else
    echo "内容已經存在於 ~/.bashrc 文件中。"
fi


# Install cuDNN
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker


nvidia-smi
nvcc -V

