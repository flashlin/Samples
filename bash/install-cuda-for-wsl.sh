#!/bin/bash
#set -e

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

export CUDA_VERSION=12.4
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# CUDA SDK PATH
# export CUDA_HOME=/usr/local/cuda
# export PATH=${CUDA_HOME}/bin:${PATH}
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
# source ~/.bashrc

#export BNB_CUDA_VERSION=117
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tim/local/cuda-11.7
# export BNB_CUDA_VERSION=121
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tim/local/cuda-12.1