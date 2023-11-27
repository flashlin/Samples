#!/bin/bash
# set exit when exception
set -e

# 查看 Driver 版本
# NVIDIA-SMI 470.73  最高支持CUDA 11.4
# NVIDIA-SMI 537.13  CUDA Version: 12.2

# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

# 安装好 CUDA Toolkit 后，屏幕上将输出：
#Driver:   Installed
#Toolkit:  Installed in /usr/local/cuda-10.1/
#Samples:  Installed in /home/abneryepku/
#
#Please make sure that
# -   PATH includes /usr/local/cuda-10.1/
# -   LD_LIBRARY_PATH includes /usr/local/cuda-10.1/lib64, or, add /usr/local/cuda-10.1/lib64 to /etc/ld.so.conf and run ldconfig as root
# export PATH=$PATH:/usr/local/cuda-10.1/bin
# add cuBLAS, cuSPARSE, cuRAND, cuSOLVER, cuFFT to path
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64:/usr/lib/x86_64-linux-gnu

# content="export PATH=\$PATH:/usr/local/$cuda/bin"
# echo $content >> ~/.bashrc
# content="export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/$cuda/lib64:/usr/lib/x86_64-linux-gnu"
# echo $content >> ~/.bashrc
