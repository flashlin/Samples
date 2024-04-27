#!/bin/bash
# set exit when exception
set -e

# 查看 Driver 版本
# NVIDIA-SMI 470.73  最高支持CUDA 11.4
# NVIDIA-SMI 537.13  CUDA Version: 12.2

cuda="cuda-12.2"

# https://developer.nvidia.com/cuda-downloads?target_os=Linux
# install CUDA for WSL or Ubuntu 
#wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run
#sudo sh cuda_11.4.0_470.42.01_linux.run

if [ ! -f "cuda_12.2.2_535.104.05_linux.run" ]; then
   echo "Downloading"
   wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
fi
echo "Installing"
sudo sh cuda_12.2.2_535.104.05_linux.run

# for WSL2
# 僅安裝 cuda-toolkit-12-x 元包

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

content="export PATH=\$PATH:/usr/local/$cuda/bin"
echo $content >> ~/.bashrc
content="export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/$cuda/lib64:/usr/lib/x86_64-linux-gnu"
echo $content >> ~/.bashrc

# sudo apt-get install -y build-essential