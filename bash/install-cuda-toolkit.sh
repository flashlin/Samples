#!/bin/bash
# set exit when exception
set -e

# https://developer.nvidia.com/cuda-downloads?target_os=Linux
# install CUDA for WSL or Ubuntu 
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run

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
