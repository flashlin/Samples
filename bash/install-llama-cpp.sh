# https://developer.nvidia.com/cuda-downloads?target_os=Linux

# 安装好 CUDA Toolkit 后，屏幕上将输出：
#Driver:   Installed
#Toolkit:  Installed in /usr/local/cuda-10.1/
#Samples:  Installed in /home/abneryepku/
#
#Please make sure that
# -   PATH includes /usr/local/cuda-10.1/
# -   LD_LIBRARY_PATH includes /usr/local/cuda-10.1/lib64, or, add /usr/local/cuda-10.1/lib64 to /etc/ld.so.conf and run ldconfig as root
#export PATH=$PATH:/usr/local/cuda-10.1/bin
# add cuBLAS, cuSPARSE, cuRAND, cuSOLVER, cuFFT to path
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64:/usr/lib/x86_64-linux-gnu

# /usr/local/cuda-10.4/extras/demo_suite
# sudo apt-get --yes install build-essential



./install-OpenBLAS.sh
sudo apt-get install pkg-config

cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make clean
make LLAMA_OPENBLAS=1 LLAMA_CUBLAS=1
