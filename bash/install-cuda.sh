# STEP1: 在windows 安裝驅動程式：GPU in Windows Subsystem for Linux (WSL)，不要在WSL 中安裝任何驅動程式
# STEP2: Install CUDA on WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4