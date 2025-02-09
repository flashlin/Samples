# cuda 12.8

sudo apt-get --purge remove "*cublas*" "cuda*" "nvidia-*"
sudo apt-get autoremove
sudo apt-get clean


wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8


nvim ~/.bashrc
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}