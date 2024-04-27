wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12

content="export PATH=\$PATH:/usr/local/$cuda/bin"
echo $content >> ~/.bashrc
content="export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/$cuda/lib64:/usr/lib/x86_64-linux-gnu"
echo $content >> ~/.bashrc
