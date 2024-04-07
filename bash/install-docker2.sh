#!/bin/bash
set -e

sudo apt-get install ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg 
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null 
sudo apt-get update 

# install docker
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin 

# start docker service
sudo service docker start 

sudo usermod -aG docker $USER
newgrp docker

sudo apt-get install -y nvidia-docker2

# echo "sudo vim /etc/wsl.conf"
# echo "添加下面內容"
# echo "[boot]"
# echo 'command="service docker start"'