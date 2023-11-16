#!/bin/bash
set -e

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker $USER

docker --version
docker compose version

# for WSL2 old version
# sudo update-alternatives --config iptables