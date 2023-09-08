#!/bin/bash
# 啟動 errexit , script will stop when exception
set -e

sudo apt-get update
sudo apt-get install -y python3-pip

# echo ""
# echo "install locales"
# sudo apt-get install -y locales
# echo "en_US.UTF-8 UTF-8" | sudo tee /etc/locale.gen
# sudo dpkg-reconfigure --frontend=noninteractive locales
# sudo update-locale LANG=en_US.UTF-8


./add-env-path.sh /mnt/d/VDisk/GitHub/Samples/bash
./install-nvm.sh
./install-node.sh
./install-neovim.sh
./setup-bash.sh

#chmod +x /mnt/d/VDisk/Github/Samples/bash/*.sh
echo "install done"
