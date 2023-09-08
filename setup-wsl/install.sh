#!/bin/bash
# 啟動 errexit , script will stop when exception
set -e

sudo apt-get update
sudo apt-get install python3-pip

./add-env-path.sh /mnt/d/VDisk/GitHub/Samples/bash
./install-node.sh
./install-neovim.sh

#chmod +x /mnt/d/VDisk/Github/Samples/bash/*.sh
echo "Done"