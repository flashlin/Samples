#!/bin/bash
# 啟動 errexit , script will stop when exception
set -e

sudo apt-get update
sudo apt-get install python3-pip

echo 'export PATH="/mnt/d/VDisk/Github/Samples/bash:$PATH"' >> ~/.bashrc
chmod +x /mnt/d/VDisk/Github/Samples/bash/*.sh
echo "Done"