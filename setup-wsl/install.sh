#!/bin/bash
# 啟動 errexit , script will stop when exception
set -e

sudo apt-get update
sudo apt-get install python3-pip


if [[ ":$PATH:" != *":/mnt/d/VDisk/GitHub/Samples/bash:"* ]]; then
    echo "Adding /mnt/d/VDisk/.../bash to PATH..."
    echo 'export PATH="/mnt/d/VDisk/GitHub/Samples/bash:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    echo "PATH updated."
fi

#chmod +x /mnt/d/VDisk/Github/Samples/bash/*.sh
echo "Done"