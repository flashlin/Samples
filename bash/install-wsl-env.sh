bash_home=/mnt/d/VDisk/Github/Samples/bash
if ! grep -q "#set env" ~/.bashrc; then
   echo "alias mc='$bash_home/mc.sh'" >> ~/.bashrc
   echo "alias py='$bash_home/py.sh'" >> ~/.bashrc
fi
