#!/bin/bash
# set exit when exception
set -e

if [ $# -lt 1 ]; then
   nvidia-smi
   conda env list
   echo "Usage: $0 <arg1>"
   echo "i        :install torch env"
   echo "b <name> :switch env <name>"
   echo "n <name> :creaate new env <name>"
   echo "cuda     :install cuda"
   exit 0
fi

action=$1

if [ "i" == "$action" ]; then
    echo "install torch env"
    conda create -n "torch" python=3.10
    conda activate torch
    # conda install conda=23.7.3
    exit
fi


if [ "n" == "$action" ]; then
    name=$2
    echo "create $name env"
    conda create -n $name python=3.10
    conda activate $name
    exit
fi


if [ "b" == "$action" ]; then
    name=$2
    echo "switch $name env"
    conda activate $name
    exit
fi



if [ "cuda" == "$action" ]; then
    echo "install torch"
    # conda install -c "nvidia/label/cuda-12.0.1" cuda

    # start install cuda toolkit
    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    # sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    # sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    # sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    # sudo apt-get update
    # sudo apt-get -y install cuda

    # start install pytorch
    # ~/miniconda3/bin/conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    # ~/miniconda3/bin/conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers
    exit
fi

if [ "t1" == "$action" ]; then
    echo ""
    conda info | grep -i 'base environment'
    source /home/flash/miniconda3/etc/profile.d/conda.sh
    /home/flash/miniconda3/bin/activate
    conda init bash
    exit
fi

if [ "t" == "$action" ]; then
    echo ""
    conda info | grep -i 'base environment'
    source /home/flash/miniconda3/etc/profile.d/conda.sh
    conda env list
    echo "conda activate torch"
    /home/flash/miniconda3/bin/activate
    conda init bash
    conda activate torch
    exit
fi

echo "unknown action"
