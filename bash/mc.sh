#!/bin/bash
# set exit when exception
set -e

if [ $# -lt 1 ]; then
   echo "Usage: $0 <arg1>"
   echo "i    :install torch env"
   echo "cuda :install cuda"
   exit 0
fi

action=$1

if [ "i" == "$action" ]; then
    echo "install torch env"
    conda create -n "torch" python=3.10
    conda activate torch
    # conda install conda=23.7.3
    exit 0
fi

if [ "cuda" == "$action" ]; then
    echo "install CUDA"
    # conda install -c "nvidia/label/cuda-12.0.1" cuda
    # conda install -c "nvidia/label/cuda-11.0.1" cuda
    ~/miniconda3/bin/conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    exit 0
fi

if [ "t" == "$action" ]; then
    echo ""
    ~/miniconda3/bin/activate torch
    exit 0
fi

echo "unknown action"