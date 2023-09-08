#!/bin/bash
# set exit when exception
set -e

if [ $# -lt 1 ]; then
   echo "Usage: $0 <arg1>"
   exit 0
fi

action=$1

if [ "i" == "$action" ]; then
    echo "install torch env"
    conda create -n "torch" python=3.10
    conda activate torch
    exit 0
fi

if [ "cuda" == "$action" ]; then
    echo "install CUDA"
    conda install -c "nvidia/label/cuda-12.0.1" cuda
    exit 0
fi

if [ "t" == "$action" ]; then
    conda activate torch
    exit 0
fi

echo "unknown action"