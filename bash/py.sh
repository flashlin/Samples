#!/bin/bash
# set exit when exception
set -e

export PYTHONPATH="/mnt/d/VDisk/GitHub/Samples/py_standard"
echo "export PYTHONPATH=\"$PYTHONPATH\""

# 確認是否有足夠參數數量
if [ $# -ne 1 ]; then
   echo "Usage: "
   echo "py.sh <xxx.py>"
   exit 0
fi

