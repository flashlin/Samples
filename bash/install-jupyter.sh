#!/bin/bash
set -e

conda install -c conda-forge jupyterlab

#Windows Example
# jupyter lab --notebook-dir=E:/ --preferred-dir E:/Documents/Somewhere/Else
#Linux Example
# jupyter lab --notebook-dir=/var/ --preferred-dir /var/www/html/example-app/
jupyter lab --notebook-dir=/mnt/d/VDisk/Github/Samples/torch-qa/ipy/ --preferred-dir /mnt/d/VDisk/Github/Samples/torch-qa/ipy/