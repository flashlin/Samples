#!/bin/bash
set -e

if [ $# -lt 1 ]; then
   git status
   echo "Usage: $0 <arg1>"
   echo "ig <folder>  : add ignore folder"
   exit 0
fi

action=$1

if [ "ig" == "$action" ]; then
    folder=$2
    if [ ! -f .gitignore ]; then
        touch .gitignore
    fi
    git add .gitignore
    echo $folder >> .gitignore
    msg="Add $folder into .gitignore file"
    echo $msg
    git commit -m $msg
fi