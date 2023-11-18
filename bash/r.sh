#!/bin/bash
set -e

sh_home="/mnt/d/VDisk/Github/Samples/bash"

params="$@"
script_name="$1"
shift
script_params="$@" 

if [ ! -f "$sh_home/$script_name.sh" ]; then
   echo "$sh_home/$script_name.sh not exists."
   exit
fi

$sh_home/$script_name.sh $script_params