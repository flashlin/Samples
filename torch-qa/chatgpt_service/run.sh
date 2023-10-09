#!/bin/bash
set -e

if [ $# -ne 1 ]; then
   echo "Usage: "
   echo "run.sh <app_name>"
   exit
fi

app=$1
streamlit run $app
