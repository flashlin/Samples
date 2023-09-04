#!/bin/bash
# set exit when exception
set -e


if ! command -v dos2unix &>/dev/null; then
    echo "install dos2unix..."
    sudo apt-get update
    sudo apt-get install dos2unix 
fi

find . -type f -name "*.sh" | while read -r file; do
    echo "Converting $file to LF format..."
    dos2unix "$file"
done
echo "Conversion complete."
