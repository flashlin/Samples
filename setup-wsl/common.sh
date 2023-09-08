#!/bin/bash

isFileExists() {
    file_path=$1
    if [ -e "$file_path" ]; then
        return 0
    else
        return 1
    fi
}