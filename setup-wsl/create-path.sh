#!/bin/bash

# 檢查是否提供了路徑參數
if [ $# -ne 1 ]; then
    echo "Please provide a path, ex: \"/demo/sample\""
    exit 1
fi

path=$1
eval path="$path"
# 分割路徑成為陣列
IFS='/' read -ra path_parts <<< "$path"

# 初始化目前路徑
current_path=""

# 依序檢查和建立路徑
for part in "${path_parts[@]}"; do
    current_path="$current_path$part"
    if [ -n "$current_path" ] && [ ! -d "$current_path" ]; then
        echo "mkdir $current_path"
        mkdir "$current_path"
    fi
    current_path="$current_path/"
done

echo "done"

