#!/bin/bash
echo "list directory..."
# 確認是否有足夠參數數量
if [ $# -ne 1 ]; then
   # echo "Usage: $0 <arg1>"
   LS_COLORS='di=01;37' ls -d */
   exit 0
fi

pattern="$1"
find . -type d -name $pattern