#!/bin/bash
find /mnt/d/demo/srt -type f -name "*.srt" | while read -r file; do
   python ./translate-srt.py "$file"
done
