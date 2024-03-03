#!/bin/bash
dirs=$(find . -type d)
selected_dir=$(echo "$dirs" | fzf --query=$1)

if [ -n "$selected_dir" ]; then
  cd "$selected_dir"
  echo "Changed directory to: $selected_dir"
else
  echo "No directory selected."
fi