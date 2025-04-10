#!/bin/bash

# Usage: ./find_small_folders.sh /path/to/folder

target_dir=$1

if [ -z "$target_dir" ]; then
  echo "Usage: $0 /path/to/folder"
  exit 1
fi

echo "Folders under $target_dir smaller than 1GB:"

# Loop through each subdirectory
du -sh "$target_dir"/*/ 2>/dev/null | while read -r size path; do
  size_in_bytes=$(du -sb "$path" | cut -f1)
  if [ "$size_in_bytes" -lt 1073741824 ]; then  # 1 GB = 1024^3 bytes
    echo "$path ($size)"
  fi
done
