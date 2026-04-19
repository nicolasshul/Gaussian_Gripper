#!/usr/bin/env bash

set -euo pipefail

SRC_DIR="/home/rm3/Pictures/Webcam"
DST_DIR="$HOME/stark_hacks/InstantSplatPP/assets/test/Sanjay/images"

i=1

find "$SRC_DIR" -maxdepth 1 -type f -name "*.jpg" -printf "%T@ %p\n" \
| sort -n \
| cut -d' ' -f2- \
| while read -r file; do
    printf -v newname "%04d.jpg" "$i"
    mv "$file" "$DST_DIR/$newname"
    ((i++))
done