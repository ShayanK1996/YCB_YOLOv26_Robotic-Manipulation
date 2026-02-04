#!/usr/bin/env bash
# Extract YCB Berkeley .tgz archives in data/raw/ycb/berkeley.
# Skips objects that already have extracted images.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${ROOT}/data/raw/ycb/berkeley"
cd "$RAW"
for dir in */; do
  obj="${dir%/}"
  tgz=$(find "$dir" -maxdepth 1 -name "*.tgz" 2>/dev/null | head -1)
  if [[ -n "$tgz" ]]; then
    n=$(find "$dir" -name "*.jpg" 2>/dev/null | wc -l)
    if [[ "${n:-0}" -lt 10 ]]; then
      echo "Extracting $obj ..."
      tar -xzf "$tgz" -C "$dir"
    else
      echo "Skip $obj (already has $n images)"
    fi
  fi
done
echo "Done."
