#!/usr/bin/env bash
# Download RGBD-VideoCount (HF dataset aerospace123/RGBD-VideoCount, CC BY 4.0).
# NOT run automatically. The repo is a single ~2.7 GB RGBD-VideoCount.zip (file tree checked
# 2026-11-24, metadata only). Cite Xu et al., "Depth-Guided Video Object Counting in Crowded Scenes".
# Usage: scripts/download_rgbd_videocount.sh <empty-target-dir>
set -euo pipefail

dest="${1:?usage: $0 <empty-target-dir>}"
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then
  echo "error: $dest is not empty" >&2
  exit 1
fi

url="https://huggingface.co/datasets/aerospace123/RGBD-VideoCount/resolve/main/RGBD-VideoCount.zip"
curl -fL --retry 3 -C - "$url" -o "$dest/RGBD-VideoCount.zip"

echo "Done. Unzip into a fresh directory (never run anything from inside it), then check the"
echo "layout against bench/datasets/rgbd_videocount.py::_read_layout (carries the TODO: verify items)."
