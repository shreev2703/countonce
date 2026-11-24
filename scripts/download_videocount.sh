#!/usr/bin/env bash
# Fetch the CountVid / VideoCount benchmark (github.com/niki-amini-naieni/CountVid).
# NOT run automatically. Release layout, size and data hosting are TODO: verify: read the repo
# README first and put the real data URL(s) in DATA_URLS below. Source-video licenses apply
# (TAO, MOT20 is research-use only); do not redistribute.
# Usage: scripts/download_videocount.sh <empty-target-dir>
set -euo pipefail

dest="${1:?usage: $0 <empty-target-dir>}"
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then
  echo "error: $dest is not empty" >&2
  exit 1
fi

DATA_URLS=()   # TODO: verify and fill from the CountVid README (files hosted outside git?)
if [ "${#DATA_URLS[@]}" -eq 0 ]; then
  echo "error: DATA_URLS is empty; see github.com/niki-amini-naieni/CountVid (TODO: verify)" >&2
  exit 1
fi
for u in "${DATA_URLS[@]}"; do
  curl -fL --retry 3 -C - -O --output-dir "$dest" "$u"
done

echo "Done. Unpack into a fresh directory, then check bench/datasets/videocount.py::_read_layout."
