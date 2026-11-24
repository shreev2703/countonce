#!/usr/bin/env bash
# Fetch OneScene-Bench (arXiv 2610.38926; counting part built on DL3DV-10K).
# NOT run automatically. Release location, license and layout are all TODO: verify: check the
# paper's project / code link, put the URL(s) in DATA_URLS below. DL3DV-10K terms bind the images.
# Usage: scripts/download_onescene.sh <empty-target-dir>
set -euo pipefail

dest="${1:?usage: $0 <empty-target-dir>}"
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then
  echo "error: $dest is not empty" >&2
  exit 1
fi

DATA_URLS=()   # TODO: verify and fill from the paper's release link
if [ "${#DATA_URLS[@]}" -eq 0 ]; then
  echo "error: DATA_URLS is empty; locate the release first (TODO: verify)" >&2
  exit 1
fi
for u in "${DATA_URLS[@]}"; do
  curl -fL --retry 3 -C - -O --output-dir "$dest" "$u"
done

echo "Done. Unpack into a fresh directory, then check bench/datasets/onescene.py::_read_layout."
