#!/usr/bin/env bash
# Fetch FSC-147 (Ranjan et al., CVPR 2021) from a Hugging Face mirror. NOT run automatically.
# Mirror layout, size (~5.2 GB) and licence are TODO: verify; the images are scraped web images,
# so use for detector sanity checks only and never redistribute. Read the dataset card first.
# Usage: scripts/download_fsc147.sh <empty-target-dir>
set -euo pipefail

dest="${1:?usage: $0 <empty-target-dir>}"
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then
  echo "error: $dest is not empty" >&2
  exit 1
fi

REPO="${FSC147_REPO:-isentropic/FSC147}"   # TODO: verify mirror and its terms
huggingface-cli download "$REPO" --repo-type dataset --local-dir "$dest"

echo "Done. Check bench/datasets/fsc147.py::_read_layout against the downloaded tree."
