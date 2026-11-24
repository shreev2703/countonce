#!/usr/bin/env bash
# Fetch WGISD (Embrapa grape clusters, ~1.3 GB). NOT run automatically.
# License is CONFLICTING (CC BY-NC-SA 4.0 vs Embrapa README): research only, read the card first.
# Usage: scripts/download_wgisd.sh <empty-target-dir>
set -euo pipefail

dest="${1:?usage: $0 <empty-target-dir>}"
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then
  echo "error: $dest is not empty" >&2
  exit 1
fi

REPO="${WGISD_REPO:-thsant/wgisd}"   # TODO: verify mirror; canonical is Zenodo 3361736
huggingface-cli download "$REPO" --repo-type dataset --local-dir "$dest"

echo "Done. Check bench/datasets/wgisd.py::_read_layout against the downloaded tree."
