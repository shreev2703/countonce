#!/usr/bin/env bash
# Download Fuji-SfM (Zenodo 10.5281/zenodo.3712808, ~6.9 GB).
# NOT run automatically. License: CC BY-NC-SA 4.0 (LICENSE.txt in the record): research use only,
# no commercial use, do not redistribute the images or derived GT.
# Usage: scripts/download_fuji_sfm.sh <empty-target-dir>
set -euo pipefail

dest="${1:?usage: $0 <empty-target-dir>}"
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then
  echo "error: $dest is not empty" >&2
  exit 1
fi

record=3712808
api="https://zenodo.org/api/records/${record}"

# File list comes from the Zenodo API; nothing from the archive is executed or imported here.
curl -fsSL "$api" -o "$dest/.record.json"
python3 -I - "$dest/.record.json" > "$dest/.files.txt" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    rec = json.load(f)
for e in rec["files"]:
    print(e["key"], e["links"]["self"], e.get("checksum", ""), sep="\t")
PY

while IFS=$'\t' read -r key url checksum; do
  case "$key" in */*|..*) echo "refusing unsafe file name: $key" >&2; exit 1;; esac
  echo "downloading $key"
  curl -fL --retry 3 -C - "$url" -o "$dest/$key"
  if [[ "$checksum" == md5:* ]]; then
    echo "${checksum#md5:}  $dest/$key" | md5sum -c -
  fi
done < "$dest/.files.txt"

echo "Done. Unpack the .rar into a fresh directory, then see bench/datasets/fuji_sfm.py for the verified layout."
