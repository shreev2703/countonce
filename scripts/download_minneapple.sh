#!/usr/bin/env bash
# Fetch MinneApple (UMN Digital Conservancy handle 11299/206575). NOT run automatically.
# Archives are separate (counting ~977 MB, detection 1.7 GB, test labels 1.17 GB). The
# Conservancy needs a terms-of-use click-through, so the bitstream URLs are TODO: verify and
# supplied via MINNEAPPLE_URLS (space-separated). License: UMN ToU, research only.
# Usage: MINNEAPPLE_URLS="<url> ..." scripts/download_minneapple.sh <empty-target-dir>
set -euo pipefail

dest="${1:?usage: $0 <empty-target-dir>}"
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then
  echo "error: $dest is not empty" >&2
  exit 1
fi
: "${MINNEAPPLE_URLS:?set MINNEAPPLE_URLS to the archive URLs (TODO: verify from the handle page)}"

for url in $MINNEAPPLE_URLS; do
  f="$dest/$(basename "$url")"
  curl -fL --output "$f" "$url"
  case "$f" in
    *.tar.gz|*.tgz) tar -xzf "$f" -C "$dest" ;;
    *.zip) unzip -q "$f" -d "$dest" ;;
  esac
done

echo "Done. Check bench/datasets/minneapple.py::_read_layout against the downloaded tree."
