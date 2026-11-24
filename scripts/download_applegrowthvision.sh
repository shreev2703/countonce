#!/usr/bin/env bash
# Fetch AppleGrowthVision (Fraunhofer HHI, CVPRW 2025) from a Hugging Face mirror. NOT run
# automatically. The mirror is FiftyOne-format and the size and the original license are
# TODO: verify (HF copy says CC BY 4.0); research only. The 10 manually counted trees may need
# the authors' page (fraunhoferhhi.github.io/AppleGrowthVision) instead.
# Usage: scripts/download_applegrowthvision.sh <empty-target-dir>
set -euo pipefail

dest="${1:?usage: $0 <empty-target-dir>}"
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then
  echo "error: $dest is not empty" >&2
  exit 1
fi

REPO="${AGV_REPO:-Voxel51/AppleGrowthVision}"   # TODO: verify mirror and layout
huggingface-cli download "$REPO" --repo-type dataset --local-dir "$dest"

echo "Done. Check bench/datasets/applegrowthvision.py::_read_* against the downloaded tree."
