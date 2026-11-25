"""Detector sanity of HF GroundingDINO-tiny on Fuji-SfM mask-set images, with per-image runtime.

Usage: python scripts/sanity_fuji_hf.py [--limit 20] [--split val] [--threshold 0.3]
Appends rows to results/detector_sanity.csv and prints a JSON summary. The first image
includes model load; the reported per-image time excludes it (warm-up image is run first).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from bench.detector_sanity import load_images, run_detector_sanity
from countonce.detect.groundingdino import GroundingDinoDetector


class Timed:
    def __init__(self, inner: GroundingDinoDetector) -> None:
        self.inner, self.times = inner, []  # type: ignore[var-annotated]
        self.name, self.version = inner.name, inner.version

    def config(self) -> dict[str, object]:
        return self.inner.config()

    def detect(self, frames, prompt, exemplars=None):  # type: ignore[no-untyped-def]
        t = time.perf_counter()
        out = self.inner.detect(frames, prompt, exemplars)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.perf_counter() - t)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/fuji_sfm")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("results/detector_sanity.csv"))
    a = ap.parse_args()
    imgs = load_images("fuji_mask", a.root, split=a.split, limit=a.limit)
    det = Timed(GroundingDinoDetector())
    # Warm-up (model load, CUDA init) on the first image, not counted.
    run_detector_sanity(det, imgs[:1], dataset="fuji_mask", split=a.split, out_csv="/dev/null")
    det.times.clear()
    s = run_detector_sanity(
        det, imgs, dataset="fuji_mask", split=a.split, score_threshold=a.threshold, out_csv=a.out
    )
    print(
        json.dumps(
            {
                "n": s.n,
                "mae": s.mae,
                "mae_ci": [s.mae_lo, s.mae_hi],
                "rmse": s.rmse,
                "mean_gt": sum(i.count for i in imgs) / len(imgs),
                "version": det.version,
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "sec_per_image_mean": statistics.mean(det.times),
                "sec_per_image_median": statistics.median(det.times),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
