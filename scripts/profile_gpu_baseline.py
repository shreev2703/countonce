"""Baseline GPU profile of the real VGGT backbone at several frame counts (PLAN §13 item 1).

Builds a scratch scene per frame count from ``examples/tiny_scene`` (views are cycled when N is
more than the 5 bundled views, so N=10 repeats each view twice; reconstruction time and VRAM
depend on N, not on content), writes a profile config derived from ``configs/profile.yaml``
(backbone vggt, detector color_blob, dedup off so every frame is kept) and runs
``countonce profile`` on it with a wall-clock cap. Stops at the first OOM and records it.

Run from the repo root inside the GPU venv:
``HF_HUB_OFFLINE=1 python scripts/profile_gpu_baseline.py``
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_scene(n: int, dest: Path) -> Path:
    views = sorted((ROOT / "examples" / "tiny_scene").glob("view_*.png"))
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for i in range(n):
        shutil.copy(views[i % len(views)], dest / f"frame_{i:02d}.png")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "profile_gpu")
    ap.add_argument("--cap-s", type=float, default=1200.0, help="total wall-clock cap")
    args = ap.parse_args()

    base = yaml.safe_load((ROOT / "configs" / "profile.yaml").read_text())
    base.pop("variants", None)
    base.pop("cache", None)
    base["prompt"] = "red ball"
    base["run"] = {
        **base["run"],
        "detector": "color_blob",
        "backbone": "vggt",
        "dedup_threshold": 0.0,
    }
    start = time.monotonic()
    summary: list[dict[str, object]] = []
    for n in args.frames:
        left = args.cap_s - (time.monotonic() - start)
        if left <= 0:
            summary.append({"frames": n, "status": "skipped: wall-clock cap reached"})
            break
        scene = build_scene(n, args.out / f"scene_{n}")
        cfg = {
            **base,
            "inputs": str(scene),
            "out_csv": str(args.out / f"profile_{n}.csv"),
        }
        cfg["run"] = {**base["run"], "max_frames": n}
        cfg_path = args.out / f"profile_{n}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "countonce.cli", "profile", "--config", str(cfg_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=left,
            )
        except subprocess.TimeoutExpired:
            summary.append({"frames": n, "status": "timeout: wall-clock cap reached"})
            break
        tail = (proc.stdout + proc.stderr)[-600:]
        if proc.returncode == 0:
            summary.append({"frames": n, "status": "ok", "csv": cfg["out_csv"]})
            continue
        oom = "ERR_GPU_OOM" in tail or "out of memory" in tail.lower()
        summary.append({"frames": n, "status": "oom" if oom else "error", "tail": tail})
        break  # first OOM (or error) ends the sweep; an OOM is a result, not a failure
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
