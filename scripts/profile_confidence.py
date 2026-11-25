"""Time the pipeline stages on a window of Fuji frames, with recon/detect caches reused.

    .venv-gpu/bin/python scripts/profile_confidence.py [--frames 10] [--assoc v1] [--cprofile]

The first call fills the caches under runs/fuji_core/cache (GPU); later calls hit them, so the
confidence stage is measured on its own. Prints the per-stage timings and the CountResult range.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from pathlib import Path

from countonce.config import Config
from countonce.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="runs/fuji_core/img")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--assoc", default="v1")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cprofile", action="store_true")
    a = ap.parse_args()
    paths = sorted(Path(a.images).glob("*.jpg"))[: a.frames]
    cfg = Config(
        detector="gdino", backbone="vggt", assoc=a.assoc, max_frames=0, dedup_threshold=0,
        out_dir=Path("runs/fuji_core/out_profile"), cache_dir=Path("runs/fuji_core/cache"),
        chunked_recon=True, chunk_size=20, chunk_overlap=6,
    )  # fmt: skip
    res = run(paths, "apple", cfg, export=False)  # fills caches on the first call
    for i in range(a.repeats):
        pr = cProfile.Profile()
        if a.cprofile and i == 0:
            pr.enable()
        res = run(paths, "apple", cfg, export=False)
        if a.cprofile and i == 0:
            pr.disable()
            buf = io.StringIO()
            pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(25)
            print(buf.getvalue())
        print(f"repeat {i}: count={res.count} interval=({res.interval.low},{res.interval.high})",
              {k: round(v, 3) for k, v in res.timings.items()})  # fmt: skip


if __name__ == "__main__":
    main()
