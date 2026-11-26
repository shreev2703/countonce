"""Photo-order invariance on the Fuji-SfM test-tree windows for v1, v2, sync and v3.

Frozen settings (score 0.2, merge_k 0.3, v2/sync reprojection gate 0.5, v3 checkpoint as-is) on the
frozen windows at k=5 and 10. Two variants, one row per (variant, associator, window):

* ``permute``: one cached reconstruction per window; photo-to-frame assignment, camera arrays,
  frames and detection-list order are permuted before association.
* ``rerun_vggt``: the photos are uploaded to the pipeline in a new order, so VGGT is re-run on
  them (no recon cache). A second identity-order run is the control for run-to-run noise.

Resumable: finished windows are skipped. The CSV is rewritten after every window, in
``<work>/rows.csv``; it is copied to ``--out`` once every planned row exists.

    PYTHONPATH=scripts:. .venv-gpu/bin/python scripts/run_order_invariance_fuji_test.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from run_fuji_core import PROMPT, WORK, _window, base_config

import countonce.pipeline as pl
from bench import fuji_core as fc
from bench.datasets.fuji_sfm import load_fuji_sfm
from bench.order_invariance import (
    FUJI_CSV_FIELDS,
    common_agreement,
    distinct_permutations,
    fuji_window_row,
    partition,
    to_photo_ids,
)
from bench.runner import read_rows
from countonce.detect.cache import CachedDetector, DetectionCache
from countonce.recon.chunked import ChunkAlignmentError

ASSOCS = ("v1", "v2", "sync", "v3")
KS = (5, 10)
SCORE, MERGE_K = 0.2, 0.3
V3_CHECKPOINT = Path("models/checkpoints/v001.pt")
EXTRA_FIELDS = (
    "variant", "n_merges", "roi_gt_frame_failures", "bench_pred", "n_dets_min", "n_dets_max",
    "control_count_diff", "control_partition_changed",
)  # fmt: skip
FIELDS = (*FUJI_CSV_FIELDS, *EXTRA_FIELDS)


@dataclass
class Captured:
    lifted: list[Any]
    recon: Any
    frames: Any
    count: int
    cache_hit: bool
    rung: int


def capture(paths: list[Path], cfg: Any, det: Any, rec: Any) -> Captured:
    """Run the pipeline and keep the inputs of its *first* association (the frozen settings).

    Later ``_make_associator`` calls belong to the sensitivity sweep and use other thresholds.
    """
    seen: list[tuple[Any, Any, Any]] = []
    orig = pl._make_associator

    def patched(c: Any, k: float) -> Any:
        inner = orig(c, k)
        if seen:
            return inner

        class Spy:
            name = inner.name

            def associate(self, lifted: Any, recon: Any = None, frames: Any = None) -> Any:
                seen.append((list(lifted), recon, frames))
                return inner.associate(lifted, recon, frames)

        return Spy()

    pl._make_associator = patched
    try:
        res = pl.run(paths, PROMPT, cfg, detector=det, reconstructor=rec, export=False)
    finally:
        pl._make_associator = orig
    lifted, recon, frames = seen[0]
    if len(pl._make_base_associator(cfg, cfg.merge_k).associate(lifted, recon, frames).instances) \
            != res.count:  # fmt: skip
        raise RuntimeError("captured inputs do not reproduce the pipeline count")
    return Captured(lifted, recon, frames, res.count, res.recon_cache_hit,
                    res.oom_fallback_rung)  # fmt: skip


def associator(name: str) -> Any:
    extra = {"v3_checkpoint": V3_CHECKPOINT} if name == "v3" else {}
    cfg = base_config(assoc=name, score_threshold=SCORE, merge_k=MERGE_K, **extra)
    return pl._make_base_associator(cfg, MERGE_K)


def merges(part: frozenset[frozenset[str]]) -> int:
    return sum(len(g) - 1 for g in part)


def roi_counter(gt_ext: np.ndarray, box: Any, recon: Any) -> Callable[[Any], int | None]:
    try:
        sim = fc.align_window(gt_ext, recon.extrinsics)
    except ChunkAlignmentError:
        return lambda res: None

    def count(res: Any) -> int:
        if not res.instances:
            return 0
        xyz = sim.apply(np.array([i.xyz for i in res.instances], dtype=float))
        return int(np.asarray(fc.in_roi(xyz, box)).sum())

    return count


def permute_rows(
    cap: Captured, sigmas: list[np.ndarray], gt_ext: np.ndarray, box: Any, **ids: Any
) -> list[dict[str, Any]]:
    try:
        sim = fc.align_window(gt_ext, cap.recon.extrinsics)
    except ChunkAlignmentError:  # registration failure: no GT frame, so no ROI count
        sim = None
    rows = []
    for name in ASSOCS:
        row = fuji_window_row(
            name, associator(name), cap.lifted, cap.recon, sigmas=sigmas, reorder_list=True,
            to_gt=sim.apply if sim else None, in_roi=lambda x: fc.in_roi(x, box),
            frames=cap.frames, **ids,
        )  # fmt: skip
        if sim is None:
            row.update(roi_count_min="", roi_count_max="", roi_count_range="")
        base = associator(name).associate(list(cap.lifted), cap.recon, cap.frames)
        n_ok = sum(ld.ok for ld in cap.lifted)
        row.update(
            variant="permute",
            n_merges=merges(partition(base)),
            roi_gt_frame_failures=len(sigmas) + 1 if sim is None else 0,
            n_dets_min=n_ok,
            n_dets_max=n_ok,
        )
        rows.append(row)
    return rows


def rerun_rows(
    paths: list[Path],
    run_capture: Callable[[list[Path]], Captured],
    sigmas: list[np.ndarray],
    gt_ext: np.ndarray,
    box: Any,
    **ids: Any,
) -> list[dict[str, Any]]:
    """Upload the photos in each new order (``paths[inv]``) and compare with the identity upload.

    The identity upload is run twice; the second is the control.
    """
    identity = np.arange(len(paths))
    orders = [identity, identity, *(np.argsort(s) for s in sigmas)]  # inv[q] = photo at q
    runs: list[dict[str, Any]] = []
    for inv in orders:
        cap = run_capture([paths[int(i)] for i in inv])
        if cap.cache_hit or cap.rung != 0:
            raise RuntimeError(f"need a fresh full-quality recon (hit={cap.cache_hit}, "
                               f"oom rung={cap.rung})")  # fmt: skip
        roi = roi_counter(gt_ext[inv], box, cap.recon)
        per: dict[str, Any] = {"n_ok": sum(ld.ok for ld in cap.lifted)}
        for name in ASSOCS:
            res = associator(name).associate(list(cap.lifted), cap.recon, cap.frames)
            per[name] = (to_photo_ids(partition(res), inv), roi(res))
        runs.append(per)

    base, control, permuted = runs[0], runs[1], runs[2:]
    rows = []
    for name in ASSOCS:
        base_part = base[name][0]
        used = [base, *permuted]
        counts = [len(r[name][0]) for r in used]
        rois = [r[name][1] for r in used if r[name][1] is not None]
        changed = [r[name][0] for r in permuted if r[name][0] != base_part]
        n_oks = [r["n_ok"] for r in used]
        rows.append({
            "assoc": name, **ids, "n_perms": len(permuted), "n_dets": base["n_ok"],
            "base_count": counts[0], "count_min": min(counts), "count_max": max(counts),
            "count_range": max(counts) - min(counts),
            "roi_count_min": min(rois) if rois else "", "roi_count_max": max(rois) if rois else "",
            "roi_count_range": max(rois) - min(rois) if rois else "",
            "n_partition_changed": len(changed),
            "min_rand_index": min((common_agreement(base_part, p) for p in changed), default=1.0),
            "variant": "rerun_vggt", "n_merges": merges(base_part),
            "roi_gt_frame_failures": len(used) - len(rois),
            "n_dets_min": min(n_oks), "n_dets_max": max(n_oks),
            "control_count_diff": len(control[name][0]) - counts[0],
            "control_partition_changed": int(control[name][0] != base_part),
        })  # fmt: skip
    return rows


def in_subset(tree_index: int, side: str) -> bool:
    """Re-recon subset, fixed in advance: one side per tree, alternating east/west."""
    return side == ("east" if tree_index % 2 == 0 else "west")


def bench_preds(csv_path: Path, split: str) -> dict[tuple[str, str, str, int], str]:
    """Latest ok ``pred`` per (assoc, tree, side, k) in the benchmark CSV."""
    method = {m: a for m, a in fc.METHOD_ASSOC.items() if a in ASSOCS and "_colmap" not in m}
    out = {}
    for r in read_rows(csv_path):
        if r["status"] == "ok" and r["split"] == split and r["method"] in method:
            out[(method[r["method"]], r["scene_id"], r["sample"], int(r["n_views"]))] = r["pred"]
    return out


def load_done(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--variants", nargs="+", default=["permute", "rerun_vggt"],
                    choices=["permute", "rerun_vggt"])  # fmt: skip
    ap.add_argument("--n-perms", type=int, default=10)
    ap.add_argument("--work", type=Path, default=Path("runs/order_inv_test"))
    ap.add_argument("--out", type=Path, default=Path("results/order_invariance_fuji_test.csv"))
    ap.add_argument("--bench-csv", type=Path, default=Path("results/bench.csv"))
    args = ap.parse_args(argv)

    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    imgs = fc.prepare_images(scene, WORK / "img")
    ov = {s: json.loads((WORK / f"overview_{s}.json").read_text()) for s in ("east", "west")}
    trees = scene.trees[{"test": "test", "val": "training"}[args.split]]
    box = fc.roi_box(scene.gt_xyz)
    kw: dict[str, Any] = dict(assoc="v1", score_threshold=SCORE, merge_k=MERGE_K,
                              out_dir=args.work / "out")  # fmt: skip
    cached_cfg = base_config(**kw)
    fresh_cfg = base_config(**kw, cache_dir=None)  # no recon cache: VGGT runs every time
    det = pl.build_detector(cached_cfg)
    rec = pl.build_reconstructor(cached_cfg)
    fresh_det = CachedDetector(det, DetectionCache(WORK / "cache" / "detect"))
    preds = bench_preds(args.bench_csv, args.split)

    partial = args.work / f"rows_{args.split}.csv"
    rows = load_done(partial)
    done = {(r["variant"], r["tree"], r["side"], int(r["k"])) for r in rows}
    planned = [
        (variant, ti, tree, side, k)
        for variant in args.variants
        for k in KS
        for ti, tree in enumerate(trees)
        for side in ("east", "west")
        if variant == "permute" or in_subset(ti, side)
    ]
    for variant, ti, tree, side, k in planned:
        if (variant, tree, side, k) in done:
            continue
        _, _, paths, gt_ext = _window(scene, imgs, ov, side, tree, k)
        sigmas = distinct_permutations(k, args.n_perms, seed=k * 1000 + 2 * ti + (side == "west"))
        ids = {"tree": tree, "side": side, "k": k}
        if variant == "permute":
            cap = capture(paths, cached_cfg, det, rec)
            if not cap.cache_hit:
                raise RuntimeError(f"{tree}/{side}/k={k}: no cached benchmark recon")
            new = permute_rows(cap, sigmas, gt_ext, box, **ids)
        else:
            new = rerun_rows(paths, lambda p: capture(p, fresh_cfg, fresh_det, rec), sigmas,
                             gt_ext, box, **ids)  # fmt: skip
        for row in new:
            row["bench_pred"] = preds.get((row["assoc"], tree, side, k), "")
            print(row, flush=True)
        rows += new
        write_rows(partial, rows)

    shutil.copyfile(partial, args.out)
    print("complete:", len(rows), "rows ->", args.out, flush=True)


if __name__ == "__main__":
    main()
