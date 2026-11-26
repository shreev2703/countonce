"""Photo-order invariance on real Fuji-SfM windows (PLAN 0.6, M4 follow-up). Validation trees only.

For each validation tree x side x k in {5, 10}: take the window's cached reconstruction and
detections (``runs/fuji_core/cache``; the detector is only used on a cache miss), lift once, and
run ``bench.order_invariance.fuji_window_row`` for v1, v2 and sync with >= 10 photo permutations.
Also re-runs v2 on the same windows with the old frame-id visit order and the new camera-centre
order and writes the count-MAE change to ``results/order_invariance_fuji_v2_mae.csv``.

    PYTHONPATH=. python scripts/run_order_invariance_fuji.py --n-perms 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from run_fuji_core import PROMPT, WORK, _window, base_config, load_recon

import countonce.merge.match as match_mod
import countonce.pipeline as pl
from bench import fuji_core as fc
from bench.datasets.fuji_sfm import load_fuji_sfm
from bench.order_invariance import FUJI_CSV_FIELDS, fuji_window_row
from countonce.pipeline import build_detector, build_reconstructor, run
from countonce.recon.chunked import align_chunk

_NEW_ORDER = match_mod.canonical_frame_order


def _id_order(frame_ids: Any, recon: Any, frame_to_view: Any = None) -> list[int]:
    return sorted(set(frame_ids))


class _Capture:
    """Associator stand-in that records its inputs and delegates to the real one."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.name, self.calls = inner, inner.name, []

    def associate(self, lifted: Any, recon: Any = None, frames: Any = None) -> Any:
        self.calls.append((list(lifted), recon))
        return self.inner.associate(lifted, recon, frames)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perms", type=int, default=10)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--score-threshold", type=float, default=0.2)
    ap.add_argument("--merge-k", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("results/order_invariance_fuji.csv"))
    ap.add_argument(
        "--mae-out", type=Path, default=Path("results/order_invariance_fuji_v2_mae.csv")
    )
    args = ap.parse_args(argv)

    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    imgs = fc.prepare_images(scene, WORK / "img")
    ov = {s: json.loads((WORK / f"overview_{s}.json").read_text()) for s in ("east", "west")}
    trees = scene.trees["training"]  # the 'val' split; test trees are never touched
    box = fc.roi_box(scene.gt_xyz)
    kw = dict(score_threshold=args.score_threshold, merge_k=args.merge_k)
    base = base_config(**kw)
    det, rec = build_detector(base), build_reconstructor(base)

    rows: list[dict[str, Any]] = []
    mae: list[dict[str, Any]] = []
    for k in args.ks:
        for tree in trees:
            for side in ("east", "west"):
                start, stop, paths, gt_ext = _window(scene, imgs, ov, side, tree, k)
                cap = None
                orig = pl._make_associator

                def patched(cfg: Any, mk: float, _o: Any = orig) -> Any:
                    nonlocal cap
                    if cap is not None:  # later calls are sensitivity-sweep cells
                        return _o(cfg, mk)
                    cap = _Capture(_o(cfg, mk))
                    return cap

                pl._make_associator = patched
                try:
                    cfg1 = base_config(assoc="v1", **kw)
                    res1 = run(paths, PROMPT, cfg1, detector=det, reconstructor=rec, export=False)
                finally:
                    pl._make_associator = orig
                assert cap is not None
                lifted, recon = cap.calls[0]
                if recon.n_views != k:
                    raise RuntimeError(f"recon has {recon.n_views} views, expected {k}")
                sim = align_chunk(gt_ext, recon.extrinsics)
                for name in ("v1", "v2", "sync"):
                    cfg = base_config(assoc=name, **kw)
                    row = fuji_window_row(
                        name, pl._make_base_associator(cfg, cfg.merge_k), lifted, recon,
                        tree=tree, side=side, k=k, n_perms=args.n_perms,
                        seed=k * 1000 + len(rows),
                        to_gt=sim.apply, in_roi=lambda x: fc.in_roi(x, box),
                    )  # fmt: skip
                    rows.append(row)
                    print(row, flush=True)
                # v2 count error: old frame-id visit order vs the new camera-centre order
                cfg2 = base_config(assoc="v2", **kw)
                rec_pred: dict[str, int] = {}
                for label, fn in (("id_order", _id_order), ("camera_order", _NEW_ORDER)):
                    match_mod.canonical_frame_order = fn
                    try:
                        res = run(paths, PROMPT, cfg2, detector=det, reconstructor=rec,
                                  export=False)  # fmt: skip
                    finally:
                        match_mod.canonical_frame_order = _NEW_ORDER
                    r2 = load_recon(cfg2, res.recon_cache_key or "")
                    s2 = align_chunk(gt_ext, r2.extrinsics)
                    ev = fc.evaluate_window(res, r2, s2, scene.gt_ids, scene.gt_xyz, box, gt_ext)
                    rec_pred[label] = ev["pred"]
                    gt = ev["gt"]
                mae.append({"tree": tree, "side": side, "k": k, "gt": gt,
                            "pred_id_order": rec_pred["id_order"],
                            "pred_camera_order": rec_pred["camera_order"]})  # fmt: skip
                print(mae[-1], flush=True)
                del res1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FUJI_CSV_FIELDS))
        w.writeheader()
        w.writerows(rows)
    with args.mae_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mae[0]))
        w.writeheader()
        w.writerows(mae)
    if mae:
        for lab in ("pred_id_order", "pred_camera_order"):
            print(lab, "MAE", float(np.mean([abs(m[lab] - m["gt"]) for m in mae])))


if __name__ == "__main__":
    main()
