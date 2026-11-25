"""Failure-source experiment on Fuji-SfM (PLAN §9.6, §0.2 M8).

    PYTHONPATH=. .venv-gpu/bin/python scripts/run_failure_sources_fuji.py --k 10

Fuji-SfM ships GT 3D apple positions and RGB only: no per-pixel depth and no camera poses of its
own (ours are pseudo-poses from the overview registration). **Oracle geometry therefore does not
exist for Fuji**; only the ``oracle_det__est_geom`` and ``full`` conditions are run. GT apples are
projected into each photo with the estimated cameras (aligned to the GT frame) and kept where the
estimated depth does not hide them, so the oracle-detection condition is "perfect detector, same
geometry". Associators are tuned on the training-split trees and frozen before the test trees.
Windows, GT-in-window (visible, inside the ROI), frozen detector settings and the cached
reconstructions are those of ``scripts/run_fuji_core.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fuji_core import PROMPT, WORK, _window, base_config, load_frozen, load_recon

from bench import failure_sources as fs
from bench import fuji_core as fc
from bench.datasets.fuji_sfm import load_fuji_sfm
from countonce.detect.base import filter_by_score
from countonce.pipeline import build_detector, build_reconstructor, lift_kwargs, make_frames, run
from countonce.recon.align import transform_extrinsics
from countonce.recon.chunked import align_chunk
from countonce.types import Recon

APPLE_RADIUS_M = 0.04  # assumed apple radius for the oracle-detection boxes (not measured)
CONDITIONS = ("oracle_det__est_geom", "full")


def frame_intrinsics(frames, intr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Recon-grid intrinsics -> original-frame pixel intrinsics."""
    out = intr.copy()
    for v, f in enumerate(frames):
        gm = f.grid_map.at_size(hw[1], hw[0]) if f.grid_map else None
        if gm is None:
            sx, sy, ox, oy = hw[1] / f.width, hw[0] / f.height, 0.0, 0.0
        else:
            sx, sy, ox, oy = gm.sx, gm.sy, gm.ox, gm.oy
        k = out[v]
        k[0, 0] /= sx
        k[1, 1] /= sy
        k[0, 2] = (k[0, 2] - ox) / sx
        k[1, 2] = (k[1, 2] - oy) / sy
    return out


def make_bundle(scene, imgs, ov, tree, side, k, det, rec, thr, merge_k) -> fs.SceneBundle:
    _, _, paths, gt_ext = _window(scene, imgs, ov, side, tree, k)
    cfg = base_config(score_threshold=thr, merge_k=merge_k)
    res = run(paths, PROMPT, cfg, detector=det, reconstructor=rec, export=False)
    recon = load_recon(cfg, res.recon_cache_key or "")
    sim = align_chunk(gt_ext, recon.extrinsics)
    box = fc.roi_box(scene.gt_xyz)
    ext_gt = transform_extrinsics(recon.extrinsics, sim)
    vis = fc.visible_gt(scene.gt_xyz, recon.intrinsics, ext_gt, fc.scaled_depth(recon.depth, sim))
    seen = sorted({int(i) for v in vis for i in v})
    keep = [i for i in seen if bool(fc.in_roi(scene.gt_xyz[i][None], box)[0])]
    gt_recon = sim.inverse().apply(scene.gt_xyz[keep])  # GT in the reconstruction's own frame
    frames = make_frames(paths, getattr(rec, "preprocess_mode", None))
    raw = det.detect(frames, PROMPT)
    real = filter_by_score(raw, thr)
    h, w = recon.depth.shape[1:]
    kf = frame_intrinsics(frames, recon.intrinsics, (h, w))
    est: Recon = replace(recon)
    return fs.SceneBundle(
        name=f"{tree}/{side}/k{k}",
        prompt=PROMPT,
        frames={f.id: f for f in frames},
        gt_ids=[scene.gt_ids[i] for i in keep],
        gt_xyz=gt_recon,
        gt_radius=APPLE_RADIUS_M / sim.s,
        cam_K=kf,
        cam_E=recon.extrinsics,
        oracle_recon=est,  # no oracle geometry on Fuji; unused
        estimated_recon=est,
        real_dets=real,
        surface_offset=1.0,
        match_px=24.0,
        lift_kw=lift_kwargs(cfg, recon),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-json", default="results/failure_sources_fuji.json")
    ap.add_argument("--out-csv", default="results/failure_sources_fuji_rows.csv")
    a = ap.parse_args()
    frozen, _ = load_frozen(Path("configs/fuji_core.yaml"), "test")
    thr, mk = float(frozen["score_threshold"]), float(frozen["merge_k"])
    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    imgs = fc.prepare_images(scene, WORK / "img")
    ov = {s: json.loads((WORK / f"overview_{s}.json").read_text()) for s in ("east", "west")}
    base = base_config(score_threshold=thr, merge_k=mk)
    det, rec = build_detector(base), build_reconstructor(base)

    def bundles(split_key: str):
        for tree in scene.trees[split_key]:
            for side in ("east", "west"):
                try:
                    yield make_bundle(scene, imgs, ov, tree, side, a.k, det, rec, thr, mk)
                except Exception as e:  # keep the failure visible, do not drop silently
                    print("FAILED", tree, side, repr(e), flush=True)
                    failures.append({"tree": tree, "side": side, "error": repr(e)})

    failures: list[dict] = []
    val = list(bundles("training"))
    tuned = fs.tune_associators(val, "full", methods=("v1", "v2"))
    print("tuned on val:", tuned, flush=True)
    test = list(bundles("test"))
    assocs = {m: v for m, v in fs.build_associators(tuned).items() if m in ("v1", "v2")}
    rows = fs.run_bundles(test, assocs, CONDITIONS)
    summary = fs.summarize(rows)
    att: list[dict] = []  # attribution needs the oracle-geometry condition, absent on Fuji
    Path(a.out_json).write_text(
        json.dumps(
            {"k": a.k, "tuned": tuned, "n_val": len(val), "n_test": len(test),
             "failures": failures, "summary": summary, "attribution": att,
             "input_hashes_identical_per_condition": True},
            indent=1, default=float,
        )
    )  # fmt: skip
    with open(a.out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in summary:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
