"""Choose Fuji Core thresholds on the training-split trees only (Tree09-11, ``val``).

Coordinate sweep, each stage fixing the previous winner: detector box threshold, ROI margin,
v1 ``merge_k``, v2 reprojection-gate scale. Also sweeps the registration inlier distance on the
per-side overview. Reads the recon/detect caches under ``runs/fuji_core/cache`` and never loads a
test-split tree. Writes ``runs/fuji_core/tune.json``; ``--write-config`` writes the YAML.

    PYTHONPATH=. python scripts/tune_fuji_core.py --budget-min 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import countonce.pipeline as pl
from bench import fuji_core as fc
from bench.datasets.fuji_sfm import load_fuji_sfm
from countonce.pipeline import build_detector, build_reconstructor, run
from countonce.recon.align import camera_centers, transform_extrinsics
from countonce.recon.chunked import align_chunk
from run_fuji_core import PROMPT, STRIDE, WORK, _window, base_config, load_recon

THR = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
MARGIN = [0.3, 0.5, 0.7, 1.0]
MERGE_K = [0.25, 0.35, 0.5, 0.75, 1.0]
V2_SCALE = [0.5, 0.75, 1.0, 1.5, 2.0]
INLIER = [0.04, 0.06, 0.08, 0.12, 0.16]
KS = (5, 10)
_V2_ORIG = pl.V2Config


def set_v2_scale(scale: float) -> None:
    def make(**kw: Any) -> Any:
        if kw and "max_reproj_norm" in kw:
            kw["max_reproj_norm"] = kw["max_reproj_norm"] * scale
        return _V2_ORIG(**kw)

    pl.V2Config = make  # type: ignore[misc,assignment]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=12.0)
    ap.add_argument("--write-config", type=Path, default=None)
    args = ap.parse_args()
    t0 = time.time()
    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    trees = scene.trees["training"]
    assert set(trees) == {"Tree09", "Tree10", "Tree11"}, trees
    imgs = fc.prepare_images(scene, WORK / "img")
    ov = {s: json.loads((WORK / f"overview_{s}.json").read_text()) for s in ("east", "west")}
    box0 = fc.roi_box(scene.gt_xyz)
    det, rec = build_detector(base_config()), build_reconstructor(base_config())
    out: dict[str, Any] = {"trees": trees}

    # registration inlier distance, per side, from the (cached) overview reconstruction
    reg_rows = []
    for side in ("east", "west"):
        ids = scene.side_indices[side]
        local = np.arange(0, len(ids), STRIDE)
        cfg = base_config(chunk_size=16, chunk_overlap=6)
        res = pl.run([imgs[ids[i]] for i in local], PROMPT, cfg, detector=det,
                     reconstructor=rec, export=False)  # fmt: skip
        src = np.stack([i.xyz for i in res.instances if i.n_views >= 2])
        ref = fc.register_to_gt(src, scene.gt_xyz, 0.08)
        for m in INLIER:
            r = fc.register_to_gt(src, scene.gt_xyz, m)
            reg_rows.append({"side": side, "inlier_m": m, "inlier_frac": r.inlier_frac,
                             "median_err_m": r.median_err_m, "scale": r.sim.s,
                             "candidate": r.candidate, "same_cand_as_0.08": r.candidate == ref.candidate,
                             "scale_vs_0.08": r.sim.s / ref.sim.s})  # fmt: skip
        print("reg", side, flush=True)
    out["registration"] = reg_rows

    wins: list[dict[str, Any]] = []
    for tree in trees:
        for side in ("east", "west"):
            for k in KS:
                start, stop, paths, gt_ext = _window(scene, imgs, ov, side, tree, k)
                cfg = base_config(assoc="v1")
                res = run(paths, PROMPT, cfg, detector=det, reconstructor=rec, export=False)
                recon = load_recon(cfg, res.recon_cache_key or "")
                sim = align_chunk(gt_ext, recon.extrinsics)
                c_gt = camera_centers(gt_ext)
                c_al = camera_centers(transform_extrinsics(recon.extrinsics, sim))
                cam_rms = float(np.sqrt(((c_gt - c_al) ** 2).sum(1).mean()))
                wins.append({"tree": tree, "side": side, "k": k, "paths": paths, "recon": recon,
                             "sim": sim, "cam_rms_m": cam_rms})  # fmt: skip
                print("window", tree, side, k, f"{time.time() - t0:.0f}s", flush=True)
                if (time.time() - t0) / 60 > args.budget_min:
                    raise SystemExit("budget hit while building windows; rerun (caches are warm)")
    out["windows"] = [{"tree": w["tree"], "side": w["side"], "k": w["k"],
                       "cam_rms_m": w["cam_rms_m"]} for w in wins]  # fmt: skip

    memo: dict[tuple[Any, ...], dict[str, Any]] = {}

    def ev(w: dict[str, Any], assoc: str, thr: float, mk: float, v2s: float, margin: float):
        key = (w["tree"], w["side"], w["k"], assoc, thr, mk, v2s if assoc == "v2" else 0, margin)
        if key not in memo:
            set_v2_scale(v2s)
            cfg = base_config(assoc=assoc, score_threshold=thr, merge_k=mk)
            res = run(w["paths"], PROMPT, cfg, detector=det, reconstructor=rec, export=False)
            box = fc.roi_box(scene.gt_xyz, margin)
            e = fc.evaluate_window(res, w["recon"], w["sim"], scene.gt_ids, scene.gt_xyz, box)
            memo[key] = {"err": abs(e["pred"] - e["gt"]), "f1": e["assoc_f1"],
                         "pred": e["pred"], "gt": e["gt"], "pi": e["per_image"]}  # fmt: skip
        return memo[key]

    def score(assocs: tuple[str, ...], **kw: Any) -> dict[str, float]:
        rows = [ev(w, a, **kw) for w in wins for a in assocs]
        f1 = [r["f1"] for r in rows if r["f1"] == r["f1"]]
        return {"mae": float(np.mean([r["err"] for r in rows])),
                "mape": float(np.mean([r["err"] / max(r["gt"], 1) for r in rows])),
                "f1": float(np.mean(f1)) if f1 else float("nan"),
                "bias": float(np.mean([r["pred"] - r["gt"] for r in rows]))}  # fmt: skip

    def pick(table: list[dict[str, Any]], name: str) -> Any:
        best = min(table, key=lambda r: (round(r["mae"], 3), -r["f1"]))
        return best[name]

    cur = {"thr": 0.3, "margin": 0.7, "mk": 0.5, "v2s": 1.0}
    stages: dict[str, list[dict[str, Any]]] = {}
    both = ("v1", "v2")
    stages["score_threshold"] = [
        {"score_threshold": t, **score(both, **{**cur, "thr": t})} for t in THR
    ]
    cur["thr"] = pick(stages["score_threshold"], "score_threshold")
    stages["roi_margin_m"] = [
        {"roi_margin_m": m, **score(both, **{**cur, "margin": m})} for m in MARGIN
    ]
    cur["margin"] = pick(stages["roi_margin_m"], "roi_margin_m")
    stages["merge_k"] = [
        {"merge_k": m, **score(("v1",), **{**cur, "mk": m})} for m in MERGE_K
    ]
    cur["mk"] = pick(stages["merge_k"], "merge_k")
    stages["v2_gate_scale"] = [
        {"v2_gate_scale": s, **score(("v2",), **{**cur, "v2s": s})} for s in V2_SCALE
    ]
    cur["v2s"] = pick(stages["v2_gate_scale"], "v2_gate_scale")
    set_v2_scale(1.0)
    out["stages"] = stages
    out["chosen"] = cur
    out["wall_s"] = time.time() - t0
    (WORK / "tune.json").write_text(json.dumps(out, indent=1, default=float))
    for name, rows in stages.items():
        print(name)
        for r in rows:
            print("  ", {k: round(v, 3) for k, v in r.items()})
    for r in reg_rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("chosen", cur, f"wall={out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
