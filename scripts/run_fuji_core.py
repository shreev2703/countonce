"""Fuji-SfM Core benchmark driver (see ``bench/fuji_core.py`` for the protocol).

    python scripts/run_fuji_core.py overview            # per-side registration + pseudo-poses
    python scripts/run_fuji_core.py bench --split val   # thresholds are chosen on val trees
    python scripts/run_fuji_core.py bench --split test --budget-min 90

Outputs go to ``runs/fuji_core/`` (git-ignored) and ``results/bench.csv`` (append-only).
Run from the repo root inside ``.venv-gpu`` with ``PYTHONPATH=.``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bench import fuji_core as fc
from bench.datasets.fuji_sfm import load_fuji_sfm
from bench.runner import DEFAULT_CSV, run_jobs
from countonce.config import Config
from countonce.pipeline import build_detector, build_reconstructor, run
from countonce.recon.cache import ReconCache
from countonce.recon.chunked import align_chunk

WORK = Path("runs/fuji_core")
PROMPT = "apple"
STRIDE = 3


def base_config(**kw) -> Config:  # type: ignore[no-untyped-def]
    d = dict(
        detector="gdino",
        backbone="vggt",
        assoc="v1",
        max_frames=0,
        dedup_threshold=0,
        out_dir=WORK / "out",
        cache_dir=WORK / "cache",
        chunked_recon=True,
        chunk_size=20,
        chunk_overlap=6,
    )
    d.update(kw)
    return Config(**d)


def load_recon(cfg: Config, key: str):  # type: ignore[no-untyped-def]
    rc = ReconCache(Path(cfg.cache_dir) / "recon")  # type: ignore[arg-type]
    rec = rc.get(key)
    if rec is None:
        raise RuntimeError(f"recon {key} missing from cache")
    return rec


def cmd_overview(args: argparse.Namespace) -> None:
    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    imgs = fc.prepare_images(scene, WORK / "img")
    det, rec = build_detector(base_config()), build_reconstructor(base_config())
    for side in args.sides:
        out = WORK / f"overview_{side}.json"
        if out.exists() and not args.force:
            print("have", out)
            continue
        ids = scene.side_indices[side]
        local = np.arange(0, len(ids), STRIDE)
        cfg = base_config(chunk_size=16, chunk_overlap=6)
        t0 = time.time()
        res = run([imgs[ids[i]] for i in local], PROMPT, cfg, detector=det, reconstructor=rec,
                  export=False)  # fmt: skip
        recon = load_recon(cfg, res.recon_cache_key or "")
        src = np.stack([i.xyz for i in res.instances if i.n_views >= 2])
        reg = fc.register_to_gt(src, scene.gt_xyz)
        ext = fc.transform_extrinsics(recon.extrinsics, reg.sim)
        cen = fc.camera_centers(ext)
        all_idx = np.arange(len(ids))
        full = fc.interpolate_poses(local, ext, all_idx)
        info = {
            "side": side, "stride": STRIDE, "n_overview_frames": len(local),
            "n_instances": len(res.instances), "n_registered_src": len(src),
            "inlier_frac": reg.inlier_frac, "median_err_m": reg.median_err_m,
            "scale": reg.sim.s, "candidate": reg.candidate, "wall_s": time.time() - t0,
            "peak_vram_mb": res.peak_vram_mb, "oom_rung": res.oom_fallback_rung,
            "cam_centres_gt_first_last": [cen[0].tolist(), cen[-1].tolist()],
            "cam_y_monotonic_frac": float(
                (np.diff(cen[:, 1]) * np.sign(cen[-1, 1] - cen[0, 1]) > 0).mean()
            ),
            "ext_full": full.tolist(),
        }  # fmt: skip
        out.write_text(json.dumps(info))
        print(side, {k: v for k, v in info.items() if k != "ext_full"})


def _window(scene, imgs, ov, side, tree, k):  # type: ignore[no-untyped-def]
    ids = scene.side_indices[side]
    ext = np.array(ov[side]["ext_full"])
    cam_y = fc.camera_centers(ext)[:, 1]
    tree_y = float(scene.gt_xyz[scene.tree_mask(tree)][:, 1].mean())
    start, stop = fc.window_for_tree(cam_y, len(ids), tree_y, k)
    return start, stop, [imgs[ids[i]] for i in range(start, stop)], ext[start:stop]


def cmd_bench(args: argparse.Namespace) -> None:
    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    imgs = fc.prepare_images(scene, WORK / "img")
    ov = {}
    for side in ("east", "west"):
        f = WORK / f"overview_{side}.json"
        if not f.exists():
            raise SystemExit(f"missing {f}; run the overview stage first")
        ov[side] = json.loads(f.read_text())
    split_key = {"test": "test", "val": "training"}[args.split]
    trees = scene.trees[split_key]
    box = fc.roi_box(scene.gt_xyz)
    base = base_config(score_threshold=args.score_threshold)
    det, rec = build_detector(base), build_reconstructor(base)
    model_cfg = {
        "detector": f"{det.name}@{det.version}",
        "backbone": rec.config() if hasattr(rec, "config") else str(rec),
        "prompt": PROMPT, "score_threshold": args.score_threshold, "merge_k": base.merge_k,
        "downscale": fc.DOWNSCALE, "stride": STRIDE, "roi_margin_m": fc.ROI_MARGIN_M,
        "gt_tol_occ_m": fc.GT_TOL_OCC_M, "gt_max_depth_m": fc.GT_MAX_DEPTH_M,
        "match_px": fc.MATCH_PX, "chunk_size": base.chunk_size, "protocol": "fuji_core_v1",
        **fc.host_info(),
    }  # fmt: skip
    t_start = time.time()
    memo: dict[tuple[str, str, int], dict] = {}

    def group(tree: str, side: str, k: int) -> dict:
        key = (tree, side, k)
        if key in memo:
            return memo[key]
        start, stop, paths, gt_ext = _window(scene, imgs, ov, side, tree, k)
        runs = {}
        for assoc in ("v1", "v2"):
            cfg = base_config(assoc=assoc, score_threshold=args.score_threshold)
            res = run(paths, PROMPT, cfg, detector=det, reconstructor=rec, export=False)
            recon = load_recon(cfg, res.recon_cache_key or "")
            if recon.n_views != k:
                raise RuntimeError(f"recon has {recon.n_views} views, expected {k}")
            sim = align_chunk(gt_ext, recon.extrinsics)
            ev = fc.evaluate_window(res, recon, sim, scene.gt_ids, scene.gt_xyz, box)
            runs[assoc] = (ev, res)
        c_gt = fc.camera_centers(gt_ext)
        c_al = fc.camera_centers(fc.transform_extrinsics(recon.extrinsics, sim))
        runs["align"] = {
            "scale": sim.s, "cam_rms_m": float(np.sqrt(((c_gt - c_al) ** 2).sum(1).mean())),
            "cam_span_m": float(np.linalg.norm(np.ptp(c_gt, axis=0))), "frames": [start, stop],
        }  # fmt: skip
        memo.clear()
        memo[key] = runs
        return runs

    def jobs():  # type: ignore[no-untyped-def]
        for k in args.ks:
            for tree in trees:
                for side in args.sides:
                    if (time.time() - t_start) / 60 > args.budget_min:
                        print(f"budget hit before {tree}/{side}/k={k}", flush=True)
                        return
                    for method in ("countonce_v1", "countonce_v2", "sum", "max", "median"):
                        cfg = {**model_cfg, "tree": tree, "side": side}
                        yield fc.make_job(tree, args.split, side, k, method, cfg)

    def fn(job):  # type: ignore[no-untyped-def]
        runs = group(job.scene_id, job.sample, job.n_views)
        v1_ev, v1_res = runs["v1"]
        if job.method.startswith("countonce"):
            ev, res = runs["v2" if job.method.endswith("v2") else "v1"]
            t = dict(res.timings)
            if job.method.endswith("v2"):  # recon and detection are shared; add v2's own stages
                t = {**v1_res.timings, "associate": t["associate"], "confidence": t["confidence"]}
                t["total"] = sum(
                    t[x] for x in ("ingest", "detect", "recon", "associate", "confidence")
                )
            rt = t["total"]
            out = fc.run_output(ev, job.method, res, rt)
        else:
            rt = v1_res.timings["ingest"] + v1_res.timings["detect"]
            out = fc.run_output(v1_ev, job.method, v1_res, rt)
        extra = {k: v for k, v in runs["align"].items()}
        print(job.scene_id, job.sample, job.n_views, job.method, "pred", out.pred, "gt", out.gt,
              "f1", out.assoc_f1, "align", extra, flush=True)  # fmt: skip
        return out

    print(run_jobs(jobs(), fn, args.csv, model_cfg), f"wall_min={(time.time() - t_start) / 60:.1f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overview")
    o.add_argument("--sides", nargs="+", default=["east", "west"])
    o.add_argument("--force", action="store_true")
    o.set_defaults(fn=cmd_overview)
    b = sub.add_parser("bench")
    b.add_argument("--split", choices=["val", "test"], default="val")
    b.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10, 20])
    b.add_argument("--sides", nargs="+", default=["east", "west"])
    b.add_argument("--score-threshold", type=float, default=0.3)
    b.add_argument("--budget-min", type=float, default=90.0)
    b.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    b.set_defaults(fn=cmd_bench)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
