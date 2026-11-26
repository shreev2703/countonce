"""Fuji-SfM Core benchmark driver (see ``bench/fuji_core.py`` for the protocol).

    python scripts/run_fuji_core.py overview            # per-side registration + pseudo-poses
    python scripts/run_fuji_core.py bench --split val   # thresholds are chosen on val trees
    python scripts/run_fuji_core.py bench --split test --budget-min 90

    python scripts/run_fuji_core.py bench --split test --backbone colmap --budget-min 600

``--backbone colmap`` (frozen v2 only) runs COLMAP on the same windows, cached detections and GT.
The window's GT and frame come from the VGGT reference reconstruction; COLMAP's cameras are placed
in that frame (``fuji_core.align_to_reference``). Failed registrations are recorded as
``status=failed`` rows.

Outputs go to ``runs/fuji_core/`` (git-ignored) and ``results/bench.csv`` (append-only).
Run from the repo root inside ``.venv-gpu`` with ``PYTHONPATH=.``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from bench import fuji_core as fc
from bench.datasets.fuji_sfm import FujiScene, load_fuji_sfm
from bench.runner import DEFAULT_CSV, Job, RunOutput, SkipRun, read_rows, run_jobs
from countonce.config import Config
from countonce.pipeline import build_detector, build_reconstructor, run
from countonce.recon.adapter import ReconAdaptError
from countonce.recon.align import camera_centers, transform_extrinsics
from countonce.recon.cache import ReconCache
from countonce.recon.chunked import ChunkAlignmentError

WORK = Path("runs/fuji_core")
PROMPT = "apple"
STRIDE = 3


def base_config(**kw: Any) -> Config:
    d: dict[str, Any] = dict(
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


def load_recon(cfg: Config, key: str) -> Any:
    assert cfg.cache_dir is not None
    rc = ReconCache(Path(cfg.cache_dir) / "recon")
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
        ext = transform_extrinsics(recon.extrinsics, reg.sim)
        cen = camera_centers(ext)
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


def _window(
    scene: FujiScene,
    imgs: list[Path],
    ov: dict[str, Any],
    side: str,
    tree: str,
    k: int,
) -> tuple[int, int, list[Path], np.ndarray]:
    ids = scene.side_indices[side]
    ext = np.array(ov[side]["ext_full"])
    cam_y = camera_centers(ext)[:, 1]
    tree_y = float(scene.gt_xyz[scene.tree_mask(tree)][:, 1].mean())
    start, stop = fc.window_for_tree(cam_y, len(ids), tree_y, k)
    return start, stop, [imgs[ids[i]] for i in range(start, stop)], ext[start:stop]


FROZEN_CONFIG = Path("configs/fuji_core.yaml")


def load_frozen(path: Path, split: str) -> tuple[dict[str, Any], str]:
    """Frozen thresholds (chosen on the training trees) and their hash. Test needs the file."""
    if not path.exists():
        if split == "test":
            raise SystemExit(f"{path} is missing: choose thresholds on the training split first")
        return {}, "none"
    import yaml

    return (yaml.safe_load(path.read_text()) or {}), fc.config_hash(path)


def _mk(args: argparse.Namespace) -> dict[str, Any]:
    return {} if args.merge_k is None else {"merge_k": args.merge_k}


def _v3_meta(args: argparse.Namespace) -> dict[str, Any]:
    if "v3" not in args.assocs:
        return {}
    ck = Path(args.v3_checkpoint)
    if not ck.is_file():
        raise SystemExit(f"v3 checkpoint {ck} is missing")
    return {"v3_checkpoint": str(ck), "v3_checkpoint_sha256": fc.file_hash(ck)}


def reference_gt(csv_path: Path, split: str, tree: str, side: str, k: int) -> float | None:
    """GT of the finished VGGT v2 row of a window, if the CSV has one."""
    for r in reversed(read_rows(csv_path)):
        if (r["status"], r["split"], r["scene_id"], r["sample"], r["n_views"], r["method"]) == (
            "ok", split, tree, side, str(k), "countonce_v2"
        ) and fc._row_protocol(r) == fc.PROTOCOL:  # fmt: skip
            return float(r["gt"])
    return None


def cmd_bench(args: argparse.Namespace) -> None:
    colmap = args.backbone == "colmap"
    if colmap and list(args.assocs) == ["v1", "v2"]:  # the parser default
        args.assocs = ["v2"]
    if colmap and list(args.assocs) != ["v2"]:
        raise SystemExit("--backbone colmap runs the frozen v2 settings only")
    frozen, cfg_hash = load_frozen(args.config, args.split)
    if "score_threshold" in frozen:
        args.score_threshold = float(frozen["score_threshold"])
    if "merge_k" in frozen:
        args.merge_k = float(frozen["merge_k"])
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
    base = base_config(score_threshold=args.score_threshold, **_mk(args))
    det, rec = build_detector(base), build_reconstructor(base)  # rec: VGGT (reference)
    col_cfg = {"backbone": "colmap", "chunked_recon": False}
    col = build_reconstructor(base_config(**col_cfg)) if colmap else None
    bench_rec = col if col is not None else rec
    model_cfg = {
        "detector": f"{det.name}@{det.version}",
        "backbone": bench_rec.config() if hasattr(bench_rec, "config") else str(bench_rec),
        "prompt": PROMPT, "score_threshold": args.score_threshold, "merge_k": base.merge_k,
        "downscale": fc.DOWNSCALE, "stride": STRIDE, "roi_margin_m": fc.ROI_MARGIN_M,
        "gt_tol_occ_m": fc.GT_TOL_OCC_M, "gt_max_depth_m": fc.GT_MAX_DEPTH_M,
        "match_px": fc.MATCH_PX, "chunk_size": base.chunk_size, "protocol": fc.PROTOCOL,
        "assocs": list(args.assocs), **_v3_meta(args),
        "frozen_config": str(args.config), "frozen_config_hash": cfg_hash,
        "frozen_values": frozen, **fc.host_info(),
    }  # fmt: skip
    if colmap:
        model_cfg["gt_reference"] = "vggt_window_visibility"
        model_cfg["alignment"] = "colmap_cameras_to_vggt_aligned_window"
    t_start = time.time()
    memo: dict[tuple[str, str, int], dict[str, Any]] = {}

    def group(tree: str, side: str, k: int) -> dict[str, Any]:
        key = (tree, side, k)
        if key in memo:
            return memo[key]
        start, stop, paths, gt_ext = _window(scene, imgs, ov, side, tree, k)
        runs: dict[str, Any] = {}
        # v1 always runs: it carries the per-image baselines and the shared ingest/detect timings
        for assoc in dict.fromkeys(("v1", *args.assocs)):
            extra = {"v3_checkpoint": args.v3_checkpoint} if assoc == "v3" else {}
            cfg = base_config(
                assoc=assoc, score_threshold=args.score_threshold, **_mk(args), **extra
            )
            res = run(paths, PROMPT, cfg, detector=det, reconstructor=rec, export=False)
            recon = load_recon(cfg, res.recon_cache_key or "")
            if recon.n_views != k:
                raise RuntimeError(f"recon has {recon.n_views} views, expected {k}")
            sim = fc.align_window(gt_ext, recon.extrinsics)
            ev = fc.evaluate_window(res, recon, sim, scene.gt_ids, scene.gt_xyz, box, gt_ext)
            runs[assoc] = (ev, res)
        c_gt = camera_centers(gt_ext)
        c_al = camera_centers(transform_extrinsics(recon.extrinsics, sim))
        runs["align"] = {
            "scale": sim.s, "cam_rms_m": float(np.sqrt(((c_gt - c_al) ** 2).sum(1).mean())),
            "cam_span_m": float(np.linalg.norm(np.ptp(c_gt, axis=0))), "frames": [start, stop],
        }  # fmt: skip
        memo.clear()
        memo[key] = runs
        return runs

    def colmap_group(tree: str, side: str, k: int) -> dict[str, Any]:
        _, _, paths, gt_ext = _window(scene, imgs, ov, side, tree, k)
        kw: dict[str, Any] = {"score_threshold": args.score_threshold, **_mk(args)}
        # the window's GT is the VGGT reference's (cached recon and detections)
        cfg = base_config(assoc="v2", **kw)
        ref = run(paths, PROMPT, cfg, detector=det, reconstructor=rec, export=False)
        ref_rec = load_recon(cfg, ref.recon_cache_key or "")
        ref_sim = fc.align_window(gt_ext, ref_rec.extrinsics)
        ref_ext = transform_extrinsics(ref_rec.extrinsics, ref_sim)
        visible = fc.visible_gt(
            scene.gt_xyz,
            ref_rec.intrinsics,
            ref_ext,
            fc.scaled_depth(ref_rec.depth, ref_sim),
        )
        n_gt = len({int(i) for v in visible for i in v})
        want = reference_gt(args.csv, args.split, tree, side, k)
        if want is not None and want != n_gt:
            raise RuntimeError(f"reference GT {n_gt} differs from the VGGT row's {want:g}")
        cfg = base_config(assoc="v2", **col_cfg, **kw)
        try:
            res = run(paths, PROMPT, cfg, detector=det, reconstructor=col, export=False)
            recon = load_recon(cfg, res.recon_cache_key or "")
            sim, rot_resid = fc.align_to_reference(ref_ext, recon.extrinsics)
            # same tolerance as a fraction of the image width as on the VGGT grid
            match_px = fc.MATCH_PX * recon.depth.shape[2] / ref_rec.depth.shape[2]
            ev = fc.evaluate_window(res, recon, sim, scene.gt_ids, scene.gt_xyz, box, ref_ext,
                                    visible=visible, match_px=match_px)  # fmt: skip
        except (ReconAdaptError, ChunkAlignmentError, fc.AlignmentError) as e:
            print(tree, side, k, "FAILED", e, flush=True)
            raise SkipRun("failed", f"{type(e).__name__}: {e}") from e
        n_reg = int(fc.registered_views(recon.extrinsics).sum())
        return {"ev": ev, "res": res, "n_registered": n_reg, "rot_resid_deg": rot_resid}

    def jobs() -> Iterator[Job]:
        for k in args.ks:
            for tree in trees:
                for side in args.sides:
                    if (time.time() - t_start) / 60 > args.budget_min:
                        print(f"budget hit before {tree}/{side}/k={k}", flush=True)
                        return
                    for method in fc.methods_for(args.assocs, not args.no_baselines, args.backbone):
                        cfg = {**model_cfg, "tree": tree, "side": side}
                        yield fc.make_job(tree, args.split, side, k, method, cfg)

    def fn(job: Job) -> RunOutput:
        assert job.n_views is not None
        if colmap:
            got = colmap_group(job.scene_id, job.sample, job.n_views)
            res = got["res"]
            out = fc.run_output(got["ev"], job.method, res, res.timings["total"])
            assert out.timings is not None
            out.timings["n_registered"] = float(got["n_registered"])
            out.timings["rot_resid_deg"] = got["rot_resid_deg"]
            print(job.scene_id, job.sample, job.n_views, job.method, "pred", out.pred, "gt",
                  out.gt, "registered", got["n_registered"], flush=True)  # fmt: skip
            return out
        runs = group(job.scene_id, job.sample, job.n_views)
        v1_ev, v1_res = runs["v1"]
        if job.method.startswith("countonce"):
            assoc = fc.METHOD_ASSOC[job.method]
            ev, res = runs[assoc]
            t = dict(res.timings)
            if assoc != "v1":  # recon and detection are shared; add this association's own stages
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

    print(
        run_jobs(fc.pending_jobs(jobs(), args.csv), fn, args.csv, model_cfg),
        f"wall_min={(time.time() - t_start) / 60:.1f}",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overview")
    o.add_argument("--sides", nargs="+", default=["east", "west"])
    o.add_argument("--force", action="store_true")
    o.set_defaults(fn=cmd_overview)
    b = sub.add_parser("bench")
    b.add_argument("--split", choices=["val", "test"], default="val")
    b.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10, 20])
    b.add_argument("--sides", nargs="+", default=["east", "west"])
    b.add_argument("--backbone", choices=fc.BACKBONES, default="vggt")
    b.add_argument("--score-threshold", type=float, default=0.3)
    b.add_argument("--merge-k", type=float, default=None, help="v1 association distance (x size)")
    b.add_argument(
        "--assocs",
        nargs="+",
        choices=sorted(fc.ASSOC_METHODS),
        default=["v1", "v2"],
        help="pipeline associations (identical cached upstream inputs); "
        "--backbone colmap runs v2 only",
    )
    b.add_argument(
        "--v3-checkpoint",
        type=Path,
        default=Path("models/checkpoints/v001.pt"),
        help="registered matcher checkpoint, used as-is (no Fuji fine-tune)",
    )
    b.add_argument(
        "--no-baselines",
        action="store_true",
        help="skip the per-image baseline rows (already in the CSV)",
    )
    b.add_argument("--budget-min", type=float, default=90.0)
    b.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    b.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    b.set_defaults(fn=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
