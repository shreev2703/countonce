"""E1 tracker side on Fuji-SfM: ByteTrack and SAM 2.1 on the detections CountOnce associates.

    python scripts/run_e1_tracker.py prepare --split val      # cache detections + GT per window
    python scripts/run_e1_tracker.py tune                      # thresholds, training trees only
    python scripts/run_e1_tracker.py prepare --split test
    python scripts/run_e1_tracker.py countonce                 # v1/v2/sync, frozen settings
    python scripts/run_e1_tracker.py run --budget-min 25       # test windows, resumable

Run in ``.venv-gpu`` from the repo root with ``PYTHONPATH=.:scripts``. Rows go to
``results/e1_tracker.csv`` (append-only, resumable); prepared windows to ``runs/fuji_core/e1/``.
Frozen detector settings come from ``configs/fuji_core.yaml`` (score 0.2), as for every Core row.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from bench import fuji_core as fc
from bench import track_e1 as e1
from bench.baselines_track import ByteTrackParams, Sam2Params
from countonce.recon.chunked import ChunkAlignmentError

WORK = Path("runs/fuji_core")
PREP = WORK / "e1"
CSV = Path("results/e1_tracker.csv")
TUNE_JSON = Path("configs/e1_tracker.json")
KS = (3, 5, 10, 20)
TUNE_KS = (5, 10)  # training windows whose reconstructions the Core tuning already cached
SCORE = 0.2


def load_fuji_trees(split: str) -> list[str]:
    return list(json.loads(Path("bench/splits_fuji_sfm.json").read_text())["splits"][split])


def prep_path(split: str, tree: str, side: str, k: int) -> Path:
    return PREP / f"{split}_{tree}_{side}_{k}.json"


def prepare(split: str, ks: tuple[int, ...]) -> None:
    from run_fuji_core import PROMPT, _window, base_config, load_recon

    import countonce.pipeline as pl
    from bench.datasets.fuji_sfm import load_fuji_sfm
    from countonce.pipeline import build_detector, build_reconstructor, run
    from countonce.recon.align import transform_extrinsics

    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    imgs = fc.prepare_images(scene, WORK / "img")
    ov = {s: json.loads((WORK / f"overview_{s}.json").read_text()) for s in ("east", "west")}
    trees = scene.trees[{"test": "test", "val": "training"}[split]]
    box = fc.roi_box(scene.gt_xyz)
    cfg0 = base_config(score_threshold=SCORE)
    det, rec = build_detector(cfg0), build_reconstructor(cfg0)
    PREP.mkdir(parents=True, exist_ok=True)
    for k, tree, side in itertools.product(ks, trees, ("east", "west")):
        out = prep_path(split, tree, side, k)
        if out.exists():
            continue
        _, _, paths, gt_ext = _window(scene, imgs, ov, side, tree, k)
        cfg = base_config(assoc="v1", score_threshold=SCORE)
        seen: list[list[Any]] = []
        orig = pl.lift_detections

        def spy(*a: Any, _orig: Any = orig, _seen: list[list[Any]] = seen, **kw: Any) -> Any:
            r = _orig(*a, **kw)
            _seen.append(list(r))
            return r

        pl.lift_detections = spy
        try:
            res = run(paths, PROMPT, cfg, detector=det, reconstructor=rec, export=False)
        finally:
            pl.lift_detections = orig
        lifted = seen[0]
        if len(lifted) != len(res.detections):
            raise RuntimeError("captured lift does not match the pipeline's detections")
        recon = load_recon(cfg, res.recon_cache_key or "")
        try:
            sim = fc.align_window(gt_ext, recon.extrinsics)
            ev1 = fc.evaluate_window(res, recon, sim, scene.gt_ids, scene.gt_xyz, box, gt_ext)
        except (ChunkAlignmentError, fc.AlignmentError) as e:
            msg = f"{type(e).__name__}: {e}"
            out.write_text(e1.PreparedWindow(tree, side, k, [], error=msg).to_json())
            print(split, tree, side, k, "ALIGNMENT FAILED", msg, flush=True)
            continue
        ext_gt = transform_extrinsics(recon.extrinsics, sim)
        vis = fc.visible_gt(
            scene.gt_xyz, recon.intrinsics, ext_gt, fc.scaled_depth(recon.depth, sim)
        )
        view_of = {r.id: r.view for r in res.detections}
        per: list[list[Any]] = [[] for _ in range(k)]
        for ld in lifted:
            per[view_of[ld.det.id]].append(ld)
        views = []
        for v in range(k):
            roi = [
                bool(ld.xyz is not None and fc.in_roi(sim.apply(np.asarray(ld.xyz)[None]), box)[0])
                for ld in per[v]
            ]
            views.append(
                e1.PreparedView(
                    image=str(paths[v]),
                    boxes=[[float(x) for x in ld.det.box_xyxy] for ld in per[v]],
                    scores=[float(ld.det.score) for ld in per[v]],
                    in_roi=roi,
                    gt_visible=sorted(int(i) for i in vis[v]),
                )
            )
        win = e1.PreparedWindow(tree, side, k, views)
        if len({g for v in views for g in v.gt_visible}) != ev1["gt"]:
            raise RuntimeError("prepared GT differs from the Core evaluation's")
        out.write_text(win.to_json())
        print(split, tree, side, k, "dets", sum(len(v.boxes) for v in views), "roi",
              sum(sum(v.in_roi) for v in views), "gt",
              len({g for v in views for g in v.gt_visible}), flush=True)  # fmt: skip


def countonce_counts() -> None:
    """v1/v2/sync counts of every prepared test window with the frozen Core settings (cached
    reconstruction and detections), scored against the same GT and ROI as the trackers."""
    import yaml
    from run_fuji_core import PROMPT, _window, base_config, load_recon

    from bench.datasets.fuji_sfm import load_fuji_sfm
    from countonce.pipeline import build_detector, build_reconstructor, run

    frozen = {
        a: yaml.safe_load(Path(f"configs/{f}.yaml").read_text())
        for a, f in (("v1", "fuji_core"), ("v2", "fuji_core"), ("sync", "fuji_core_sync"))
    }
    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    imgs = fc.prepare_images(scene, WORK / "img")
    ov = {s: json.loads((WORK / f"overview_{s}.json").read_text()) for s in ("east", "west")}
    box = fc.roi_box(scene.gt_xyz)
    det, rec = build_detector(base_config()), build_reconstructor(base_config())
    for p in sorted(PREP.glob("test_*.json")):
        win = e1.PreparedWindow.from_json(p.read_text())
        if win.error or (win.countonce and "sync" in win.countonce):
            continue
        _, _, paths, gt_ext = _window(scene, imgs, ov, win.side, win.tree, win.k)
        counts: dict[str, float] = {}
        for assoc, fz in frozen.items():
            cfg = base_config(
                assoc=assoc, score_threshold=fz["score_threshold"], merge_k=fz["merge_k"]
            )
            res = run(paths, PROMPT, cfg, detector=det, reconstructor=rec, export=False)
            recon = load_recon(cfg, res.recon_cache_key or "")
            sim = fc.align_window(gt_ext, recon.extrinsics)
            ev = fc.evaluate_window(res, recon, sim, scene.gt_ids, scene.gt_xyz, box, gt_ext)
            if counts.setdefault("gt", ev["gt"]) != ev["gt"]:
                raise RuntimeError(f"{p.name}: GT differs between associators")
            counts[assoc] = ev["pred"]
        win.countonce = counts
        p.write_text(win.to_json())
        print(p.stem, counts, flush=True)


def load_windows(
    split: str, ks: tuple[int, ...], with_errors: bool = False
) -> list[e1.PreparedWindow]:
    out = []
    for p in sorted(PREP.glob(f"{split}_*.json")):
        w = e1.PreparedWindow.from_json(p.read_text())
        if w.k in ks and (with_errors or not w.error):
            out.append(w)
    return out


BYTE_GRID = [
    ByteTrackParams(high_thr=h, match_iou=m, second_iou=0.5)
    for h in (0.2, 0.3, 0.4, 0.5)
    for m in (0.1, 0.2, 0.3, 0.5)
]
SAM_GRID = [Sam2Params(cover_iou=c) for c in (0.2, 0.3, 0.5)]


def cell_error(win_runs: list[tuple[e1.PreparedWindow, dict[str, int]]]) -> float:
    return float(np.mean([abs(r["pred"] - r["gt"]) for _, r in win_runs]))


def tune(args: argparse.Namespace) -> None:
    """Pick each tracker's thresholds by mean |error| on the training trees (Tree09-11) only,
    over windows k=5,10 at strides 1 and 2, in capture order. Test trees are never read."""
    wins = load_windows("val", TUNE_KS)
    if not wins:
        raise SystemExit("no prepared val windows; run `prepare --split val` first")
    cells = [(w, s) for w in wins for s in (1, 2)]
    log: dict[str, Any] = {"n_windows": len(wins), "bytetrack": [], "sam2": []}
    for p in BYTE_GRID:
        runs = [
            (w, e1.run_bytetrack(w, e1.select_views(w.k, s, "capture", 0), p)) for w, s in cells
        ]
        log["bytetrack"].append({"params": p.__dict__, "mae": cell_error(runs)})
        print("bytetrack", p, round(cell_error(runs), 2), flush=True)
    best_b = min(log["bytetrack"], key=lambda r: r["mae"])
    if not args.no_sam2:
        from bench.baselines_track import Sam2Tracker

        t0 = time.time()
        for sp in SAM_GRID:
            tr = Sam2Tracker(sp)
            runs = []
            for w, s in [(w, s) for w, s in cells if w.k == 5]:
                runs.append((w, e1.run_sam2(w, e1.select_views(w.k, s, "capture", 0), tr)))
                if (time.time() - t0) / 60 > args.budget_min:
                    break
            log["sam2"].append({"params": sp.__dict__, "mae": cell_error(runs), "n": len(runs)})
            print("sam2", sp, round(cell_error(runs), 2), len(runs), flush=True)
            del tr
        best_s = min(log["sam2"], key=lambda r: r["mae"])
    else:
        best_s = {"params": Sam2Params().__dict__, "mae": float("nan")}
    TUNE_JSON.write_text(
        json.dumps(
            {
                "chosen_on": "training trees only (bench/splits_fuji_sfm.json 'training')",
                "bytetrack": best_b["params"],
                "sam2": best_s["params"],
                "grid": log,
            },
            indent=1,
        )  # fmt: skip
    )
    print("chosen", best_b["params"], best_s["params"])


def run_test(args: argparse.Namespace) -> None:
    frozen = json.loads(TUNE_JSON.read_text())
    bp = ByteTrackParams(**frozen["bytetrack"])
    sp = Sam2Params(**frozen["sam2"])
    wins = load_windows("test", KS, with_errors=True)
    n_expected = len(load_fuji_trees("test")) * 2 * len(KS)
    if len(wins) != n_expected:
        raise SystemExit(f"{len(wins)} of {n_expected} test windows prepared; run `prepare` first")
    done = e1.read_done(CSV)
    sam = None
    t_start = time.time()
    # capture order first so a budget stop leaves the primary comparison complete
    for order, k, stride in itertools.product(e1.ORDERS, KS, e1.STRIDES):
        for win in [x for x in wins if x.k == k]:
            for tracker in ("bytetrack", "sam2"):
                if tracker not in args.trackers:
                    continue
                params = bp if tracker == "bytetrack" else sp
                row: dict[str, Any] = {
                    "split": "test", "tree": win.tree, "side": win.side, "k": k,
                    "stride": stride, "order": order, "tracker": tracker,
                    "params": json.dumps(params.__dict__, sort_keys=True),
                }  # fmt: skip
                if e1.row_key(row) in done:
                    continue
                views = e1.select_views(
                    k, stride, order, e1.window_seed(win.tree, win.side, k, stride)
                )
                row["n_views"] = len(views)
                if win.error:  # no GT frame, so nothing to score against
                    e1.append_row(CSV, {**row, "status": "failed", "error": win.error[:300]})
                    continue
                if len(views) < 2:
                    e1.append_row(CSV, {**row, "status": "skipped",
                                        "error": "fewer than 2 views at this stride"})  # fmt: skip
                    continue
                if (time.time() - t_start) / 60 > args.budget_min:
                    print("budget hit", flush=True)
                    return
                t0 = time.time()
                try:
                    if tracker == "bytetrack":
                        res = e1.run_bytetrack(win, views, bp)
                        vram = ""
                    else:
                        import torch

                        from bench.baselines_track import Sam2Tracker

                        if sam is None:
                            sam = Sam2Tracker(sp)
                        torch.cuda.reset_peak_memory_stats()
                        res = e1.run_sam2(win, views, sam)
                        vram = torch.cuda.max_memory_allocated() / 2**20
                    done_row = {**row, **res, "status": "ok", "peak_vram_mb": vram}
                    e1.append_row(CSV, {**done_row, "runtime_s": time.time() - t0})
                    print(win.tree, win.side, k, stride, order, tracker, res, flush=True)
                except Exception as e:  # noqa: BLE001 - a failed run is a row (OOM included)
                    traceback.print_exc()
                    msg = f"{type(e).__name__}: {e}"[:300]
                    status = "failed" if "out of memory" in msg.lower() else "error"
                    e1.append_row(CSV, {**row, "status": status, "error": msg,
                                        "runtime_s": time.time() - t0})  # fmt: skip
                    if "sam" in tracker:
                        sam = None
                        import torch

                        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--split", choices=["val", "test"], required=True)
    t = sub.add_parser("tune")
    t.add_argument("--no-sam2", action="store_true")
    t.add_argument("--budget-min", type=float, default=15.0)
    sub.add_parser("countonce")
    r = sub.add_parser("run")
    r.add_argument("--budget-min", type=float, default=25.0)
    r.add_argument("--trackers", nargs="+", default=["bytetrack", "sam2"])
    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.split, TUNE_KS if args.split == "val" else KS)
    elif args.cmd == "tune":
        tune(args)
    elif args.cmd == "countonce":
        countonce_counts()
    else:
        run_test(args)


if __name__ == "__main__":
    main()
