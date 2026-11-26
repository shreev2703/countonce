"""Photo-order invariance on Fuji validation windows at a gate where v2 and sync merge.

Follow-up to ``results/order_invariance_fuji.csv`` (no merges at the frozen gate). Validation trees
only (Tree09-11), cached detections and reconstructions, CPU. Sweeps the v2/sync reprojection gate
(``max_reproj_norm``; frozen value 0.5), picks the smallest gate where at least half the k=10
windows contain a merge for both, then measures invariance at k=5 and k=10 there.
Diagnostic only: configs/fuji_core*.yaml are not edited.

    PYTHONPATH=scripts:. python scripts/run_order_invariance_fuji_merging.py --n-perms 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from run_fuji_core import PROMPT, WORK, _window, base_config

import countonce.pipeline as pl
import countonce.recon.vggt as vggt_mod
from bench import fuji_core as fc
from bench.datasets.fuji_sfm import load_fuji_sfm
from bench.order_invariance import (
    FUJI_CSV_FIELDS,
    fuji_window_row,
    gate_sweep,
    n_merges,
    smallest_merging_gate,
)
from countonce.merge.match import V2Config
from countonce.pipeline import build_detector, build_reconstructor, run
from countonce.recon.chunked import align_chunk


class _Capture:
    """Associator stand-in that records its inputs (lifted, recon, frames) and delegates."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.name, self.calls = inner, inner.name, []

    def associate(self, lifted: Any, recon: Any = None, frames: Any = None) -> Any:
        self.calls.append((list(lifted), recon, frames))
        return self.inner.associate(lifted, recon, frames)


FROZEN_GATE = V2Config().max_reproj_norm
GATES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def make_at_gate(name: str, gate: float, merge_k: float) -> Any:
    cfg = base_config(assoc=name, merge_k=merge_k)
    return pl._make_base_associator(cfg, merge_k * gate / FROZEN_GATE)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perms", type=int, default=10)
    ap.add_argument("--score-threshold", type=float, default=0.2)
    ap.add_argument("--merge-k", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("results/order_invariance_fuji_merging.csv"))
    ap.add_argument(
        "--sweep-out", type=Path, default=Path("results/order_invariance_fuji_gate_sweep.csv")
    )
    args = ap.parse_args(argv)

    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    imgs = fc.prepare_images(scene, WORK / "img")
    ov = {s: json.loads((WORK / f"overview_{s}.json").read_text()) for s in ("east", "west")}
    trees = scene.trees["training"]  # 'val' split only; test trees never touched
    box = fc.roi_box(scene.gt_xyz)
    kw = dict(score_threshold=args.score_threshold, merge_k=args.merge_k)
    if not torch.cuda.is_available():
        # The recon cache key records the autocast dtype of the GPU run (RTX 5070: bf16); keep it
        # so keys match. Models load lazily, so a cache miss fails instead of running on CPU.
        vggt_mod.pick_autocast_dtype = lambda: torch.bfloat16
    base = base_config(**kw)
    det, rec = build_detector(base), build_reconstructor(base)

    wins: dict[int, list[tuple[Any, Any, Any, str, str, Any]]] = {5: [], 10: []}
    for k in wins:
        for tree in trees:
            for side in ("east", "west"):
                _, _, paths, gt_ext = _window(scene, imgs, ov, side, tree, k)
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
                    run(paths, PROMPT, base_config(assoc="v1", **kw), detector=det,
                        reconstructor=rec, export=False)  # fmt: skip
                finally:
                    pl._make_associator = orig
                assert cap is not None
                lifted, recon, frames = cap.calls[0]
                wins[k].append((recon, lifted, gate_ext(gt_ext, recon), tree, side, frames))

    sweep_rows: list[dict[str, Any]] = []
    sweeps = []
    for name in ("v2", "sync"):
        for k in (5, 10):
            sw = gate_sweep(
                lambda g, n=name: make_at_gate(n, g, args.merge_k),
                [(r, lf) for r, lf, _, _, _, _ in wins[k]],
                GATES,
                [w[5] for w in wins[k]],
            )
            for r in sw:
                sweep_rows.append({"assoc": name, "k": k, **r})
                print(sweep_rows[-1], flush=True)
            if k == 10:
                sweeps.append(sw)
    gate = smallest_merging_gate(sweeps, 0.5)
    print("chosen gate:", gate, flush=True)
    args.sweep_out.parent.mkdir(parents=True, exist_ok=True)
    with args.sweep_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
        w.writeheader()
        w.writerows(sweep_rows)
    if gate is None:
        raise SystemExit("no gate in the sweep gives merges in half the k=10 windows")

    rows: list[dict[str, Any]] = []
    for k in (5, 10):
        for recon, lifted, sim, tree, side, frames in wins[k]:
            for name in ("v2", "sync"):
                row = fuji_window_row(
                    name, make_at_gate(name, gate, args.merge_k), lifted, recon,
                    tree=tree, side=side, k=k, n_perms=args.n_perms, seed=k * 1000 + len(rows),
                    to_gt=sim.apply, in_roi=lambda x: fc.in_roi(x, box), frames=frames,
                )  # fmt: skip
                row["gate"] = gate
                row["n_merges"] = n_merges(row["n_dets"], row["base_count"])
                rows.append(row)
                print(row, flush=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*FUJI_CSV_FIELDS, "gate", "n_merges"])
        w.writeheader()
        w.writerows(rows)


def gate_ext(gt_ext: Any, recon: Any) -> Any:
    return align_chunk(gt_ext, recon.extrinsics)


if __name__ == "__main__":
    main()
