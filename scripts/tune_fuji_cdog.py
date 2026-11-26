"""Choose the C-DOG thresholds on the Fuji training-split trees only (Tree09-11, ``val``).

Same protocol as configs/fuji_core.yaml and configs/fuji_core_sync.yaml: windows of 5 and 10 views
on the 3 training trees x east/west (12 rows), GroundingDINO-tiny "apple" at the frozen
score_threshold 0.2 and VGGT-1B cameras, protocol ``fuji_core_v2``; the winner has the lowest mean
absolute count error with all 12 rows complete. Grid: epipolar gate ``cdog_tau_px`` x overlap
threshold ``cdog_delta``. One ``run_fuji_core.py bench --split val`` call per cell, each into
its own CSV (finished slices are skipped on a rerun). No test-split tree is read.

    PYTHONPATH=. python scripts/tune_fuji_cdog.py --out-dir ~/countonce/results [--write-config F]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

TAUS = [2.0, 4.0, 8.0, 16.0, 32.0]
DELTAS = [0.3, 0.5, 0.55, 0.7]
SCORE_THRESHOLD = 0.2
MERGE_K = 0.3
N_ROWS = 12


def cell_csv(out_dir: Path, tau: float, delta: float) -> Path:
    return out_dir / f"fuji_tune_cdog_t{tau:g}_d{delta:g}.csv"


def summarize(path: Path) -> dict[str, float] | None:
    """MAE / signed bias of a cell's 12 complete ``countonce_cdog`` rows, else None."""
    if not path.exists():
        return None
    with path.open() as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["method"] == "countonce_cdog" and r["split"] == "val" and r["status"] == "ok"
        ]
    seen = {(r["scene_id"], r["sample"], r["n_views"]): r for r in rows}
    if len(seen) < N_ROWS:
        return None
    err = [float(r["pred"]) - float(r["gt"]) for r in seen.values()]
    return {
        "mae": sum(abs(e) for e in err) / len(err),
        "bias": sum(err) / len(err),
        "n": float(len(err)),
    }


def pick(table: list[dict[str, float]]) -> dict[str, float]:
    """Lowest MAE; ties go to the smaller gate, then the delta nearest the paper's 0.5-0.55."""
    return min(
        table,
        key=lambda r: (round(r["mae"], 3), r["tau"], abs(r["delta"] - 0.525)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--write-config", type=Path, default=None)
    ap.add_argument("--budget-min", type=float, default=60.0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table: list[dict[str, float]] = []
    for tau in TAUS:
        for delta in DELTAS:
            csv_path = cell_csv(args.out_dir, tau, delta)
            s = summarize(csv_path)
            if s is None:
                cmd = [
                    sys.executable, "-u", "scripts/run_fuji_core.py", "bench", "--split", "val",
                    "--ks", "5", "10", "--assocs", "cdog", "--no-baselines",
                    "--score-threshold", str(SCORE_THRESHOLD), "--merge-k", str(MERGE_K),
                    "--cdog-tau-px", str(tau), "--cdog-delta", str(delta),
                    "--config", "none", "--csv", str(csv_path),
                    "--budget-min", str(args.budget_min),
                ]  # fmt: skip
                subprocess.run(cmd, check=True)
                s = summarize(csv_path)
            if s is None:
                raise SystemExit(f"cell tau={tau} delta={delta} has fewer than {N_ROWS} ok rows")
            table.append({"tau": tau, "delta": delta, **s})
            print(
                f"tau={tau:g} delta={delta:g} mae={s['mae']:.2f} bias={s['bias']:+.2f}", flush=True
            )
    best = pick(table)
    print("chosen", best)
    (args.out_dir / "fuji_tune_cdog_summary.json").write_text(json.dumps(table, indent=1))
    if args.write_config:
        lines = [
            "# Frozen Fuji-SfM Core settings for the C-DOG association (assoc: cdog), chosen on",
            "# the TRAINING-split trees only (Tree09, Tree10, Tree11; `val` in",
            "# bench/splits_fuji_sfm.json),",
            '# windows of 5 and 10 views (12 rows), GroundingDINO-tiny "apple" + VGGT-1B, protocol',
            "# fuji_core_v2. Same procedure as configs/fuji_core.yaml: lowest mean absolute count",
            "# error",
            "# with all 12 window rows complete. score_threshold and merge_k are the frozen v1/v2",
            "# values",
            "# (identical detections); only the C-DOG settings were swept.",
            f"# Grid: cdog_tau_px {TAUS} x cdog_delta {DELTAS}. cdog_iqr_alpha=2 (paper) and",
            "# bpe_floor_px=0.5 (DEVIATIONS) are fixed.",
            "# MAE by (tau, delta): "
            + "; ".join(f"({r['tau']:g}, {r['delta']:g}) -> {r['mae']:.2f}" for r in table),
            "# Evidence: results/fuji_tune_cdog_t*_d*.csv, results/fuji_tune_cdog_summary.json.",
            "# Never edit after looking at test-split numbers.",
            f"score_threshold: {SCORE_THRESHOLD}",
            f"merge_k: {MERGE_K}",
            f"cdog_tau_px: {best['tau']:g}",
            f"cdog_delta: {best['delta']:g}",
            "",
        ]
        args.write_config.write_text("\n".join(lines))
        print("wrote", args.write_config)


if __name__ == "__main__":
    main()
