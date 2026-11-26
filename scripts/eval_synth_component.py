"""M5 component test: tune v1/v2/sync on the synthetic val split, compare with v3 on test.

    .venv-gpu/bin/python scripts/eval_synth_component.py

Stages: `--stage tune` (resumable grid search on val, parallel), `--stage test` (reads the
chosen rows from the tuning CSV, never tunes), `--stage all` (default). Both depth variants run
every time:

  nan_depth     the stand-in reconstruction has NaN depth, so the v2/sync visibility
                (occlusion) check never rejects anything on depth, as in the first v3 run.
  finite_depth  z-buffered depth with multiplicative noise (instances occlude one another
                over a ground plane), so the visibility check is active.

Outputs: a summary CSV (one row per variant and method; columns mae, signed_bias,
within_10pct and zero_scene_false_count first, then the error decomposition and F1), a
per-episode CSV and a JSON file. zero_scene_false_count is nan when n_zero is 0.
This is a component test on generated geometry, not evidence about real photos.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# `bench` and `synth` are not installed packages
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.stats import bootstrap_ci, paired_bootstrap
from bench.synth_component import (
    Prepared,
    StopAfter,
    chosen_params,
    evaluate,
    make_sync,
    make_v1,
    make_v2,
    prepare,
    read_tuning_rows,
    run_tune,
    summarize_counts,
    tuning_val_mae,
    write_summary_csv,
)
from countonce.merge.cluster import DEFAULT_K
from countonce.merge.match import V2Config
from countonce.merge.sync import SyncConfig
from countonce.merge.v3 import V3Associator


def defaults() -> dict[str, dict[str, Any]]:
    v2 = V2Config()
    sy = SyncConfig()
    return {
        "v1": {"k": DEFAULT_K, "score_threshold": 0.0},
        "v2": {
            "max_reproj_norm": v2.max_reproj_norm,
            "depth_tol": v2.depth_tol,
            "score_threshold": 0.0,
        },
        "sync": {
            "eig_threshold": sy.eig_threshold,
            "assign_threshold": sy.assign_threshold,
            "score_threshold": 0.0,
        },
    }


def ci(vals: list[float], n_boot: int) -> tuple[float, float, float]:
    c = bootstrap_ci(vals, n_boot=n_boot, seed=0)
    return c.estimate, c.low, c.high


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default="data/dag_v3_capped/episodes")
    ap.add_argument("--checkpoint", default="runs/match_v3_capped/best.pt")
    ap.add_argument("--out-json", default="results/synth_component.json")
    ap.add_argument("--out-csv", default="results/synth_component_summary.csv")
    ap.add_argument("--out-rows-csv", default="results/synth_component_rows.csv")
    ap.add_argument("--out-tuning-csv", default="results/synth_component_tuning.csv")
    ap.add_argument("--device", default="cpu", help="device for v3 inference (cpu or cuda)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-val", type=int, default=None, help="tune on the first N val episodes")
    ap.add_argument(
        "--stage",
        choices=("tune", "test", "all"),
        default="all",
        help="tune: grid search on val, resumable, appends to the tuning CSV; "
        "test: read the chosen rows from that CSV and score the test split (never tunes)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 3) - 2),
        help="processes for the tune stage",
    )
    ap.add_argument("--stop-after", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    dflt = defaults()
    tuning_csv = Path(a.out_tuning_csv)
    if a.stage in ("tune", "all"):
        val_files = sorted(Path(a.data, "val").glob("*.npz"))[: a.max_val]
        try:
            run_tune(val_files, tuning_csv, dflt, workers=a.workers, stop_after=a.stop_after)
        except StopAfter:
            print(f"stopped after {a.stop_after} rows (test hook)")
            return 1
    if a.stage == "tune":
        return 0

    tuned_all = chosen_params(tuning_csv)  # exits if any method has no chosen row
    val_maes = tuning_val_mae(tuning_csv)
    test_files = sorted(Path(a.data, "test").glob("*.npz"))
    v3 = V3Associator(a.checkpoint, device=a.device)
    v3s = V3Associator(a.checkpoint, device=a.device, synchronize=True)
    result: dict[str, Any] = {}
    row_dump: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str], dict[str, float]] = {}
    for variant, finite in (("nan_depth", False), ("finite_depth", True)):
        test: list[Prepared] = prepare(test_files, finite)
        best = tuned_all[variant]
        arms = {
            "v1 (default)": make_v1(dflt["v1"]),
            "v1 (tuned)": make_v1(best["v1"]),
            "v2 (default)": make_v2(dflt["v2"]),
            "v2 (tuned)": make_v2(best["v2"]),
            "sync (default)": make_sync(dflt["sync"]),
            "sync (tuned)": make_sync(best["sync"], best["v2"]),
            "v3": v3,
            "v3+sync": v3s,
        }
        rows = {name: evaluate(assoc, test) for name, assoc in arms.items()}
        summ = {}
        for name, r in rows.items():
            s: dict[str, Any] = summarize_counts(r)
            summaries[(variant, name)] = dict(s)
            m, lo, hi = ci([x["abs_err"] for x in r], a.n_boot)
            s["mae_ci"] = [m, lo, hi]
            summ[name] = s
            for i, x in enumerate(r):
                row_dump.append(dict(variant=variant, method=name, episode=i, **x))
        base_err = [x["abs_err"] for x in rows["v3"]]
        paired = {}
        for name in arms:
            if name != "v3":
                d, lo, hi, p = paired_bootstrap(
                    base_err, [x["abs_err"] for x in rows[name]], n_boot=a.n_boot, seed=0
                )
                paired[name] = dict(diff=d, lo=lo, hi=hi, p=p)
        result[variant] = dict(
            tuned={**best, "val_mae": val_maes[variant]},
            summary=summ,
            v3_minus=paired,
            n_val=int(next(iter(read_tuning_rows(tuning_csv)))["n_val"]),
            n_test=len(test),
            mean_true_count=float(np.mean([d.ep.true_count() for d in test])),
        )
        print(variant, json.dumps(result[variant]["tuned"], default=float))
    for out in (a.out_json, a.out_rows_csv):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(result, indent=1, default=float))
    write_summary_csv(Path(a.out_csv), summaries)
    with Path(a.out_rows_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dump[0]))
        w.writeheader()
        w.writerows(row_dump)
    print(f"wrote {a.out_json}, {a.out_csv}, {a.out_rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
