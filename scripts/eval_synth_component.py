"""M5 component test: tune v1/v2/sync on the synthetic val split, compare with v3 on test.

    .venv-gpu/bin/python scripts/eval_synth_component.py

Both depth variants run every time:

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
import sys
from pathlib import Path
from typing import Any

import numpy as np

# `bench` and `synth` are not installed packages
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.stats import bootstrap_ci, paired_bootstrap
from bench.synth_component import (
    SYNC_GRID,
    V1_GRID,
    V2_GRID,
    Prepared,
    evaluate,
    make_sync,
    make_v1,
    make_v2,
    prepare,
    summarize_counts,
    tune,
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
        "v1": {"k": DEFAULT_K},
        "v2": {"max_reproj_norm": v2.max_reproj_norm, "depth_tol": v2.depth_tol},
        "sync": {
            "eig_threshold": sy.eig_threshold,
            "assign_threshold": sy.assign_threshold,
            "min_pair_affinity": sy.min_pair_affinity,
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
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-val", type=int, default=None)
    a = ap.parse_args(argv)

    val_files = sorted(Path(a.data, "val").glob("*.npz"))[: a.max_val]
    test_files = sorted(Path(a.data, "test").glob("*.npz"))
    dflt = defaults()
    v3 = V3Associator(a.checkpoint)
    result: dict[str, Any] = {}
    row_dump: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str], dict[str, float]] = {}
    for variant, finite in (("nan_depth", False), ("finite_depth", True)):
        val: list[Prepared] = prepare(val_files, finite)
        test: list[Prepared] = prepare(test_files, finite)
        t1 = tune("v1", make_v1, V1_GRID, dflt["v1"], val)
        t2 = tune("v2", make_v2, V2_GRID, dflt["v2"], val)
        ts = tune("sync", lambda p, v2=t2.best: make_sync(p, v2), SYNC_GRID, dflt["sync"], val)
        arms = {
            "v1 (default)": make_v1(dflt["v1"]),
            "v1 (tuned)": make_v1(t1.best),
            "v2 (default)": make_v2(dflt["v2"]),
            "v2 (tuned)": make_v2(t2.best),
            "sync (default)": make_sync(dflt["sync"]),
            "sync (tuned)": make_sync(ts.best, t2.best),
            "v3": v3,
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
            tuned={
                "v1": t1.best,
                "v2": t2.best,
                "sync": ts.best,
                "val_mae": {
                    "v1": [t1.default_val_mae, t1.val_mae],
                    "v2": [t2.default_val_mae, t2.val_mae],
                    "sync": [ts.default_val_mae, ts.val_mae],
                },
            },  # fmt: skip
            summary=summ,
            v3_minus=paired,
            n_val=len(val),
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
