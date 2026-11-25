"""M5 component test: tune v1/v2/sync on the synthetic val split, compare with v3 on test.

    PYTHONPATH=. .venv-gpu/bin/python scripts/eval_synth_component.py

Two depth variants: ``nan_depth`` (visibility check inert, as in the first v3 run) and
``finite_depth`` (z-buffered noisy depth, check active). Writes ``results/synth_component.json``
and ``results/synth_component_rows.csv`` and prints the Markdown table for the doc.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/dag_v3_capped/episodes")
    ap.add_argument("--checkpoint", default="runs/match_v3_capped/best.pt")
    ap.add_argument("--out-json", default="results/synth_component.json")
    ap.add_argument("--out-csv", default="results/synth_component_rows.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-val", type=int, default=None)
    a = ap.parse_args()

    val_files = sorted(Path(a.data, "val").glob("*.npz"))[: a.max_val]
    test_files = sorted(Path(a.data, "test").glob("*.npz"))
    dflt = defaults()
    v3 = V3Associator(a.checkpoint)
    result: dict[str, Any] = {}
    row_dump: list[dict[str, Any]] = []
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
            s = summarize_counts(r)
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
    Path(a.out_json).write_text(json.dumps(result, indent=1, default=float))
    with Path(a.out_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dump[0]))
        w.writeheader()
        w.writerows(row_dump)
    print(f"wrote {a.out_json}, {a.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
