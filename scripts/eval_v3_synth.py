"""Held-out synthetic evaluation of v1 / v2 / v3 (``co_v3_vggt``'s matcher) with bootstrap CIs.

Runs ``bench.synth_assoc`` on the test episodes (resumable, rows appended to ``--bench-csv``),
then bootstraps per-episode metrics over episodes and writes a long-format summary CSV
(``method, metric, mean, lo, hi, n``) plus paired v3-vs-baseline differences.

    .venv-gpu/bin/python scripts/eval_v3_synth.py --test-dir data/dag_v3_capped/episodes/test \
        --checkpoint runs/match_v3_capped/best.pt
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from bench.stats import bootstrap_ci, paired_bootstrap
from bench.synth_assoc import SynthAssocConfig, run_synth_assoc

METRICS = ("assoc_f1", "duplicate_rate", "miss_rate", "count_error")


def per_episode(bench_csv: Path, methods: list[str]) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {m: {} for m in methods}
    for row in csv.DictReader(io.StringIO(bench_csv.read_text())):
        if row["method"] in out and row["status"] == "ok" and row["domain"] == "synthetic":
            out[row["method"]][row["scene_id"]] = {m: float(row[m]) for m in METRICS}
    common = sorted(set.intersection(*(set(v) for v in out.values())))
    return {m: {k: [out[m][s][k] for s in common] for k in METRICS} for m in methods}


def summarize(data: dict[str, dict[str, list[float]]], n_boot: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for method, cols in data.items():
        for metric, vals in cols.items():
            ci = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            rows.append(
                dict(method=method, metric=metric, mean=ci.estimate, lo=ci.low, hi=ci.high, n=ci.n)
            )
    if "v3" in data:
        for base in (m for m in data if m != "v3"):
            for metric in METRICS:
                d, lo, hi, p = paired_bootstrap(
                    data["v3"][metric], data[base][metric], n_boot=n_boot, seed=seed
                )
                rows.append(
                    dict(
                        method=f"v3-minus-{base}", metric=metric, mean=d, lo=lo, hi=hi,
                        n=len(data["v3"][metric]), p=p,
                    )
                )  # fmt: skip
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-dir", default="data/dag_v3_capped/episodes/test")
    ap.add_argument("--checkpoint", default="runs/match_v3_capped/best.pt")
    ap.add_argument("--bench-csv", default="results/bench_synth_assoc_v3_capped.csv")
    ap.add_argument("--out", default="results/v3_capped_synth_eval.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    methods = ["v1", "v2", "v3"]
    cfg = SynthAssocConfig(
        test_dir=a.test_dir, csv=a.bench_csv, methods=methods, v3_checkpoint=a.checkpoint
    )
    ran, skipped, errored = run_synth_assoc(cfg)
    print(f"synth-assoc: ran {ran}, skipped {skipped}, errored {errored} -> {a.bench_csv}")
    rows = summarize(per_episode(Path(a.bench_csv), methods), a.n_boot, a.seed)
    cols = ["method", "metric", "mean", "lo", "hi", "n", "p"]
    with Path(a.out).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=1))
    return 0 if not errored else 1


if __name__ == "__main__":
    raise SystemExit(main())
