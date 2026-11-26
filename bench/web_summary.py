"""Aggregate Fuji-SfM rows of ``results/bench.csv`` into ``results/benchmark_summary.csv``.

The web ``/benchmark`` page (PLAN §17.8) reads this file at build time. All statistics come from
``bench.fuji_report`` so the site and ``docs/RESULTS_FUJI.md`` show the same numbers. Only
aggregates are written: no images and no per-tree ground truth (PLAN §0.1 rule 4).

    python -m bench.web_summary
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from bench import fuji_report as fr

FIELDS = (
    "dataset", "method", "k", "metric", "estimate", "low", "high", "n",
    "protocol", "git_commit", "git_dirty", "timestamp_utc",
)  # fmt: skip
WEB_METRICS = ("mae", "signed_bias", "mape")
BASELINES = ("sum", "max")


def slice_wins(
    rows: Sequence[dict[str, str]], method: str, baseline: str, k: int
) -> tuple[int, int]:
    """``(slices where method has strictly lower |error| than baseline, slices compared)``."""
    err: dict[str, dict[tuple[str, str], float]] = {method: {}, baseline: {}}
    for r in rows:
        if r["status"] == "ok" and r["method"] in err and r["n_views"] == str(k):
            err[r["method"]][(r["scene_id"], r["sample"])] = abs(float(r["pred"]) - float(r["gt"]))
    shared = err[method].keys() & err[baseline].keys()
    return sum(err[method][s] < err[baseline][s] for s in shared), len(shared)


def summarise(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    """Long-format summary rows; empty when there is nothing to summarise."""
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return []
    meta = {
        "dataset": "fuji_sfm",
        "protocol": fr.PROTOCOL,
        "git_commit": ";".join(sorted({r["git_commit"][:12] for r in ok})),
        "git_dirty": str(any(r["git_dirty"] == "True" for r in ok)).lower(),
        "timestamp_utc": max(r["timestamp_utc"] for r in ok),
    }
    report = fr.compute(rows)
    out: list[dict[str, object]] = []

    def add(
        method: str, k: int, metric: str, est: object, lo: object, hi: object, n: object
    ) -> None:
        row = {"method": method, "k": k, "metric": metric, "estimate": est}
        out.append({**meta, **row, "low": lo, "high": hi, "n": n})

    for m in fr.METHODS:
        for k in fr.KS:
            for metric in WEB_METRICS:
                c = report["cells"].get(f"{m}|{k}|{metric}")
                if c:
                    add(m, k, metric, c["estimate"], c["low"], c["high"], c["n_trees"])
    for m in fr.PIPELINE:
        for b in BASELINES:
            for k in fr.KS:
                wins, n = slice_wins(rows, m, b, k)
                if n:
                    add(m, k, f"slices_won_vs_{b}", wins, "", "", n)
    return out


def write(csv_path: Path, out: Path) -> int:
    """Write the summary; with no usable rows, remove a stale one. Returns the row count."""
    rows = summarise(fr.fuji_rows(csv_path))
    if not rows:
        out.unlink(missing_ok=True)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--csv", type=Path, default=Path("results/bench.csv"))
    p.add_argument("--out", type=Path, default=Path("results/benchmark_summary.csv"))
    a = p.parse_args(argv)
    print(f"{write(a.csv, a.out)} summary rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
