"""Reliable-path associator choice from the Fuji-SfM validation windows (PLAN 0.1 rule 1).

Reads only the frozen-setting validation tuning rows (never ``results/bench.csv``). Writes count
MAE per method with a bootstrap CI over trees, plus paired differences against v2, and prints the
rule's choice: v2, unless sync has a lower validation MAE.

    PYTHONPATH=. .venv/bin/python scripts/decide_assoc_fuji_val.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from bench.stats import bootstrap_ci, paired_bootstrap

# frozen settings: configs/fuji_core.yaml and configs/fuji_core_sync.yaml (score 0.2, merge_k 0.3)
SOURCES = {
    "v1": Path("results/fuji_tune2_s0.2_k0.3.csv"),
    "v2": Path("results/fuji_tune2_s0.2_k0.3.csv"),
    "sync": Path("results/fuji_tune_sync_s0.2_k0.3.csv"),
}
OUT = Path("results/assoc_decision_fuji_val.csv")
N_BOOT = 10_000


def window_errors(path: Path, method: str) -> dict[tuple[str, str, str], float]:
    """Absolute count error per (tree, side, k); refuses non-validation or failed rows."""
    out: dict[tuple[str, str, str], float] = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["method"] != f"countonce_{method}":
                continue
            if r["split"] != "val" or r["status"] != "ok":
                raise ValueError(f"{path}: unexpected row {r['split']}/{r['status']}")
            out[(r["scene_id"], r["sample"], r["n_views"])] = abs(float(r["pred"]) - float(r["gt"]))
    return out


def per_tree(errors: dict[tuple[str, str, str], float]) -> list[float]:
    trees = sorted({key[0] for key in errors})
    return [float(np.mean([e for key, e in errors.items() if key[0] == t])) for t in trees]


def choose(mae: dict[str, float]) -> str:
    return "sync" if mae["sync"] < mae["v2"] else "v2"


def main() -> None:
    errors = {m: window_errors(p, m) for m, p in SOURCES.items()}
    assert all(e.keys() == errors["v2"].keys() for e in errors.values()), "windows differ"
    mae: dict[str, float] = {}
    rows = []
    for m, e in errors.items():
        ci = bootstrap_ci(per_tree(e), n_boot=N_BOOT, seed=0)
        mae[m] = float(np.mean(list(e.values())))
        diff, lo, hi, _ = paired_bootstrap(per_tree(e), per_tree(errors["v2"]), N_BOOT, seed=0)
        rows.append(
            {
                "method": m, "mae": round(mae[m], 2), "ci_low": round(ci.low, 2),
                "ci_high": round(ci.high, 2), "n_windows": len(e), "n_trees": ci.n,
                "diff_vs_v2": round(diff, 2), "diff_ci_low": round(lo, 2),
                "diff_ci_high": round(hi, 2), "source": SOURCES[m].as_posix(),
            }
        )  # fmt: skip
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(row)
    print("chosen:", choose(mae))


if __name__ == "__main__":
    main()
