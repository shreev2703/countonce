"""Render docs/SERVING.md from results/coldstart.csv and the admin cost rates.

    python scripts/make_serving_report.py [--csv results/coldstart.csv] \
        [--cost countonce/configs/cost.yaml] [--out docs/SERVING.md]

Anything without data renders ``TODO: fill from results``. Cost per 1,000 frames treats one warm
inference as one frame: p50 warm seconds * 1000 / 3600 * the cloud ``cost_per_hour`` rate.
Uses the most recent run in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

TODO = "TODO: fill from results"


def latest_run(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    last = rows[-1]["run_id"]
    return [r for r in rows if r["run_id"] == last]


def load_rate(path: Path | None, kind: str) -> float | None:
    if path is None or not path.exists():
        return None
    import yaml

    rates = (yaml.safe_load(path.read_text()) or {}).get("cost_per_hour") or {}
    v = rates.get(kind)
    return None if v is None else float(v)


def _fmt(v: float | None, spec: str = ".3f") -> str:
    return TODO if v is None else format(v, spec)


def render(rows: list[dict[str, str]], rate: float | None) -> str:
    def pick(phase: str) -> list[dict[str, str]]:
        return [r for r in rows if r["phase"] == phase]

    def one(phase: str) -> float | None:
        got = pick(phase)
        return float(got[0]["seconds"]) if got else None

    p50, p95 = one("warm_p50"), one("warm_p95")
    cost = None if p50 is None or rate is None else p50 * 1000 / 3600 * rate
    out = ["# Serving: cloud worker cold start and cost", ""]
    out += [
        "The cloud worker is the same image as the laptop worker (`docker/worker.Dockerfile`, "
        "PLAN 14.4). Weights are mounted at `/weights`, not baked in. The image has not been "
        "built yet.",
        "",
        "Measured with `countonce coldstart`; rows append to `results/coldstart.csv`.",
        "",
        "## Cold start",
        "",
        "| Phase | Name | Seconds |",
        "|---|---|---|",
        f"| Process start | interpreter | {_fmt(one('process_start'))} |",
        f"| Import | pipeline | {_fmt(one('import'))} |",
    ]
    loads = pick("model_load")
    if loads:
        out += [f"| Model load | {r['name']} | {float(r['seconds']):.3f} |" for r in loads]
    else:
        out.append(f"| Model load | all models | {TODO} |")
    out += [
        f"| First warm inference | warm | {_fmt(one('first_inference'))} |",
        "",
        "## Warm latency",
        "",
        "| p50 (s) | p95 (s) |",
        "|---|---|",
        f"| {_fmt(p50)} | {_fmt(p95)} |",
        "",
        "## Cost per 1,000 frames",
        "",
        "One warm inference is counted as one frame; rate is the `cloud` `cost_per_hour` in "
        "`cost.yaml`.",
        "",
        "| Rate (per GPU-hour) | Cost per 1,000 frames |",
        "|---|---|",
        f"| {_fmt(rate, '.4f') if rate is not None else 'TODO: set rate'} | "
        f"{_fmt(cost, '.4f') if cost is not None else TODO} |",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("results/coldstart.csv"))
    ap.add_argument("--cost", type=Path, default=Path("countonce/configs/cost.yaml"))
    ap.add_argument("--kind", default="cloud")
    ap.add_argument("--out", type=Path, default=Path("docs/SERVING.md"))
    a = ap.parse_args(argv)
    text = render(latest_run(a.csv), load_rate(a.cost, a.kind))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
