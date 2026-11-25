"""Controlled failure-source experiment (PLAN §9.6): oracle/real detections x geometry.

    PYTHONPATH=. .venv/bin/python scripts/run_failure_sources.py tiny
    PYTHONPATH=. .venv/bin/python scripts/run_failure_sources.py synth --n-val 8 --n-test 8

Every associator sees identical upstream inputs per condition. On the synthetic fixture v1/v2/sync
use settings tuned on separate validation episodes; the tiny scene is a single scene with no
validation split, so it runs at default settings (labelled as such).
Writes ``--out-json``, ``--out-csv`` and prints a Markdown table.
Fuji-SfM / own captures: build a ``bench.failure_sources.SceneBundle`` per scene and call
``run_bundles`` (the M8 Fuji task does this); no change here is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bench import failure_sources as fs
from synth.geometry import GeometryConfig, generate_episode


def markdown(summary: list[dict], att: list[dict], note: str) -> str:
    lines = [
        note,
        "",
        "| condition | method | MAE | signed bias | missed | dup | FP | wrong merge |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['condition']} | {r['method']} | {r['mae']:.2f} | {r['signed_bias']:+.2f} | "
            f"{r['missed']:.2f} | {r['duplicates']:.2f} | {r['false_positives']:.2f} | "
            f"{r['wrong_merges']:.2f} |"
        )
    lines += [
        "",
        "| method | MAE full | removed by oracle detections | removed by oracle geometry |",
        "|---|---|---|---|",
    ]
    for a in att:
        lines.append(
            f"| {a['method']} | {a['mae_full']:.2f} | {a['removed_by_oracle_detections']:+.2f} | "
            f"{a['removed_by_oracle_geometry']:+.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", choices=["tiny", "synth"])
    ap.add_argument("--tiny-dir", default="examples/tiny_scene")
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--v3-checkpoint", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    a = ap.parse_args()
    stem = f"results/failure_sources_{a.scene}"
    out_json, out_csv = Path(a.out_json or f"{stem}.json"), Path(a.out_csv or f"{stem}_rows.csv")

    cfg = GeometryConfig(n_instances=(6, 15), n_views=(4, 6))
    if a.scene == "tiny":
        bundles, tuned = [fs.tiny_scene_bundle(a.tiny_dir)], {}
        note = "Tiny scene, default associator settings (no validation split exists for one scene)."
    else:
        val = [fs.synth_bundle(generate_episode(1000 + i, cfg), seed=i) for i in range(a.n_val)]
        bundles = [fs.synth_bundle(generate_episode(i, cfg), seed=i) for i in range(a.n_test)]
        tuned = fs.tune_associators(val)
        note = (
            f"Synthetic fixture, {a.n_test} test episodes, "
            f"settings tuned on {a.n_val} val episodes."
        )
    rows = fs.run_bundles(bundles, fs.build_associators(tuned, a.v3_checkpoint))
    summary = fs.summarize(rows)
    att = fs.attribution(summary)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"tuned": tuned, "summary": summary, "attribution": att}, indent=1)
    )
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(markdown(summary, att, note))


if __name__ == "__main__":
    main()
