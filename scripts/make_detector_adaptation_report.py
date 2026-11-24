"""Render docs/DETECTOR_ADAPTATION.md from results/detector_adaptation.csv.

    python scripts/make_detector_adaptation_report.py [--csv results/detector_adaptation.csv] \
        [--out docs/DETECTOR_ADAPTATION.md]

Zero-shot vs fine-tuned, on seen and unseen orchards: per-image count MAE and end-to-end
deduplicated count error (mean absolute error per scene), each with its bootstrap CI. Uses the most
recent run in the CSV; any missing cell is ``TODO: fill from results``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

TODO = "TODO: fill from results"
SPLITS = ("seen", "unseen")
VARIANTS = ("zero_shot", "finetuned")
KINDS = (("per_image", "Per-image count MAE"), ("e2e", "End-to-end deduplicated count error"))


def latest_run(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    last = rows[-1]["run_id"]
    return [r for r in rows if r["run_id"] == last]


def _cell(rows: list[dict[str, str]], split: str, variant: str, kind: str) -> str:
    for r in rows:
        if (r["split"], r["variant"], r["kind"]) == (split, variant, kind):
            return (
                f"{float(r['estimate']):.2f} [{float(r['lo']):.2f}, {float(r['hi']):.2f}] "
                f"(n={r['n']})"
            )
    return TODO


def render(rows: list[dict[str, str]]) -> str:
    out = ["# Detector adaptation: zero-shot vs fine-tuned", ""]
    out += [
        "Track D (PLAN 11). LoRA or head-only fine-tuning of the detector on seen orchards "
        "(MinneApple, WGISD); Fuji-SfM is the unseen-orchard test. Values are mean absolute error "
        "with a 95% bootstrap CI over images (per-image) or scenes (end to end).",
        "",
        "The real CountGD fine-tune is gated on its weights; this table stays "
        f"`{TODO}` until `countonce finetune-detector` has run.",
        "",
    ]
    versions = sorted({r["adapter_version"] for r in rows if r.get("adapter_version")})
    if versions:
        out += [f"Adapter registry version: {', '.join(versions)}", ""]
    for kind, title in KINDS:
        out += [f"## {title}", "", "| Orchards | Zero-shot | Fine-tuned |", "|---|---|---|"]
        for split in SPLITS:
            cells = [_cell(rows, split, v, kind) for v in VARIANTS]
            out.append(f"| {split} | {cells[0]} | {cells[1]} |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=Path("results/detector_adaptation.csv"))
    ap.add_argument("--out", type=Path, default=Path("docs/DETECTOR_ADAPTATION.md"))
    args = ap.parse_args(argv)
    text = render(latest_run(args.csv))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
