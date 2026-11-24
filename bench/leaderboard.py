"""Markdown leaderboard from a folder of prediction files (PLAN §16).

Rows are only rendered when scored on a frozen split covering exactly that split's scenes;
anything else is refused with the reason, never silently listed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from bench import evaluate, splits


class LeaderboardError(ValueError):
    """One or more prediction files were scored on a non-frozen split."""


def _fmt(ci: dict[str, float] | None) -> str:
    if not ci:
        return "-"
    return f"{ci['estimate']:.2f} [{ci['low']:.2f}, {ci['high']:.2f}]"


def build_rows(
    pred_dir: str | Path,
    gt: str | Path,
    reference: str | None = None,
    splits_path: str | Path = splits.DEFAULT_PATH,
    n_boot: int = 2000,
) -> list[dict[str, Any]]:
    """Score every ``*.json`` in ``pred_dir``; raise LeaderboardError on non-frozen rows."""
    files = sorted(Path(pred_dir).glob("*.json"))
    if not files:
        raise evaluate.PredictionFormatError(f"no prediction files in {pred_dir}")
    gt_doc = evaluate.load_gt(gt)
    loaded = {f: evaluate.load_predictions(f) for f in files}
    ref = next((p for p in loaded.values() if p.method == reference), None)
    if reference and ref is None:
        raise evaluate.PredictionFormatError(f"reference method {reference!r} not in {pred_dir}")
    rows, refused = [], []
    for f, preds in loaded.items():
        row = evaluate.evaluate(
            preds, gt_doc, None if preds is ref else ref, n_boot=n_boot, splits_path=splits_path
        )
        if row["frozen_problems"]:
            refused.append(f"{f.name} ({preds.method}): " + "; ".join(row["frozen_problems"]))
        else:
            rows.append(row)
    if refused:
        raise LeaderboardError("refusing to render non-frozen rows:\n  " + "\n  ".join(refused))
    return sorted(rows, key=lambda r: r["mae"]["estimate"])


def render(rows: list[dict[str, Any]]) -> str:
    split = rows[0]["split"] if rows else "?"
    out = [
        f"# CountOnce-Bench leaderboard ({split} split)",
        "",
        "| Rank | Method | Kind | Scenes | MAE [95% CI] | MAPE | Within 10% | Interval cov. "
        "| Assoc. F1 | Paired diff vs ref (p) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        mape = r["mape"]
        vr = r.get("vs_reference")
        cov = r.get("interval_coverage")
        paired = "-" if not vr else "{:+.2f} ({:.3f})".format(vr["diff"], vr["p_bootstrap"])
        out.append(
            f"| {i} | {r['method']} | {r['kind']} | {r['n_scenes']} | {_fmt(r['mae'])} "
            f"| {'-' if mape != mape else f'{mape:.3f}'} | {r['within_10pct']:.2f} "
            f"| {'-' if cov is None else f'{cov:.2f}'} | {_fmt(r.get('assoc_f1'))} "
            f"| {paired} |"
        )
    out += ["", "Lower MAE is better. Paired diff = mean(method error - reference error).", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.leaderboard", description=__doc__)
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        text = render(build_rows(a.predictions_dir, a.gt, a.reference))
    except (LeaderboardError, evaluate.PredictionFormatError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
