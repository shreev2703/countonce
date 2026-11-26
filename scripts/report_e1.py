"""Markdown tables for E1 (tracker side) from ``results/e1_tracker.csv`` and the prepared windows.

    python scripts/report_e1.py > /tmp/e1.md

CountOnce v1/v2/sync counts of each window (stride 1, capture order, identical detections, same
GT) are read from the prepared windows. They exist only at stride 1: reconstructing strided
subsets is not part of this experiment.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench import track_e1 as e1

CSV = Path("results/e1_tracker.csv")
PREP = Path("runs/fuji_core/e1")


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    wins = {}
    for p in PREP.glob("test_*.json"):
        w = e1.PreparedWindow.from_json(p.read_text())
        wins[(w.tree, w.side, w.k)] = w
    ok = [r for r in rows if r["status"] == "ok"]
    print(
        f"rows: {len(rows)} total, "
        + ", ".join(
            f"{s} {sum(r['status'] == s for r in rows)}"
            for s in sorted({r["status"] for r in rows})
        )
    )
    for status in ("failed", "skipped", "error"):
        for tr in ("bytetrack", "sam2"):
            sel = [r for r in rows if r["status"] == status and r["tracker"] == tr]
            if sel:
                ks = sorted({int(r["k"]) for r in sel})
                print(f"{status} {tr}: {len(sel)} rows, k in {ks}")
    vram = [float(r["peak_vram_mb"]) for r in ok if r["peak_vram_mb"]]
    print(f"SAM 2.1 peak VRAM (allocated) max {max(vram):.0f} MB" if vram else "no VRAM recorded")

    # strides on the same windows only: k=10 and 20 have at least 2 views at every stride
    print("\nStride sweep, k = 10 and 20 only (the same windows at every stride):\n")
    print("| tracker | order | stride | n rows | mean pred | mean GT | MAE | mean signed err |")
    print("|---|---|---|---|---|---|---|---|")
    for tr in ("bytetrack", "sam2"):
        for order in e1.ORDERS:
            for s in e1.STRIDES:
                sel = [
                    r
                    for r in ok
                    if (r["tracker"], r["order"], int(r["stride"])) == (tr, order, s)
                    and int(r["k"]) >= 10
                ]
                if not sel:
                    continue
                p = np.array([float(r["pred"]) for r in sel])
                g = np.array([float(r["gt"]) for r in sel])
                print(
                    f"| {tr} | {order} | {s} | {len(sel)} | {p.mean():.1f} | {g.mean():.1f} | "
                    f"{np.abs(p - g).mean():.1f} | {(p - g).mean():+.1f} |"
                )

    print(
        "\nOrder sensitivity (same views, shuffled minus capture order, per window and stride):\n"
    )
    print("| tracker | n pairs | pairs with a different count | mean abs change | max abs change |")
    print("|---|---|---|---|---|")
    for tr in ("bytetrack", "sam2"):
        by = {
            (r["tree"], r["side"], r["k"], r["stride"], r["order"]): float(r["pred"])
            for r in ok
            if r["tracker"] == tr
        }
        d = [
            abs(by[(*key[:4], "shuffled")] - v)
            for key, v in by.items()
            if key[4] == "capture" and (*key[:4], "shuffled") in by
        ]
        if d:
            print(
                f"| {tr} | {len(d)} | {sum(x > 0 for x in d)} | {np.mean(d):.1f} | {max(d):.0f} |"
            )

    # GT-free view of the same stride-1 windows: how far each method de-duplicates the detections
    print("\nMean counts without GT (stride 1, capture order, same windows and detections):\n")
    print("| k | n windows | in-ROI detections | ByteTrack | SAM 2.1 | v1 | v2 | sync |")
    print("|---|---|---|---|---|---|---|---|")
    first = {
        (r["tracker"], r["tree"], r["side"], int(r["k"])): r
        for r in ok
        if r["order"] == "capture" and r["stride"] == "1"
    }
    for k in sorted({key[3] for key in first}):
        keys = [
            key[1:]
            for key in first
            if key[0] == "bytetrack" and key[3] == k and ("sam2", *key[1:]) in first
        ]
        cols = [np.mean([float(first[("bytetrack", *x)]["n_dets_roi"]) for x in keys])]
        cols += [
            np.mean([float(first[(tr, *x)]["pred"]) for x in keys]) for tr in ("bytetrack", "sam2")
        ]
        cols += [np.mean([wins[x].countonce[a] for x in keys]) for a in ("v1", "v2", "sync")]
        print(f"| {k} | {len(keys)} | " + " | ".join(f"{c:.1f}" for c in cols) + " |")

    print(
        "\nStride 1, capture order, windows where a tracker and CountOnce both ran "
        "(error = |count - GT|, tree = mean of its two sides, 95% CI bootstraps trees):\n"
    )
    print(
        "| k | tracker | vs | n windows | n trees | tracker MAE | CountOnce MAE | "
        "tracker - CountOnce [CI] |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted({int(r["k"]) for r in ok}) + [0]:
        for tr in ("bytetrack", "sam2"):
            for ref in ("v1", "v2", "sync"):
                per_tree: dict[str, list[tuple[float, float]]] = defaultdict(list)
                for r in ok:
                    if (r["tracker"], r["order"], r["stride"]) != (tr, "capture", "1"):
                        continue
                    if k and int(r["k"]) != k:
                        continue
                    w = wins.get((r["tree"], r["side"], int(r["k"])))
                    if not w or not w.countonce:
                        raise SystemExit(f"no CountOnce counts for {r}; run `countonce`")
                    if w.countonce["gt"] != float(r["gt"]):
                        raise SystemExit(f"GT differs from CountOnce's for {r}")
                    per_tree[r["tree"]].append(
                        (
                            abs(float(r["pred"]) - float(r["gt"])),
                            abs(w.countonce[ref] - w.countonce["gt"]),
                        )
                    )
                if len(per_tree) < 2:
                    continue
                trees = sorted(per_tree)
                a = [np.mean([x for x, _ in per_tree[t]]) for t in trees]
                b = [np.mean([y for _, y in per_tree[t]]) for t in trees]
                n_win = sum(len(per_tree[t]) for t in trees)
                diffs = [x - y for x, y in zip(a, b, strict=True)]
                m, lo, hi = e1.paired_bootstrap_ci(diffs, trees)
                print(
                    f"| {k or 'all'} | {tr} | {ref} | {n_win} | {len(trees)} | {np.mean(a):.1f} | "
                    f"{np.mean(b):.1f} | {m:+.1f} [{lo:+.1f}, {hi:+.1f}] |"
                )


if __name__ == "__main__":
    main()
