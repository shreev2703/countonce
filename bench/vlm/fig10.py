"""Figure 10 (PLAN §9.7): VLM vs CountOnce counting error by view overlap.

Groups are the scene's ``overlap`` label from ``gt.json`` (copied into each job's config) and fall
back to the number of views when scenes carry no overlap label. Bars are scene-level MAPE with a
95% bootstrap CI (``bench.figures.point``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from bench.figures import Row, load_rows, point, rel_error
from bench.runner import read_rows
from countonce.plotting import apply_style, color_for, save


def _group_key(g: str) -> tuple[int, float | str]:
    try:
        return (0, float(g.split()[0]))
    except ValueError:
        return (1, g)


def split_rows(csv_path: str | Path) -> tuple[list[Row], dict[str, str], dict[str, str]]:
    """Usable rows plus run_key -> overlap group and method -> kind."""
    group: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for r in read_rows(csv_path):
        cfg = json.loads(r["config_json"] or "{}")
        kinds[r["method"]] = cfg.get("kind", "")
        group[r["run_key"]] = str(cfg["overlap"]) if "overlap" in cfg else f"{r['n_views']} views"
    return load_rows(csv_path), group, kinds


def fig_vlm_vs_countonce(csv_path: str | Path, out: Path) -> Path:
    rows, group, kinds = split_rows(csv_path)
    rows = [r for r in rows if r.ordered and kinds.get(r.method) in ("vlm", "countonce")]
    if not any(kinds[r.method] == "vlm" for r in rows):
        raise ValueError(f"{csv_path}: no ok VLM rows to plot")
    by: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in rows:
        by[(r.method, group[r.run_key])].append(r)
    groups = sorted({g for _, g in by}, key=_group_key)
    methods = sorted({m for m, _ in by}, key=lambda m: (kinds[m] != "countonce", m))
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        for k, g in enumerate(groups):
            p = point(by.get((m, g), []), rel_error)
            if p is None:
                continue
            x = k + (i - (len(methods) - 1) / 2) * width
            ax.bar(x, p.mean, width, color=color_for(m, i), label=m if k == 0 else None)
            ax.errorbar(x, p.mean, yerr=[[p.mean - p.lo], [p.hi - p.mean]], fmt="none",
                        ecolor="black", capsize=2)  # fmt: skip
    ax.set_xticks(range(len(groups)), groups)
    ax.set_xlabel("view overlap")
    ax.set_ylabel("MAPE (%)  (mean over scenes, 95% bootstrap CI)")
    ax.set_title("Frontier VLMs vs CountOnce by view overlap")
    ax.legend(fontsize=8)
    save(fig, out)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Render figure 10 from results/bench.csv")
    p.add_argument("--csv", type=Path, default=Path("results/bench.csv"))
    p.add_argument("--out", type=Path, default=Path("results/figures/vlm_vs_countonce.png"))
    a = p.parse_args(argv)
    print(fig_vlm_vs_countonce(a.csv, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
