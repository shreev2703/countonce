# ruff: noqa: E501
"""Render ``experimental/paper/tables/*.tex`` from ``results/bench.csv`` (PLAN §23 Paper).

    python scripts/make_paper_tables.py [--csv results/bench.csv] [--out experimental/paper/tables]

Always writes every file in ``TABLES`` so ``experimental/paper/main.tex`` never ``\\input``s a missing file.
A cell without data renders ``TODO: fill from results``. Values are means over scenes (a scene's
rows are averaged first) with 95% scene-level bootstrap CIs from ``bench.stats``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # bench/ lives at the repo root

TODO = "TODO: fill from results"
TABLES = ("main_results", "coverage", "ablation")


def esc(s: str) -> str:
    """Escape LaTeX specials in free text (method and scene names)."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):  # fmt: skip
        s = s.replace(a, b)
    return s


def _cell(vals: dict[str, float], fmt: str, scale: float = 1.0) -> str:
    from bench.stats import bootstrap_ci

    xs = list(vals.values())
    if not xs:
        return TODO
    ci = bootstrap_ci(xs, n_boot=2000, seed=0)
    return f"{fmt.format(ci.estimate * scale)} [{fmt.format(ci.low * scale)}, {fmt.format(ci.high * scale)}]"


def _table(caption: str, label: str, header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    cols = "l" * len(header)
    lines = [
        r"\begin{table*}[t]\centering\small",
        rf"\caption{{{caption}}}\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}\toprule",
        " & ".join(header) + r" \\ \midrule",
    ]
    lines += [" & ".join(r_) + r" \\" for r_ in body]
    lines += [r"\bottomrule\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def _placeholder(caption: str, label: str, why: str) -> str:
    return _table(caption, label, ["Result"], [[esc(f"{TODO} ({why})")]])


def _load(csv_path: Path) -> list:
    if not csv_path.exists():
        return []
    from bench.figures import load_rows

    try:
        return load_rows(csv_path)
    except ValueError:
        return []


def _ok(r) -> bool:  # type: ignore[no-untyped-def]
    return r.ordered


def main_results(rows: list) -> str:
    cap, lab = (
        "Count error by method and number of views (ordered input; 95\\% scene bootstrap CI).",
        "tab:main",
    )
    rows = [r for r in rows if _ok(r)]
    if not rows:
        return _placeholder(cap, lab, "no bench rows")
    from bench.figures import rel_error, scene_level

    out = []
    for m in sorted({r.method for r in rows}):
        for v in sorted({r.n_views for r in rows if r.method == m}):
            g = [r for r in rows if r.method == m and r.n_views == v]
            mape = scene_level(g, rel_error)
            w10 = scene_level(
                g, lambda r: float("nan") if r.gt == 0 else float(rel_error(r) <= 10.0)
            )
            f1 = scene_level([r for r in g if r.assoc_f1 is not None], lambda r: r.assoc_f1)
            out.append([esc(m), str(v), str(len({r.scene for r in g})), _cell(mape, "{:.1f}"),
                        _cell(w10, "{:.0f}", 100.0), _cell(f1, "{:.2f}")])  # fmt: skip
    return _table(
        cap,
        lab,
        ["Method", "Views", "Scenes", "MAPE (\\%)", "Within 10\\% (\\%)", "Assoc. F1"],
        out,
    )


def coverage(rows: list) -> str:
    from bench.figures import covered

    cap, lab = "Interval coverage of the ground-truth count (nominal 95\\%).", "tab:coverage"
    rows = [r for r in rows if r.low is not None and r.high is not None]
    if not rows:
        return _placeholder(cap, lab, "no rows with intervals")
    from bench.figures import scene_level

    out = [[esc(m), _cell(scene_level([r for r in rows if r.method == m], covered), "{:.0f}", 100.0)]
           for m in sorted({r.method for r in rows})]  # fmt: skip
    return _table(cap, lab, ["Method", "Coverage (\\%)"], out)


def ablation(rows: list) -> str:
    """Ablations are methods named ``<base>+<change>``/``<base>-<change>``; shown as one MAPE column."""
    cap, lab = "Ablations: MAPE over all view counts (ordered input).", "tab:ablation"
    rows = [r for r in rows if _ok(r)]
    names = sorted({r.method for r in rows if any(c in r.method for c in "+-")})
    if not names:
        return _placeholder(cap, lab, "no ablation methods in bench rows")
    from bench.figures import rel_error, scene_level

    out = [
        [esc(m), _cell(scene_level([r for r in rows if r.method == m], rel_error), "{:.1f}")]
        for m in names
    ]
    return _table(cap, lab, ["Variant", "MAPE (\\%)"], out)


RENDERERS: dict[str, Callable[[list], str]] = {
    "main_results": main_results, "coverage": coverage, "ablation": ablation,
}  # fmt: skip


def render_all(csv_path: Path, out_dir: Path) -> list[Path]:
    rows = _load(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in TABLES:
        p = out_dir / f"{name}.tex"
        p.write_text(RENDERERS[name](rows))
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=Path("results/bench.csv"))
    ap.add_argument("--out", type=Path, default=Path("experimental/paper/tables"))
    a = ap.parse_args(argv)
    for p in render_all(a.csv, a.out):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
