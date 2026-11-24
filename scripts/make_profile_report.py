"""Render the optimization and repeat-job caching tables in docs/PROFILE.md.

    python scripts/make_profile_report.py [--csv CSV] [--profile-csv CSV] [--out docs/PROFILE.md]

``--csv`` is results/profile_variants.csv; ``--profile-csv`` is results/profile.csv, whose rows with
a ``cache_state`` feed the "Repeat-job caching" table (``countonce profile --variant cache``).

The table sits between ``<!-- profile-variants:start -->`` and ``<!-- profile-variants:end -->``;
the rest of the file is left alone. If the markers are absent the block is appended; if the
output file does not exist a minimal one is created. A missing CSV renders
``TODO: fill from results`` (never invented numbers). Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

START = "<!-- profile-variants:start -->"
END = "<!-- profile-variants:end -->"
CACHE_START = "<!-- profile-cache:start -->"
CACHE_END = "<!-- profile-cache:end -->"
CACHE_STATES = ("cold", "new_prompt", "repeat")
STATE_LABEL = {
    "cold": "Cold job",
    "new_prompt": "New prompt (recon hit)",
    "repeat": "Exact repeat (recon + detect hit)",
}
TODO = "TODO: fill from results"


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    xs = sorted(values)
    if not xs:
        raise ValueError("no values")
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _f(x: str) -> float | None:
    return float(x) if x not in ("", None) else None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """One summary per variant, in CSV order (the first variant is the baseline)."""
    order: list[str] = []
    for r in rows:
        if r["variant"] not in order:
            order.append(r["variant"])
    out: list[dict[str, object]] = []
    for name in order:
        mine = [r for r in rows if r["variant"] == name]
        err = next((r["status"] for r in mine if r.get("status", "ok") != "ok"), None)
        totals = [v for r in mine if r["stage"] == "total" and (v := _f(r["wall_s"])) is not None]
        recon = [v for r in mine if r["stage"] == "recon" and (v := _f(r["wall_s"])) is not None]
        peaks = [
            v for r in mine if r["stage"] in ("recon", "total") and (v := _f(r["peak_vram_mb"]))
        ]
        deltas = [
            v for r in mine if r["stage"] == "total" and (v := _f(r["count_delta"])) is not None
        ]
        out.append(
            {
                "variant": name,
                "error": err,
                "n": len(totals),
                "p50": percentile(totals, 50) if totals else None,
                "p90": percentile(totals, 90) if totals else None,
                "recon_p50": percentile(recon, 50) if recon else None,
                "peak": max(peaks) if peaks else None,
                "max_abs_delta": max((abs(d) for d in deltas), default=None),
            }
        )
    return out


def _cell(x: float | None, fmt: str = "{:.3f}") -> str:
    return TODO if x is None else fmt.format(x)


def render_table(summ: list[dict[str, object]]) -> str:
    base = summ[0] if summ else None
    lines = [
        "| Variant | n | p50 total (s) | p90 total (s) | Speedup vs baseline (p50) "
        "| Recon p50 (s) | Peak VRAM (MB) | VRAM vs baseline (MB) | Max abs count delta |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summ:
        if s["error"]:
            lines.append(f"| {s['variant']} | 0 | {s['error']} | | | | | | |")
            continue
        p50 = s["p50"]
        bp50 = base["p50"] if base else None
        speed = bp50 / p50 if isinstance(p50, float) and isinstance(bp50, float) and p50 else None
        peak, bpeak = s["peak"], base["peak"] if base else None
        dv = peak - bpeak if isinstance(peak, float) and isinstance(bpeak, float) else None
        lines.append(
            f"| {s['variant']}{' (baseline)' if s is base else ''} | {s['n']} "
            f"| {_cell(p50)} | {_cell(s['p90'])} "  # type: ignore[arg-type]
            f"| {_cell(speed, '{:.2f}x')} | {_cell(s['recon_p50'])} "  # type: ignore[arg-type]
            f"| {_cell(peak, '{:.0f}')} | {_cell(dv, '{:+.0f}')} "
            f"| {_cell(s['max_abs_delta'], '{:.0f}')} |"  # type: ignore[arg-type]
        )
    return "\n".join(lines)


def _share(mine: list[dict[str, str]], col: str) -> float | None:
    flags = [r[col] for r in mine if r["stage"] == "total" and r.get(col, "") != ""]
    return sum(f == "1" for f in flags) / len(flags) if flags else None


def summarize_cache(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """One summary per cache state present. Rows without a ``cache_state`` (old CSVs) are skipped.

    GPU-stage time is detect + recon wall time per repeat; hit flags are the share of repeats
    whose flag was 1.
    """
    out: list[dict[str, object]] = []
    for state in CACHE_STATES:
        mine = [r for r in rows if r.get("cache_state") == state]
        if not mine:
            continue
        by_rep: dict[str, dict[str, float]] = {}
        for r in mine:
            if (v := _f(r["wall_s"])) is not None:
                by_rep.setdefault(r["repeat"], {})[r["stage"]] = v
        totals = [d["total"] for d in by_rep.values() if "total" in d]
        gpu = [d["detect"] + d["recon"] for d in by_rep.values() if "detect" in d and "recon" in d]
        peaks = [
            v for r in mine if r["stage"] in ("recon", "total") and (v := _f(r["peak_vram_mb"]))
        ]

        out.append(
            {
                "state": state,
                "n": len(totals),
                "p50": percentile(totals, 50) if totals else None,
                "p90": percentile(totals, 90) if totals else None,
                "gpu_p50": percentile(gpu, 50) if gpu else None,
                "peak": max(peaks) if peaks else None,
                "recon_hit": _share(mine, "recon_cache_hit"),
                "det_hit": _share(mine, "det_cache_hit"),
            }
        )
    return out


def render_cache_table(summ: list[dict[str, object]]) -> str:
    cold = next((s for s in summ if s["state"] == "cold"), None)

    def ratio(key: str, s: dict[str, object]) -> str:
        a, b = (cold[key] if cold else None), s[key]
        ok = isinstance(a, float) and isinstance(b, float) and b > 0
        return f"{a / b:.2f}x" if ok else TODO  # type: ignore[operator]

    lines = [
        "| State | n | Recon hit | Detect hit | p50 total (s) | p90 total (s) "
        "| Savings vs cold (p50 total) | p50 GPU stages (s) | GPU-time savings vs cold "
        "| Peak VRAM (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summ:
        st = str(s["state"])
        lines.append(
            f"| {STATE_LABEL[st]} | {s['n']} | {_cell(s['recon_hit'], '{:.0%}')} "  # type: ignore[arg-type]
            f"| {_cell(s['det_hit'], '{:.0%}')} "  # type: ignore[arg-type]
            f"| {_cell(s['p50'])} | {_cell(s['p90'])} "  # type: ignore[arg-type]
            f"| {'1.00x' if st == 'cold' and s['p50'] else ratio('p50', s)} "
            f"| {_cell(s['gpu_p50'])} "  # type: ignore[arg-type]
            f"| {'1.00x' if st == 'cold' and s['gpu_p50'] else ratio('gpu_p50', s)} "
            f"| {_cell(s['peak'], '{:.0f}')} |"  # type: ignore[arg-type]
        )
    return "\n".join(lines)


def render_cache_block(csv_path: Path) -> str:
    summ = summarize_cache(load_rows(csv_path)) if csv_path.exists() else []
    if summ:
        body = render_cache_table(summ)
    else:
        body = (
            f"{TODO} (no cache-state rows in `{csv_path}`; run "
            "`countonce profile --variant cache` on the GPU)."
        )
    return (
        f"{CACHE_START}\n"
        "## Repeat-job caching\n\n"
        "Generated by `scripts/make_profile_report.py` from the profile CSV; do not edit by hand. "
        "Each repeat starts with empty caches: a cold job, the same photos with a new prompt "
        "(reconstruction reused), then an exact repeat (reconstruction and detections reused). "
        "Savings are cold p50 divided by the state's p50; GPU stages are detect + recon "
        "wall time. Measured, not assumed (PLAN section 13 item 6b).\n\n"
        f"{body}\n"
        f"{CACHE_END}"
    )


def render_block(csv_path: Path) -> str:
    if not csv_path.exists():
        body = (
            f"{TODO} (`{csv_path}` not found; run `countonce profile` with `variants:` on the GPU)."
        )
    else:
        rows = load_rows(csv_path)
        body = render_table(summarize(rows)) if rows else f"{TODO} (`{csv_path}` is empty)."
    return (
        f"{START}\n"
        "## Optimization variants\n\n"
        "Generated by `scripts/make_profile_report.py` from the variants CSV; do not edit by hand. "
        "Speedup is baseline p50 / variant p50 of end-to-end wall time. A count delta of 0 means "
        "the variant counted the same as the baseline (accuracy proxy only; benchmark MAPE is "
        "separate).\n\n"
        f"{body}\n"
        f"{END}"
    )


def _splice(doc: str, block: str, start: str, end: str) -> str:
    if start in doc and end in doc:
        head, rest = doc.split(start, 1)
        _, tail = rest.split(end, 1)
        return head + block + tail
    return doc.rstrip("\n") + "\n\n" + block + "\n"


def update_doc(doc: str | None, block: str, cache_block: str | None = None) -> str:
    if doc is None:
        intro = "# Profiling report\n\nSee PLAN section 13. Related: [[HANDOFF]], [[DECISIONS]]."
        doc = f"{intro}\n\n{block}\n"
    else:
        doc = _splice(doc, block, START, END)
    if cache_block is not None:
        doc = _splice(doc, cache_block, CACHE_START, CACHE_END)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=Path("results/profile_variants.csv"))
    ap.add_argument("--profile-csv", type=Path, default=Path("results/profile.csv"))
    ap.add_argument("--out", type=Path, default=Path("docs/PROFILE.md"))
    args = ap.parse_args(argv)
    doc = args.out.read_text() if args.out.exists() else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        update_doc(doc, render_block(args.csv), render_cache_block(args.profile_csv))
    )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
