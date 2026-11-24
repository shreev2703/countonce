"""Pilot report (PLAN 19 Phase 5): usage, repeat use, p50/p95 job time, feedback-accuracy proxy.

Reads the job store (``jobs`` + ``results``) and, optionally, a corrections file (the store has no
corrections table, see docs/DEVIATIONS.md). Every number is computed from rows; with no rows the
report renders ``TODO: fill from results`` in each cell. Bootstrap CIs are percentile intervals
(seeded) and are only reported when ``n >= MIN_CI_N``.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from countonce.corrections import Correction, load_corrections

TODO = "TODO: fill from results"
MIN_CI_N = 5
N_BOOT = 1000
SEED = 0


def bootstrap_ci(
    values: Sequence[Any], stat: Callable[[Sequence[Any]], float], *, level: float = 0.95
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of ``stat`` over resampled ``values``; None when n is too small."""
    n = len(values)
    if n < MIN_CI_N:
        return None
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    arr = list(values)
    boots = [stat([arr[i] for i in row]) for row in idx]
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return float(lo), float(hi)


def _metric(values: Sequence[Any], stat: Callable[[Sequence[Any]], float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {"value": float(stat(values)), "n": len(values), "ci": bootstrap_ci(values, stat)}


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")


def _load_rows(db: Path, since: float | None) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT j.id, j.user_id, j.status, j.site_id, j.captured_at, j.created_at, "
            "j.finished_at, r.count AS count FROM jobs j LEFT JOIN results r ON r.job_id = j.id "
            "WHERE j.status != 'deleted'"
        )
        params: list[Any] = []
        if since is not None:
            sql += " AND j.created_at >= ?"
            params.append(since)
        return [dict(r) for r in conn.execute(sql + " ORDER BY j.created_at", params)]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _is_repeat(jobs: list[dict[str, Any]]) -> bool:
    if len({_day(j["created_at"]) for j in jobs}) >= 2:
        return True
    visits: dict[str, set[float]] = defaultdict(set)
    for j in jobs:
        if j["site_id"]:
            visits[j["site_id"]].add(float(j["captured_at"] or j["created_at"]))
    return any(len(v) >= 2 for v in visits.values())


def compute_pilot(
    db: Path | str, *, since: float | None = None, corrections: Sequence[Correction] = ()
) -> dict[str, Any]:
    """Pilot metrics as a dict of ``{value, n, ci}`` (None when there is no data)."""
    rows = _load_rows(Path(db), since)
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_user[r["user_id"]].append(r)
    done = [r for r in rows if r["status"] == "done"]
    done_ids = {r["id"] for r in done}
    times = [
        float(r["finished_at"]) - float(r["created_at"])
        for r in done
        if r["finished_at"] is not None
    ]
    mean = lambda v: float(np.mean(v))  # noqa: E731
    out: dict[str, Any] = {
        "n_jobs": len(rows),
        "active_users": len(by_user) or None,
        "jobs_per_user": _metric([len(v) for v in by_user.values()], mean),
        "repeat_use_rate": _metric([_is_repeat(v) for v in by_user.values()], mean),
        "time_p50_s": _metric(times, lambda v: float(np.percentile(v, 50))),
        "time_p95_s": _metric(times, lambda v: float(np.percentile(v, 95))),
        "corrected_share": None,
        "corrections_per_instance": None,
        "rejects_per_job": None,
    }
    if corrections and done:
        mine = [c for c in corrections if c.job_id in done_ids]
        per_job: dict[str, list[Correction]] = defaultdict(list)
        for c in mine:
            per_job[c.job_id].append(c)
        out["corrected_share"] = _metric([r["id"] in per_job for r in done], mean)
        out["rejects_per_job"] = _metric(
            [sum(c.action == "reject" for c in per_job.get(r["id"], [])) for r in done], mean
        )
        pairs = [(len(per_job.get(r["id"], [])), int(r["count"] or 0)) for r in done]
        if sum(p[1] for p in pairs) > 0:
            out["corrections_per_instance"] = _metric(
                pairs, lambda v: sum(a for a, _ in v) / max(1, sum(b for _, b in v))
            )
    return out


def _cell(m: dict[str, Any] | None, fmt: str = "{:.2f}", pct: bool = False) -> tuple[str, str, str]:
    if m is None:
        return TODO, TODO, TODO
    scale = 100.0 if pct else 1.0
    suf = "%" if pct else ""
    val = fmt.format(m["value"] * scale) + suf
    ci = m["ci"]
    ci_s = (
        f"[{fmt.format(ci[0] * scale)}, {fmt.format(ci[1] * scale)}]{suf}"
        if ci
        else (f"n/a (n<{MIN_CI_N})")
    )
    return val, ci_s, str(m["n"])


def render_pilot(metrics: dict[str, Any], *, since: float | None = None) -> str:
    """Markdown for ``docs/PILOT.md``."""
    au = metrics["active_users"]
    rows = [
        ("Active users", (str(au) if au else TODO, "n/a", str(au) if au else TODO)),
        ("Jobs per user (mean)", _cell(metrics["jobs_per_user"])),
        ("Repeat-use rate", _cell(metrics["repeat_use_rate"], "{:.0f}", pct=True)),
        ("End-to-end time p50 (s)", _cell(metrics["time_p50_s"], "{:.1f}")),
        ("End-to-end time p95 (s)", _cell(metrics["time_p95_s"], "{:.1f}")),
        ("Jobs with corrections", _cell(metrics["corrected_share"], "{:.0f}", pct=True)),
        ("Corrections per instance", _cell(metrics["corrections_per_instance"], "{:.3f}")),
        ("Rejects per job", _cell(metrics["rejects_per_job"], "{:.2f}")),
    ]
    since_s = datetime.fromtimestamp(since, UTC).isoformat(timespec="seconds") if since else "all"
    lines = [
        "# Pilot report",
        "",
        "Generated by `countonce pilot-report`. Every cell is `TODO: fill from results` until the",
        "job store holds pilot jobs. CIs are 95% percentile bootstrap, shown when n >= "
        f"{MIN_CI_N}.",
        "",
        f"Window start: {since_s}. Jobs counted: {metrics['n_jobs'] or TODO}.",
        "",
        "Repeat use = a user with jobs on >= 2 different days, or >= 2 visits to one site.",
        "Time = job `finished_at - created_at` over finished jobs. Feedback accuracy is a proxy",
        "(corrections per instance, rejects per job), not ground-truth accuracy.",
        "",
        "| Metric | Value | 95% CI | n |",
        "|---|---|---|---|",
    ]
    lines += [f"| {name} | {v} | {ci} | {n} |" for name, (v, ci, n) in rows]
    lines += ["", "See [[INTERVIEWS]] for qualitative notes.", ""]
    return "\n".join(lines)


def write_pilot_report(
    db: Path | str,
    out: Path | str,
    *,
    since: float | None = None,
    corrections_path: Path | str | None = None,
) -> Path:
    corrections = load_corrections(corrections_path) if corrections_path else []
    text = render_pilot(compute_pilot(db, since=since, corrections=corrections), since=since)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
