"""Admin metrics and alert rules over the job store (PLAN 14.6, 16B.2).

``compute_metrics`` reads a ``JobStore`` (``list_jobs`` / ``get_result``; feedback rows through the
optional ``feedback`` table of a ``SqliteJobStore``) and returns plain dicts. ``load_rules`` +
``evaluate_alerts`` turn ``countonce/configs/alerts.yaml`` into a typed ``Alert`` list.
``render_html`` writes
a static page; when a drift report is given its PSI table is included.

Definitions: success rate = done / (done + failed + dead_letter). Total time = ``finished_at -
started_at``; per-stage time comes from ``timings`` in the stored ``result_json``. Percentiles use
linear interpolation (``numpy.percentile`` default).
"""

from __future__ import annotations

import html
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from countonce.jobs.store import JobStore

FAILED = ("failed", "dead_letter")
RULE_TYPES = ("failure_rate_spike", "queue_age", "p95_regression", "cost_per_job")
WORKER_KINDS = ("laptop", "cloud")
SEVERITIES = ("warning", "critical")


@dataclass(frozen=True)
class Alert:
    rule: str
    severity: str
    message: str
    value: float
    threshold: float


@dataclass(frozen=True)
class AlertRule:
    id: str
    type: str
    severity: str = "warning"
    window_s: float = 3600.0
    baseline_s: float = 86400.0
    factor: float = 2.0
    min_rate: float = 0.0
    min_jobs: int = 5
    max_age_s: float = 900.0
    max_cost: float | None = None  # cost_per_job: mean cost of recent done jobs above this fires


def percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None}
    p50, p90, p95 = np.percentile(np.asarray(values, dtype=float), [50, 90, 95])
    return {"p50": float(p50), "p90": float(p90), "p95": float(p95)}


def _duration(job: Mapping[str, Any]) -> float | None:
    if job.get("started_at") is None or job.get("finished_at") is None:
        return None
    return float(job["finished_at"]) - float(job["started_at"])


def _feedback_accuracy(store: JobStore) -> dict[str, Any] | None:
    """``{n, accuracy}`` from the ``feedback`` table, or None when there is no such data."""
    if hasattr(store, "feedback_accuracy"):  # Parquet export (monitor.etl.ParquetStore)
        return store.feedback_accuracy()  # type: ignore[no-any-return]
    path = getattr(store, "path", None)
    if path is None:
        return None
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT COUNT(*), SUM(correct) FROM feedback").fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    n = int(row[0])
    return {"n": n, "accuracy": float(row[1]) / n} if n else None


def load_cost_rates(path: str | Path) -> dict[str, float | None]:
    """``cost_per_hour`` per worker kind from ``cost.yaml``; missing/null stays None."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    rates = raw.get("cost_per_hour") or {}
    unknown = set(rates) - set(WORKER_KINDS)
    if unknown:
        raise ValueError(f"cost_per_hour: unknown worker kind(s) {sorted(unknown)}")
    out: dict[str, float | None] = {k: None for k in WORKER_KINDS}
    for k, v in rates.items():
        if v is not None and float(v) < 0:
            raise ValueError(f"cost_per_hour.{k} must be >= 0")
        out[k] = None if v is None else float(v)
    return out


def _worker_kinds(store: JobStore) -> dict[str, str]:
    if hasattr(store, "worker_kinds"):  # Parquet export
        return store.worker_kinds()  # type: ignore[no-any-return]
    path = getattr(store, "path", None)
    if path is None:
        return {}
    conn = sqlite3.connect(path)
    try:
        return {r[0]: r[1] for r in conn.execute("SELECT id, kind FROM workers")}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def _job_usage(store: JobStore, job: Mapping[str, Any]) -> dict[str, Any] | None:
    """Stage timings, GPU-seconds (sum of stored stage timings) and frame count of a done job."""
    res = store.get_result(job["id"])
    raw = res.get("result_json") if res else None
    if not raw:
        return None
    doc = json.loads(raw)
    timings = {s: float(t) for s, t in (doc.get("timings") or {}).items()}
    frames = doc.get("n_frames")
    if frames is None and isinstance(doc.get("per_frame_counts"), Mapping):
        frames = len(doc["per_frame_counts"])
    return {"timings": timings, "gpu_s": sum(timings.values()), "frames": frames}


def _cost(gpu_s: float, rate: float | None) -> float | None:
    """Cost of ``gpu_s`` seconds at ``rate`` per hour; None (never 0) while the rate is unset."""
    return None if rate is None else gpu_s / 3600.0 * rate


def compute_cost(
    store: JobStore, cost_per_hour: Mapping[str, float | None] | None = None
) -> dict[str, Any]:
    """GPU-seconds, cost per job / per 1,000 frames and per-stage p50/p95, per worker kind."""
    rates: dict[str, float | None] = dict.fromkeys(WORKER_KINDS)
    rates.update(cost_per_hour or {})
    kinds = _worker_kinds(store)
    groups: dict[str, list[dict[str, Any]]] = {}
    for j in store.list_jobs():
        if j["status"] != "done":
            continue
        use = _job_usage(store, j)
        if use is not None:
            kind = kinds.get(j.get("worker_id") or "", "unknown")
            groups.setdefault(kind, []).append(use)
    by_kind: dict[str, Any] = {}
    for kind, uses in sorted(groups.items()):
        rate = rates.get(kind)
        gpu = [u["gpu_s"] for u in uses]
        stages: dict[str, list[float]] = {}
        for u in uses:
            for s, t in u["timings"].items():
                stages.setdefault(s, []).append(t)
        framed = [u for u in uses if u["frames"]]
        frames = sum(int(u["frames"]) for u in framed)
        framed_cost = _cost(sum(u["gpu_s"] for u in framed), rate)
        by_kind[kind] = {
            "n_jobs": len(uses),
            "cost_per_hour": rate,
            "gpu_seconds_total": sum(gpu),
            "gpu_seconds_per_job": {**percentiles(gpu), "mean": sum(gpu) / len(gpu)},
            "cost_total": _cost(sum(gpu), rate),
            "cost_per_job_mean": _cost(sum(gpu) / len(gpu), rate),
            "cost_per_1000_frames": (
                framed_cost / frames * 1000.0 if framed_cost is not None and frames else None
            ),
            "n_jobs_with_frames": len(framed),
            "stage_time_s": {
                s: {k: v for k, v in percentiles(t).items() if k in ("p50", "p95")}
                for s, t in sorted(stages.items())
            },
        }
    return {"cost_per_hour": rates, "by_worker_kind": by_kind}


def compute_metrics(
    store: JobStore, now: float, cost_per_hour: Mapping[str, float | None] | None = None
) -> dict[str, Any]:
    jobs = store.list_jobs()
    by_status = Counter(j["status"] for j in jobs)
    per_day = Counter(
        datetime.fromtimestamp(j["created_at"], UTC).strftime("%Y-%m-%d") for j in jobs
    )
    n_done = by_status["done"]
    n_failed = sum(by_status[s] for s in FAILED)
    finished = n_done + n_failed

    dead = [j for j in jobs if j["status"] == "dead_letter"]
    dead_by_code = Counter(j.get("error_code") or "UNKNOWN" for j in dead)

    totals: list[float] = []
    stage_vals: dict[str, list[float]] = {}
    vram: list[float] = []
    cache_hits = 0
    rungs: Counter[int] = Counter()
    n_results = 0
    for j in jobs:
        if j["status"] != "done":
            continue
        d = _duration(j)
        if d is not None:
            totals.append(d)
        res = store.get_result(j["id"])
        if res is None:
            continue
        n_results += 1
        cache_hits += int(bool(res["recon_cache_hit"]))
        rungs[int(res["oom_fallback_rung"])] += 1
        vram.append(float(res["peak_vram_mb"]))
        raw = res.get("result_json")
        timings = json.loads(raw).get("timings", {}) if raw else {}
        for stage, sec in timings.items():
            stage_vals.setdefault(stage, []).append(float(sec))

    queued = [j for j in jobs if j["status"] == "queued"]
    oldest = max((now - j["created_at"] for j in queued), default=None)
    return {
        "generated_at": now,
        "n_jobs": len(jobs),
        "by_status": dict(by_status),
        "jobs_per_day": dict(sorted(per_day.items())),
        "success_rate": n_done / finished if finished else None,
        "dead_letter_rate": len(dead) / finished if finished else None,
        "dead_letter_by_code": dict(sorted(dead_by_code.items())),
        "time_s": {"total": percentiles(totals)},
        "stage_time_s": {s: percentiles(v) for s, v in sorted(stage_vals.items())},
        "peak_vram_mb": max(vram) if vram else None,
        "recon_cache_hit_rate": cache_hits / n_results if n_results else None,
        "oom_rung_histogram": {str(k): v for k, v in sorted(rungs.items())},
        "queue_depth": len(queued),
        "oldest_queued_age_s": oldest,
        "feedback": _feedback_accuracy(store),
        "cost": compute_cost(store, cost_per_hour),
    }


# ---------------------------------------------------------------- alerts


def load_rules(path: str | Path) -> list[AlertRule]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    rules = []
    for item in raw.get("rules") or []:
        rule = AlertRule(**item)
        if rule.type not in RULE_TYPES:
            raise ValueError(f"alert rule {rule.id!r}: unknown type {rule.type!r}")
        if rule.severity not in SEVERITIES:
            raise ValueError(f"alert rule {rule.id!r}: unknown severity {rule.severity!r}")
        rules.append(rule)
    return rules


def _window(jobs: Sequence[Mapping[str, Any]], lo: float, hi: float) -> list[Mapping[str, Any]]:
    return [j for j in jobs if j.get("finished_at") is not None and lo < j["finished_at"] <= hi]


def _fail_rate(jobs: Sequence[Mapping[str, Any]]) -> float | None:
    done = [j for j in jobs if j["status"] == "done" or j["status"] in FAILED]
    return sum(j["status"] in FAILED for j in done) / len(done) if done else None


def evaluate_alerts(
    store: JobStore,
    rules: Sequence[AlertRule],
    now: float,
    cost_per_hour: Mapping[str, float | None] | None = None,
) -> list[Alert]:
    jobs = store.list_jobs()
    alerts: list[Alert] = []
    for r in rules:
        recent = _window(jobs, now - r.window_s, now)
        base = _window(jobs, now - r.window_s - r.baseline_s, now - r.window_s)
        if r.type == "queue_age":
            ages = [now - j["created_at"] for j in jobs if j["status"] == "queued"]
            if ages and max(ages) > r.max_age_s:
                alerts.append(
                    Alert(
                        r.id,
                        r.severity,
                        f"oldest queued job is {max(ages):.0f}s old",
                        max(ages),
                        r.max_age_s,
                    )
                )
        elif r.type == "failure_rate_spike":
            cur, ref = _fail_rate(recent), _fail_rate(base)
            if cur is None or len(recent) < r.min_jobs or cur < r.min_rate:
                continue
            limit = r.factor * (ref or 0.0)
            if cur > limit:
                alerts.append(
                    Alert(
                        r.id,
                        r.severity,
                        f"failure rate {cur:.0%} vs baseline {(ref or 0.0):.0%}",
                        cur,
                        limit,
                    )
                )
        elif r.type == "cost_per_job":
            # Needs a rate and a threshold; with either unset the rule stays quiet (no made-up 0).
            if r.max_cost is None:
                continue
            kinds = _worker_kinds(store)
            costs: list[float] = []
            for j in recent:
                rate = (cost_per_hour or {}).get(kinds.get(j.get("worker_id") or "", ""))
                use = _job_usage(store, j) if j["status"] == "done" else None
                c = _cost(use["gpu_s"], rate) if use else None
                if c is not None:
                    costs.append(c)
            if len(costs) < r.min_jobs:
                continue
            mean = sum(costs) / len(costs)
            if mean > r.max_cost:
                alerts.append(
                    Alert(
                        r.id,
                        r.severity,
                        f"mean cost per job {mean:.4g} exceeds {r.max_cost:.4g}",
                        mean,
                        r.max_cost,
                    )
                )
        else:  # p95_regression
            cur_t = [d for j in recent if j["status"] == "done" and (d := _duration(j)) is not None]
            ref_t = [d for j in base if j["status"] == "done" and (d := _duration(j)) is not None]
            if len(cur_t) < r.min_jobs or not ref_t:
                continue
            cur_p95 = percentiles(cur_t)["p95"] or 0.0
            limit = r.factor * (percentiles(ref_t)["p95"] or 0.0)
            if cur_p95 > limit:
                alerts.append(
                    Alert(
                        r.id,
                        r.severity,
                        f"p95 time {cur_p95:.1f}s exceeds {limit:.1f}s",
                        cur_p95,
                        limit,
                    )
                )
    return alerts


# ---------------------------------------------------------------- report


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(c))}</td>" for c in r) + "</tr>" for r in rows
    )
    return f"<table><tr>{th}</tr>{body}</table>"


def render_html(
    metrics: Mapping[str, Any],
    alerts: Sequence[Alert],
    drift: Mapping[str, Any] | None = None,
) -> str:
    m = metrics
    parts = ["<h1>CountOnce admin</h1>"]
    if alerts:
        parts.append(
            "<h2>Alerts</h2>"
            + _table(
                ["rule", "severity", "message"], [(a.rule, a.severity, a.message) for a in alerts]
            )
        )
    else:
        parts.append("<h2>Alerts</h2><p>none firing</p>")
    summary = [
        ("jobs", m["n_jobs"]),
        ("success rate", m["success_rate"]),
        ("dead-letter rate", m["dead_letter_rate"]),
        ("queue depth", m["queue_depth"]),
        ("oldest queued age (s)", m["oldest_queued_age_s"]),
        ("peak VRAM (MB)", m["peak_vram_mb"]),
        ("recon cache hit rate", m["recon_cache_hit_rate"]),
    ]
    if m["feedback"]:
        summary.append(("feedback accuracy", m["feedback"]["accuracy"]))
    parts.append("<h2>Summary</h2>" + _table(["metric", "value"], summary))
    parts.append("<h2>Jobs per day</h2>" + _table(["day", "jobs"], list(m["jobs_per_day"].items())))
    parts.append(
        "<h2>Dead letters by error code</h2>"
        + _table(["code", "jobs"], list(m["dead_letter_by_code"].items()))
    )
    times = {"total": m["time_s"]["total"], **m["stage_time_s"]}
    parts.append(
        "<h2>Time (s)</h2>"
        + _table(
            ["stage", "p50", "p90", "p95"],
            [(k, v["p50"], v["p90"], v["p95"]) for k, v in times.items()],
        )
    )
    cost = m.get("cost") or {"by_worker_kind": {}}
    todo = "TODO: set rate"
    parts.append(
        "<h2>Cost per worker kind</h2>"
        + _table(
            [
                "worker kind",
                "jobs",
                "GPU-s total",
                "GPU-s/job p50",
                "GPU-s/job p95",
                "cost/job",
                "cost/1,000 frames",
            ],
            [
                (
                    k,
                    v["n_jobs"],
                    v["gpu_seconds_total"],
                    v["gpu_seconds_per_job"]["p50"],
                    v["gpu_seconds_per_job"]["p95"],
                    todo if v["cost_per_hour"] is None else v["cost_per_job_mean"],
                    todo if v["cost_per_hour"] is None else v["cost_per_1000_frames"],
                )
                for k, v in cost["by_worker_kind"].items()
            ],
        )
    )
    parts.append(
        "<h2>Stage time by worker kind (s)</h2>"
        + _table(
            ["worker kind", "stage", "p50", "p95"],
            [
                (k, s, t["p50"], t["p95"])
                for k, v in cost["by_worker_kind"].items()
                for s, t in v["stage_time_s"].items()
            ],
        )
    )
    parts.append(
        "<h2>OOM fallback rungs</h2>"
        + _table(["rung", "jobs"], list(m["oom_rung_histogram"].items()))
    )
    if drift is not None:
        parts.append(
            f"<h2>Input drift (PSI): {html.escape(str(drift['status']))}</h2>"
            + _table(
                ["feature", "psi", "level"],
                [(f, v["psi"], v["level"]) for f, v in drift["features"].items()],
            )
        )
    style = (
        "body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:4px 10px;text-align:left}"
    )
    return (
        f"<!doctype html><meta charset='utf-8'><title>CountOnce admin</title>"
        f"<style>{style}</style>" + "".join(parts)
    )


def alerts_to_dicts(alerts: Sequence[Alert]) -> list[dict[str, Any]]:
    return [asdict(a) for a in alerts]
