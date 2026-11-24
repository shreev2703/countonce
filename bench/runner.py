"""Resumable benchmark runner: one row per run appended to ``results/bench.csv`` (PLAN §9.5)."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from countonce.provenance import run_metadata

DEFAULT_CSV = Path("results/bench.csv")

COLUMNS = [
    "run_key", "status", "scene_id", "split", "method", "n_views", "sample", "ordered",
    "pred", "gt", "low", "high", "runtime_s", "error",
    "git_commit", "git_dirty", "timestamp_utc", "config_json",
]  # fmt: skip


@dataclass(frozen=True)
class Job:
    """One benchmark run. ``config`` must be JSON-serialisable and fully determines the method."""

    scene_id: str
    method: str
    n_views: int | None = None
    sample: str = ""  # e.g. "random0", "even"
    ordered: bool = True
    split: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        payload = json.dumps(
            [self.scene_id, self.method, self.n_views, self.sample, self.ordered, self.config],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class RunOutput:
    pred: float
    gt: float | None = None
    low: float | None = None
    high: float | None = None
    runtime_s: float | None = None


def _fmt(v: object) -> str:
    return "" if v is None else str(v)


def completed_keys(path: str | Path = DEFAULT_CSV) -> set[str]:
    """Keys of runs that finished with status ``ok``; errored runs are retried on resume."""
    return {r["run_key"] for r in read_rows(path) if r["status"] == "ok"}


def read_rows(path: str | Path = DEFAULT_CSV) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{p} has unexpected columns {reader.fieldnames}")
        return list(reader)


def run_jobs(
    jobs: Iterable[Job],
    fn: Callable[[Job], RunOutput],
    path: str | Path = DEFAULT_CSV,
    run_config: dict[str, Any] | None = None,
) -> tuple[int, int, int]:
    """Run ``fn`` for each job not already completed, appending a row (flushed) after each.

    Exceptions from ``fn`` are recorded as ``status=error`` rows and the loop continues;
    KeyboardInterrupt propagates. Returns (ran, skipped, errored).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(p)
    read_rows(p)  # validates header of an existing file
    meta = run_metadata(run_config or {})
    ran = skipped = errored = 0
    write_header = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        for job in jobs:
            key = job.key()
            if key in done:
                skipped += 1
                continue
            t0 = time.perf_counter()
            out: RunOutput | None = None
            err = ""
            try:
                out = fn(job)
            except Exception as e:  # noqa: BLE001 - recorded, run continues
                err = f"{type(e).__name__}: {e}"
                errored += 1
            wall = time.perf_counter() - t0
            row = {
                "run_key": key,
                "status": "ok" if out is not None else "error",
                "scene_id": job.scene_id,
                "split": job.split,
                "method": job.method,
                "n_views": _fmt(job.n_views),
                "sample": job.sample,
                "ordered": int(job.ordered),
                "pred": _fmt(out.pred) if out else "",
                "gt": _fmt(out.gt) if out else "",
                "low": _fmt(out.low) if out else "",
                "high": _fmt(out.high) if out else "",
                "runtime_s": _fmt(out.runtime_s if out and out.runtime_s is not None else wall),
                "error": err.replace("\n", " "),
                "git_commit": _fmt(meta.get("git_commit")),
                "git_dirty": _fmt(meta.get("git_dirty")),
                "timestamp_utc": meta["timestamp_utc"],
                "config_json": json.dumps(job.config, sort_keys=True),
            }
            w.writerow(row)
            f.flush()
            if out is not None:
                ran += 1
    return ran, skipped, errored
