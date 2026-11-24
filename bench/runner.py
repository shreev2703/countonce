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
    # Appended later (append-only schema): older CSVs lack these and are upgraded on resume.
    "peak_vram_mb", "timings_json", "assoc_f1", "run_dir",
]  # fmt: skip
_V1_COLUMN_COUNT = 18  # columns of the original schema; any longer prefix of COLUMNS is valid


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


class SkipRun(Exception):  # noqa: N818 - control flow, not an error
    """Raised by an executor when a run cannot happen for a known, non-failure reason
    (e.g. the v3 checkpoint is absent). Recorded with ``status`` = ``status`` and not retried
    as an error; skipped rows are re-attempted on resume."""

    def __init__(self, status: str, reason: str = ""):
        super().__init__(reason or status)
        self.status = status


@dataclass(frozen=True)
class RunOutput:
    pred: float
    gt: float | None = None
    low: float | None = None
    high: float | None = None
    runtime_s: float | None = None
    peak_vram_mb: float | None = None
    timings: dict[str, float] | None = None  # per-stage seconds (ingest, detect, recon, ...)
    assoc_f1: float | None = None  # pairwise same-object F1; only for scenes with identity labels
    run_dir: str | None = None  # where the run's export (overlays, result.json) was written


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
        names = reader.fieldnames or []
        if not _is_known_header(names):
            raise ValueError(f"{p} has unexpected columns {reader.fieldnames}")
        # Old files predate the appended columns; they read back as empty strings.
        return [{c: (r.get(c) or "") for c in COLUMNS} for r in reader]


def _is_known_header(names: list[str]) -> bool:
    """Current schema or any older prefix of it (append-only evolution)."""
    return len(names) >= _V1_COLUMN_COUNT and names == COLUMNS[: len(names)]


def _upgrade_header(p: Path) -> None:
    """Rewrite an old-schema CSV in place with the current header (atomic replace)."""
    with p.open(newline="") as f:
        names = csv.DictReader(f).fieldnames or []
    if names == COLUMNS:
        return
    rows = read_rows(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(p)


def run_jobs(
    jobs: Iterable[Job],
    fn: Callable[[Job], RunOutput],
    path: str | Path = DEFAULT_CSV,
    run_config: dict[str, Any] | None = None,
) -> tuple[int, int, int]:
    """Run ``fn`` for each job not already completed, appending a row (flushed) after each.

    Exceptions from ``fn`` are recorded as ``status=error`` rows and the loop continues;
    KeyboardInterrupt propagates. ``SkipRun`` rows are recorded with their own status and
    counted as skipped. Returns (ran, skipped, errored).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(p)
    read_rows(p)  # validates header of an existing file
    if p.exists() and p.stat().st_size > 0:
        _upgrade_header(p)
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
            status = "ok"
            try:
                out = fn(job)
            except SkipRun as e:
                status, err = e.status, str(e)
                skipped += 1
            except Exception as e:  # noqa: BLE001 - recorded, run continues
                err = f"{type(e).__name__}: {e}"
                status = "error"
                errored += 1
            wall = time.perf_counter() - t0
            row = {
                "run_key": key,
                "status": status,
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
                "peak_vram_mb": _fmt(out.peak_vram_mb) if out else "",
                "timings_json": json.dumps(out.timings, sort_keys=True)
                if out and out.timings
                else "",
                "assoc_f1": _fmt(out.assoc_f1) if out else "",
                "run_dir": _fmt(out.run_dir) if out else "",
            }
            w.writerow(row)
            f.flush()
            if out is not None:
                ran += 1
    return ran, skipped, errored
