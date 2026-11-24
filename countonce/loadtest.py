"""Queue load-test harness: N submitter and M fake-worker processes on one sqlite job store.

PLAN 16B.2 (Phase 5). Stdlib ``multiprocessing`` only; there is no HTTP API yet, so the
store is exercised directly. Fake workers sleep per stage instead of running the pipeline and
inject failures (reported errors) and crashes (``os._exit`` mid-job, so the job is left
``running`` and recovered by ``requeue_stale``). Nothing here is a measured result until a run
appends a row to ``results/load_test.csv``.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import random
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from countonce.errors import ErrorCode
from countonce.jobs.store import SqliteJobStore

CSV_COLUMNS = [
    "timestamp",
    "run_id",
    "seed",
    "n_submitters",
    "n_workers",
    "n_jobs",
    "stage_latency_total_s",
    "fail_rate",
    "crash_rate",
    "submitted",
    "done",
    "dead_letter",
    "failed",
    "unfinished",
    "duplicate_completions",
    "crashes",
    "attempts_total",
    "wall_s",
    "submit_throughput_jps",
    "completed_jps",
    "queue_wait_p50_s",
    "queue_wait_p95_s",
    "e2e_p50_s",
    "e2e_p95_s",
    "error_rate",
    "dead_letter_rate",
    "lock_retries",
    "reconciled",
]


@dataclass
class LoadTestConfig:
    n_jobs: int = 200
    n_submitters: int = 2
    n_workers: int = 4
    stage_latency_s: dict[str, float] = field(
        default_factory=lambda: {"ingest": 0.02, "reconstruct": 0.1, "detect": 0.05}
    )
    latency_jitter: float = 0.2  # +/- fraction of each stage latency
    fail_rate: float = 0.05  # per attempt, reported error
    permanent_fraction: float = 0.3  # of failures, share that are permanent (dead-letter at once)
    crash_rate: float = 0.0  # per attempt, worker dies mid-job
    max_crashes: int = 2  # budget so the run cannot melt down
    submit_rate_per_s: float = 0.0  # per submitter; 0 = unthrottled
    heartbeat_interval_s: float = 0.2
    heartbeat_timeout_s: float = 1.5
    max_stale_attempts: int = 3
    max_attempts: int = 3
    backoff_base_s: float = 0.05
    poll_s: float = 0.02
    timeout_s: float = 120.0
    seed: int = 0
    work_dir: str = "runs/loadtest"
    results_csv: str = "results/load_test.csv"
    summary_json: str = "results/load_test_summary.json"

    @classmethod
    def from_yaml(cls, path: Path | str) -> LoadTestConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        bad = sorted(set(raw) - {f.name for f in fields(cls)})
        if bad:
            raise ValueError(f"unknown loadtest config keys: {bad}")
        return cls(**raw)


class CountingStore(SqliteJobStore):
    """``SqliteJobStore`` whose lock waits are visible: fail fast on SQLITE_BUSY, count, retry."""

    def __init__(self, path: Path, lock_counter: Any, **kw: Any) -> None:
        self._lock_counter = lock_counter
        super().__init__(path, **kw)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            deadline = time.monotonic() + 60.0
            while True:
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc) and "busy" not in str(exc):
                        raise
                    if time.monotonic() > deadline:
                        raise
                    with self._lock_counter.get_lock():
                        self._lock_counter.value += 1
                    time.sleep(random.uniform(0.001, 0.01))
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()


def _store(cfg: LoadTestConfig, db: Path, lock_counter: Any) -> CountingStore:
    return CountingStore(
        db,
        lock_counter,
        heartbeat_timeout_s=cfg.heartbeat_timeout_s,
        max_stale_attempts=cfg.max_stale_attempts,
        max_attempts=cfg.max_attempts,
        backoff_base_s=cfg.backoff_base_s,
    )


def _submitter(idx: int, cfg: LoadTestConfig, db: str, n: int, lock_counter: Any, out: Any) -> None:
    store = _store(cfg, Path(db), lock_counter)
    lat: list[float] = []
    new = 0
    first = time.time()
    for k in range(n):
        if cfg.submit_rate_per_s > 0:
            time.sleep(1.0 / cfg.submit_rate_per_s)
        t0 = time.perf_counter()
        _, created = store.submit(
            f"user{idx}", "fake", [], idempotency_key=f"lt-{idx}-{k}", input_hash=None
        )
        lat.append(time.perf_counter() - t0)
        new += int(created)
    out.put({"idx": idx, "new": new, "first": first, "last": time.time(), "latency": lat})


def _event(path: Path, **rec: Any) -> None:
    with path.open("a") as fh:  # one short append per event; survives os._exit
        fh.write(json.dumps(rec) + "\n")


def _fake_worker(
    cfg: LoadTestConfig,
    db: str,
    wid: str,
    events_dir: str,
    lock_counter: Any,
    crash_counter: Any,
    stop: Any,
) -> None:
    store = _store(cfg, Path(db), lock_counter)
    store.register_worker(wid, name=wid, kind="fake")
    events = Path(events_dir) / f"{wid}.jsonl"
    while not stop.is_set():
        store.requeue_stale()
        job = store.claim(wid)
        if job is None:
            stop.wait(cfg.poll_s)
            continue
        jid = job["id"]
        rng = random.Random(f"{cfg.seed}:{jid}:{job['attempts']}")
        halt = threading.Event()

        def beat(jid: str = jid, halt: threading.Event = halt) -> None:
            while not halt.wait(cfg.heartbeat_interval_s):
                store.heartbeat(jid, wid)

        threading.Thread(target=beat, daemon=True).start()
        crash_at = None
        if cfg.crash_rate > 0 and rng.random() < cfg.crash_rate:
            crash_at = rng.randrange(len(cfg.stage_latency_s) or 1)
        failure: ErrorCode | None = None
        if rng.random() < cfg.fail_rate:
            failure = (
                ErrorCode.ERR_INVALID_INPUT
                if rng.random() < cfg.permanent_fraction
                else ErrorCode.ERR_STORAGE_TRANSIENT
            )
        n_stages = len(cfg.stage_latency_s)
        for i, (stage, lat) in enumerate(cfg.stage_latency_s.items()):
            store.set_progress(jid, wid, stage, i / max(n_stages, 1))
            time.sleep(lat * (1 + rng.uniform(-cfg.latency_jitter, cfg.latency_jitter)))
            if crash_at == i:
                with crash_counter.get_lock():
                    allowed = crash_counter.value < cfg.max_crashes
                    if allowed:
                        crash_counter.value += 1
                if allowed:
                    _event(events, kind="crash", job=jid, worker=wid)
                    os._exit(137)  # no cleanup: the job stays running until heartbeats lapse
        if failure is not None:
            outcome = store.fail(jid, wid, failure, "injected failure")
            _event(events, kind="fail", job=jid, code=failure.value, outcome=outcome.status)
        elif store.complete(jid, wid, {"count": 1}):
            _event(events, kind="complete", job=jid, worker=wid)
        halt.set()


def _pct(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def _read_events(events_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(events_dir.glob("*.jsonl")):
        out += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return out


def run_load_test(cfg: LoadTestConfig) -> dict[str, Any]:
    """Run one load test in a fresh ``work_dir/<run_id>``; return the summary dict."""
    run_id = uuid.uuid4().hex[:8]
    root = Path(cfg.work_dir) / run_id
    events_dir = root / "events"
    events_dir.mkdir(parents=True)
    db = root / "jobs.db"
    ctx = mp.get_context("spawn")
    lock_counter = ctx.Value("i", 0)
    crash_counter = ctx.Value("i", 0)
    stop = ctx.Event()
    results: Any = ctx.Queue()
    store = _store(cfg, db, lock_counter)

    def spawn_worker(n: int) -> Any:
        p = ctx.Process(
            target=_fake_worker,
            args=(cfg, str(db), f"fw{n}", str(events_dir), lock_counter, crash_counter, stop),
        )
        p.start()
        return p

    t_start = time.time()
    workers = [spawn_worker(i) for i in range(cfg.n_workers)]
    next_id = cfg.n_workers
    per = [
        cfg.n_jobs // cfg.n_submitters + (i < cfg.n_jobs % cfg.n_submitters)
        for i in range(cfg.n_submitters)
    ]
    subs = [
        ctx.Process(target=_submitter, args=(i, cfg, str(db), per[i], lock_counter, results))
        for i in range(cfg.n_submitters)
    ]
    for p in subs:
        p.start()
    reports = [results.get(timeout=cfg.timeout_s) for _ in subs]
    for p in subs:
        p.join()
    submitted = sum(r["new"] for r in reports)

    deadline = time.time() + cfg.timeout_s
    while time.time() < deadline:
        c = store.counts()
        if c["done"] + c["dead_letter"] + c["failed"] >= submitted:
            break
        for i, w in enumerate(workers):  # replace crashed workers so the pool stays at M
            if not w.is_alive() and w.exitcode not in (0, None):
                workers[i] = spawn_worker(next_id)
                next_id += 1
        time.sleep(0.05)
    t_end = time.time()
    stop.set()
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()

    rows = store._query("SELECT status, attempts, created_at, started_at, finished_at FROM jobs")
    counts = {s: sum(r["status"] == s for r in rows) for s in ("done", "dead_letter", "failed")}
    unfinished = len(rows) - sum(counts.values())
    done_rows = [r for r in rows if r["status"] == "done"]
    waits = [r["started_at"] - r["created_at"] for r in rows if r["started_at"] is not None]
    e2e = [r["finished_at"] - r["created_at"] for r in done_rows]
    events = _read_events(events_dir)
    completes = [e["job"] for e in events if e["kind"] == "complete"]
    dup = len(completes) - len(set(completes))
    n_errors = sum(e["kind"] in ("fail", "crash") for e in events)
    attempts = sum(int(r["attempts"]) for r in rows)
    first = min(r["first"] for r in reports)
    last = max(r["last"] for r in reports)
    wall = t_end - t_start
    n_sub = max(sum(r["new"] for r in reports), 1)
    summary: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": run_id,
        "seed": cfg.seed,
        "n_submitters": cfg.n_submitters,
        "n_workers": cfg.n_workers,
        "n_jobs": cfg.n_jobs,
        "stage_latency_total_s": round(sum(cfg.stage_latency_s.values()), 6),
        "fail_rate": cfg.fail_rate,
        "crash_rate": cfg.crash_rate,
        "submitted": submitted,
        "done": counts["done"],
        "dead_letter": counts["dead_letter"],
        "failed": counts["failed"],
        "unfinished": unfinished,
        "duplicate_completions": dup,
        "crashes": int(crash_counter.value),
        "attempts_total": attempts,
        "wall_s": wall,
        "submit_throughput_jps": submitted / max(last - first, 1e-9),
        "completed_jps": counts["done"] / max(wall, 1e-9),
        "queue_wait_p50_s": _pct(waits, 50),
        "queue_wait_p95_s": _pct(waits, 95),
        "e2e_p50_s": _pct(e2e, 50),
        "e2e_p95_s": _pct(e2e, 95),
        "error_rate": n_errors / max(attempts, 1),
        "dead_letter_rate": counts["dead_letter"] / n_sub,
        "lock_retries": int(lock_counter.value),
        "reconciled": bool(
            submitted == cfg.n_jobs
            and submitted == counts["done"] + counts["dead_letter"] + counts["failed"]
            and dup == 0
            and len(done_rows) == len(set(completes))
        ),
    }
    summary["submit_latency_p95_s"] = _pct([x for r in reports for x in r["latency"]], 95)
    summary["config"] = asdict(cfg)
    append_csv(Path(cfg.results_csv), summary)
    out = Path(cfg.summary_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


def append_csv(path: Path, summary: dict[str, Any]) -> None:
    """Append one row; the header is fixed (``CSV_COLUMNS``) and checked on an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != CSV_COLUMNS:
            raise ValueError(f"{path} has a different header; move it aside before appending")
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: summary.get(k) for k in CSV_COLUMNS})


def format_summary(s: dict[str, Any]) -> str:
    def f(x: Any) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    return (
        f"loadtest {s['run_id']}: submitted={s['submitted']} done={s['done']} "
        f"dead_letter={s['dead_letter']} failed={s['failed']} unfinished={s['unfinished']} "
        f"crashes={s['crashes']} reconciled={s['reconciled']}\n"
        f"  submit {f(s['submit_throughput_jps'])} jobs/s, completed {f(s['completed_jps'])} "
        f"jobs/s, wall {f(s['wall_s'])} s\n"
        f"  queue wait p50/p95 {f(s['queue_wait_p50_s'])}/{f(s['queue_wait_p95_s'])} s, "
        f"e2e p50/p95 {f(s['e2e_p50_s'])}/{f(s['e2e_p95_s'])} s\n"
        f"  error rate {f(s['error_rate'])}, dead-letter rate {f(s['dead_letter_rate'])}, "
        f"lock retries {s['lock_retries']}"
    )
