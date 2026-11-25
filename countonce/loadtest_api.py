"""HTTP API load test with fake workers (PLAN 16B.2: load test the API with a fake worker).

Starts the real FastAPI app with uvicorn in-process (background thread, free localhost port)
over a temporary sqlite store and ``LocalBlobStore``, plus N fake worker processes reusing
``countonce.loadtest._fake_worker``. M concurrent ``httpx`` clients each run create -> chunked
upload -> submit -> poll or SSE-follow -> fetch the result, with idempotency keys and deliberate
duplicate retries (the duplicate create/submit pair is sent concurrently). 429s honour
``Retry-After`` (capped by ``retry_after_cap_s``) and are retried.

The clients share the process (and the GIL) with the server, so absolute latency is pessimistic
for the server; use it for relative comparison. Nothing here is a result until a run appends a
row to ``results/api_load.csv``. No k6/Locust: stdlib + httpx + uvicorn only.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import random
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml

from countonce.loadtest import LoadTestConfig, _fake_worker, _pct, _read_events
from countonce.provenance import git_commit, git_is_dirty

CSV_COLUMNS = [
    "timestamp",
    "run_id",
    "git_sha",
    "git_dirty",
    "host",
    "platform",
    "cpu_count",
    "python",
    "n_clients",
    "n_users",
    "n_workers",
    "jobs_per_client",
    "follow",
    "duplicate_rate",
    "request_rate_limit",
    "request_rate_window_s",
    "submit_rate_limit",
    "route",
    "requests",
    "throughput_rps",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "status_counts",
    "error_rate",
    "n_429",
    "wall_s",
    "jobs_expected",
    "jobs_in_db",
    "jobs_done",
    "jobs_lost",
    "dup_pairs",
    "dup_replays",
    "double_queued",
    "client_failures",
    "reconciled",
]

ROUTES = (
    "POST /jobs",
    "POST /uploads",
    "PUT /uploads/{id}/chunks/{n}",
    "POST /uploads/{id}/finalize",
    "POST /jobs/{id}/submit",
    "GET /jobs/{id}",
    "GET /jobs/{id}/events (time to first event)",
    "GET /jobs/{id}/result",
)


@dataclass
class ApiLoadConfig:
    n_clients: int = 16
    jobs_per_client: int = 5
    n_users: int = 8  # clients are spread round-robin over users, so users share rate limits
    n_workers: int = 4
    stage_latency_s: dict[str, float] = field(
        default_factory=lambda: {"ingest": 0.02, "reconstruct": 0.1, "detect": 0.05}
    )
    follow: str = "mixed"  # poll | sse | mixed (alternate per job)
    poll_s: float = 0.25
    duplicate_rate: float = 0.3  # share of jobs whose create and submit are each sent twice at once
    request_rate_limit: int | None = 300  # per user, all authenticated requests (None = off)
    request_rate_window_s: float = 60.0
    submit_rate_limit: int | None = 30  # per user, job submits (None = off)
    submit_rate_window_s: float = 60.0
    retry_after_cap_s: float = 1.0
    max_429_retries: int = 60
    upload_bytes: int = 12_288  # rough size of the random PNG each job uploads
    chunk_bytes: int = 4096
    sse_poll_s: float = 0.1  # server-side SSE poll interval
    timeout_s: float = 180.0  # whole run
    seed: int = 0
    work_dir: str = "runs/loadtest_api"
    results_csv: str = "results/api_load.csv"
    summary_json: str = "results/api_load_summary.json"

    @classmethod
    def from_yaml(cls, path: Path | str) -> ApiLoadConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        bad = sorted(set(raw) - {f.name for f in fields(cls)})
        if bad:
            raise ValueError(f"unknown loadtest_api config keys: {bad}")
        cfg = cls(**raw)
        if cfg.follow not in ("poll", "sse", "mixed"):
            raise ValueError(f"follow must be poll|sse|mixed, got {cfg.follow!r}")
        if cfg.n_clients < 1 or cfg.jobs_per_client < 1 or cfg.n_users < 1:
            raise ValueError("n_clients, jobs_per_client and n_users must be >= 1")
        return cfg


# -- server ---------------------------------------------------------------------------------
class _Server:
    """uvicorn in a background thread on a pre-bound free localhost port."""

    def __init__(self, app: Any) -> None:
        import uvicorn

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = int(self.sock.getsockname()[1])
        self.server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", access_log=False, lifespan="off")
        )
        self.thread = threading.Thread(
            target=lambda: self.server.run(sockets=[self.sock]), daemon=True
        )

    def __enter__(self) -> _Server:
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15)
        self.sock.close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# -- client ---------------------------------------------------------------------------------
def _png(seed: int, approx_bytes: int) -> bytes:
    """A valid, incompressible-ish PNG (random noise) of roughly ``approx_bytes``."""
    from PIL import Image

    side = max(8, int((approx_bytes / 3) ** 0.5))
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


class _Stats:
    def __init__(self) -> None:
        self.lat: dict[str, list[float]] = {r: [] for r in ROUTES}
        self.status: dict[str, dict[int, int]] = {r: {} for r in ROUTES}
        self.failures: list[str] = []
        self.dup_pairs = 0
        self.dup_replays = 0
        self.nonreplay_submits: dict[str, int] = {}
        self.final: dict[str, str] = {}  # job id -> terminal status seen by the client

    def rec(self, route: str, status: int, secs: float) -> None:
        self.lat[route].append(secs)
        self.status[route][status] = self.status[route].get(status, 0) + 1


async def _req(
    http: httpx.AsyncClient,
    stats: _Stats,
    cfg: ApiLoadConfig,
    route: str,
    method: str,
    url: str,
    **kw: Any,
) -> httpx.Response:
    """One logical request: every attempt is recorded; 429 sleeps then retries."""
    for _ in range(cfg.max_429_retries + 1):
        t0 = time.perf_counter()
        r = await http.request(method, url, **kw)
        stats.rec(route, r.status_code, time.perf_counter() - t0)
        if r.status_code != 429:
            return r
        wait = float(r.headers.get("Retry-After", "1"))
        await asyncio.sleep(min(wait, cfg.retry_after_cap_s))
    return r


async def _follow_sse(http: httpx.AsyncClient, stats: _Stats, cfg: ApiLoadConfig, jid: str) -> str:
    route = "GET /jobs/{id}/events (time to first event)"
    for _ in range(cfg.max_429_retries + 1):
        t0 = time.perf_counter()
        async with http.stream("GET", f"/jobs/{jid}/events", timeout=cfg.timeout_s) as r:
            if r.status_code == 429:
                stats.rec(route, 429, time.perf_counter() - t0)
                await asyncio.sleep(
                    min(float(r.headers.get("Retry-After", "1")), cfg.retry_after_cap_s)
                )
                continue
            first = True
            event = ""
            async for line in r.aiter_lines():
                if first and line.startswith("event:"):
                    stats.rec(route, r.status_code, time.perf_counter() - t0)
                    first = False
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event == "done":
                    return str(json.loads(line.split(":", 1)[1])["status"])
            if first:
                stats.rec(route, r.status_code, time.perf_counter() - t0)
            return "stream_ended"
    return "rate_limited"


async def _follow_poll(http: httpx.AsyncClient, stats: _Stats, cfg: ApiLoadConfig, jid: str) -> str:
    while True:
        r = await _req(http, stats, cfg, "GET /jobs/{id}", "GET", f"/jobs/{jid}")
        if r.status_code != 200:
            return f"http_{r.status_code}"
        st = str(r.json()["status"])
        if st in ("done", "failed", "dead_letter", "deleted"):
            return st
        await asyncio.sleep(cfg.poll_s)


async def _upload(
    http: httpx.AsyncClient, stats: _Stats, cfg: ApiLoadConfig, jid: str, data: bytes
) -> bool:
    body = {
        "filename": "a.png",
        "total_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "chunk_size": cfg.chunk_bytes,
    }
    r = await _req(http, stats, cfg, "POST /uploads", "POST", "/uploads", json=body)
    if r.status_code != 201:
        stats.failures.append(f"create_upload {r.status_code} {r.text[:120]}")
        return False
    sid, n = r.json()["session_id"], r.json()["n_chunks"]
    for i in range(n):
        piece = data[i * cfg.chunk_bytes : (i + 1) * cfg.chunk_bytes]
        r = await _req(
            http, stats, cfg, "PUT /uploads/{id}/chunks/{n}", "PUT", f"/uploads/{sid}/chunks/{i}",
            content=piece, headers={"X-Chunk-SHA256": hashlib.sha256(piece).hexdigest()},
        )  # fmt: skip
        if r.status_code != 200:
            stats.failures.append(f"chunk {r.status_code} {r.text[:120]}")
            return False
    r = await _req(
        http, stats, cfg, "POST /uploads/{id}/finalize", "POST", f"/uploads/{sid}/finalize",
        json={"job_id": jid},
    )  # fmt: skip
    if r.status_code != 200:
        stats.failures.append(f"finalize {r.status_code} {r.text[:120]}")
        return False
    return True


async def _one_job(
    http: httpx.AsyncClient, stats: _Stats, cfg: ApiLoadConfig, run: str, c: int, k: int
) -> None:
    rng = random.Random(f"{cfg.seed}:{c}:{k}")
    dup = rng.random() < cfg.duplicate_rate
    ckey = f"{run}-{c}-{k}-create"
    skey = f"{run}-{c}-{k}-submit"
    n_send = 2 if dup else 1

    rs = await asyncio.gather(*[
        _req(http, stats, cfg, "POST /jobs", "POST", "/jobs",
             json={"prompt": "apples"}, headers={"Idempotency-Key": ckey})
        for _ in range(n_send)
    ])  # fmt: skip
    if any(r.status_code not in (200, 201) for r in rs):
        stats.failures.append(f"create {[r.status_code for r in rs]}")
        return
    ids = {r.json()["id"] for r in rs}
    if len(ids) != 1:
        stats.failures.append(f"duplicate create made {len(ids)} jobs")
        return
    jid = ids.pop()
    if dup:
        stats.dup_pairs += 1
        stats.dup_replays += sum(bool(r.json()["idempotent_replay"]) for r in rs)

    if not await _upload(http, stats, cfg, jid, _png(c * 100_003 + k, cfg.upload_bytes)):
        return

    rs = await asyncio.gather(*[
        _req(http, stats, cfg, "POST /jobs/{id}/submit", "POST", f"/jobs/{jid}/submit",
             headers={"Idempotency-Key": skey})
        for _ in range(n_send)
    ])  # fmt: skip
    if any(r.status_code != 200 for r in rs):
        stats.failures.append(f"submit {[r.status_code for r in rs]} {rs[0].text[:120]}")
        return
    fresh = sum(not r.json()["idempotent_replay"] for r in rs)
    stats.nonreplay_submits[jid] = fresh
    if dup:
        stats.dup_replays += n_send - fresh

    sse = cfg.follow == "sse" or (cfg.follow == "mixed" and (c + k) % 2 == 1)
    final = await (_follow_sse if sse else _follow_poll)(http, stats, cfg, jid)
    stats.final[jid] = final
    if final != "done":
        stats.failures.append(f"job {jid} ended {final}")
        return
    r = await _req(http, stats, cfg, "GET /jobs/{id}/result", "GET", f"/jobs/{jid}/result")
    if r.status_code != 200 or r.json()["job"]["status"] != "done":
        stats.failures.append(f"result {r.status_code} {r.text[:120]}")


async def _run_clients(
    cfg: ApiLoadConfig, url: str, keys: list[str], run: str, stats: _Stats
) -> None:
    limits = httpx.Limits(max_connections=cfg.n_clients * 2 + 4)

    async def client(c: int) -> None:
        h = {"Authorization": f"Bearer {keys[c % cfg.n_users]}"}
        async with httpx.AsyncClient(base_url=url, headers=h, timeout=60.0, limits=limits) as http:
            for k in range(cfg.jobs_per_client):
                await _one_job(http, stats, cfg, run, c, k)

    await asyncio.wait_for(
        asyncio.gather(*[client(c) for c in range(cfg.n_clients)]), timeout=cfg.timeout_s
    )


# -- orchestration --------------------------------------------------------------------------
def _route_rows(stats: _Stats, wall: float) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    all_lat: list[float] = []
    all_status: dict[int, int] = {}
    for route in ROUTES:
        lat = stats.lat[route]
        all_lat += lat
        for s, n in stats.status[route].items():
            all_status[s] = all_status.get(s, 0) + n
        rows[route] = _agg(lat, stats.status[route], wall)
    rows["ALL"] = _agg(all_lat, all_status, wall)
    return rows


def _agg(lat: list[float], status: dict[int, int], wall: float) -> dict[str, Any]:
    n = sum(status.values())
    ms = [x * 1000 for x in lat]
    bad = sum(v for s, v in status.items() if s >= 400)
    return {
        "requests": n,
        "throughput_rps": n / max(wall, 1e-9),
        "p50_ms": _pct(ms, 50),
        "p95_ms": _pct(ms, 95),
        "p99_ms": _pct(ms, 99),
        "status_counts": json.dumps({str(s): v for s, v in sorted(status.items())}),
        "error_rate": bad / n if n else None,
        "n_429": status.get(429, 0),
    }


def host_info() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def run_api_load(cfg: ApiLoadConfig) -> dict[str, Any]:
    """One run in a fresh ``work_dir/<run_id>``; returns the summary (also written to CSV/JSON)."""
    from countonce.api.app import create_app
    from countonce.jobs.store import SqliteJobStore
    from countonce.jobs.uploads import UploadService
    from countonce.storage import LocalBlobStore

    run = uuid.uuid4().hex[:8]
    root = Path(cfg.work_dir) / run
    events_dir = root / "events"
    events_dir.mkdir(parents=True)
    db = root / "jobs.db"
    blobs = LocalBlobStore(root / "blobs")
    store = SqliteJobStore(
        db,
        output_root=root / "outputs",
        blobs=blobs,
        rate_limit=cfg.submit_rate_limit,
        rate_window_s=cfg.submit_rate_window_s,
    )
    keys = [store.create_api_key(f"user{u}")[1] for u in range(cfg.n_users)]
    app = create_app(
        store,
        UploadService(store, root / "uploads", blobs=blobs),
        request_rate_limit=cfg.request_rate_limit,
        request_rate_window_s=cfg.request_rate_window_s,
        sse_poll_s=cfg.sse_poll_s,
    )

    wcfg = LoadTestConfig(
        stage_latency_s=dict(cfg.stage_latency_s),
        fail_rate=0.0,
        crash_rate=0.0,
        seed=cfg.seed,
    )
    ctx = mp.get_context("spawn")
    lock_counter, crash_counter, stop = ctx.Value("i", 0), ctx.Value("i", 0), ctx.Event()
    workers = [
        ctx.Process(
            target=_fake_worker,
            args=(wcfg, str(db), f"fw{i}", str(events_dir), lock_counter, crash_counter, stop),
        )
        for i in range(cfg.n_workers)
    ]
    stats = _Stats()
    for w in workers:
        w.start()
    timed_out = False
    t0 = time.perf_counter()
    try:
        with _Server(app) as srv:
            try:
                asyncio.run(_run_clients(cfg, srv.url, keys, run, stats))
            except TimeoutError:
                timed_out = True
                stats.failures.append(f"run timed out after {cfg.timeout_s}s")
    finally:
        wall = time.perf_counter() - t0
        stop.set()
        for w in workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()

    expected = cfg.n_clients * cfg.jobs_per_client
    rows = store._query("SELECT id, status, attempts FROM jobs")
    done = sum(r["status"] == "done" for r in rows)
    completes = [e["job"] for e in _read_events(events_dir) if e["kind"] == "complete"]
    dup_completes = len(completes) - len(set(completes))
    client_double = sum(max(n - 1, 0) for n in stats.nonreplay_submits.values())
    over_attempted = sum(int(r["attempts"]) > 1 for r in rows)  # a re-queue would show here
    double_queued = client_double + max(len(rows) - expected, 0) + dup_completes + over_attempted
    lost = expected - done
    reconciled = bool(
        not timed_out
        and not stats.failures
        and len(rows) == expected
        and lost == 0
        and double_queued == 0
    )
    per_route = _route_rows(stats, wall)
    summary: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": run,
        "git_sha": (git_commit() or "")[:12],
        "git_dirty": git_is_dirty(),
        **host_info(),
        "n_clients": cfg.n_clients,
        "n_users": cfg.n_users,
        "n_workers": cfg.n_workers,
        "jobs_per_client": cfg.jobs_per_client,
        "follow": cfg.follow,
        "duplicate_rate": cfg.duplicate_rate,
        "request_rate_limit": cfg.request_rate_limit,
        "request_rate_window_s": cfg.request_rate_window_s,
        "submit_rate_limit": cfg.submit_rate_limit,
        "wall_s": wall,
        "jobs_expected": expected,
        "jobs_in_db": len(rows),
        "jobs_done": done,
        "jobs_lost": lost,
        "dup_pairs": stats.dup_pairs,
        "dup_replays": stats.dup_replays,
        "double_queued": double_queued,
        "double_queued_parts": {
            "client_double_fresh_submit": client_double,
            "extra_job_rows": max(len(rows) - expected, 0),
            "duplicate_completions": dup_completes,
            "jobs_attempted_more_than_once": over_attempted,
        },
        "client_failures": len(stats.failures),
        "reconciled": reconciled,
        "routes": per_route,
        "failure_samples": stats.failures[:10],
        "config": asdict(cfg),
    }
    append_csv(Path(cfg.results_csv), summary)
    out = Path(cfg.summary_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


def append_csv(path: Path, summary: dict[str, Any]) -> None:
    """Append one row per route plus an ``ALL`` row; the header is fixed and checked."""
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
        for route, agg in summary["routes"].items():
            w.writerow({k: {**summary, **agg, "route": route}.get(k) for k in CSV_COLUMNS})


def format_summary(s: dict[str, Any]) -> str:
    def f(x: Any) -> str:
        return "n/a" if x is None else f"{x:.1f}"

    a = s["routes"]["ALL"]
    lines = [
        f"loadtest-api {s['run_id']}: jobs {s['jobs_done']}/{s['jobs_expected']} done, "
        f"lost={s['jobs_lost']} double_queued={s['double_queued']} "
        f"client_failures={s['client_failures']} reconciled={s['reconciled']}",
        f"  {a['requests']} requests in {s['wall_s']:.1f} s = {a['throughput_rps']:.1f} req/s, "
        f"429s={a['n_429']}, status {a['status_counts']}",
        f"  duplicate pairs {s['dup_pairs']}, replays seen {s['dup_replays']}",
    ]
    for route, r in s["routes"].items():
        if route != "ALL" and r["requests"]:
            lines.append(
                f"  {route:<46} n={r['requests']:<5} p50/p95/p99 "
                f"{f(r['p50_ms'])}/{f(r['p95_ms'])}/{f(r['p99_ms'])} ms"
            )
    lines += [f"  failure: {m}" for m in s["failure_samples"][:3]]
    return "\n".join(lines)
