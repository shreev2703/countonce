"""JobStore contract suite: one set of tests, run against every store implementation.

``SqliteJobStore`` always; ``PostgresJobStore`` only when ``COUNTONCE_PG_URL`` points at a
scratch database (its tables are truncated before each test). Without the variable the Postgres
cases skip with a reason on the CI skip allow-list (``scripts/check_skips.py``).
Collected via ``python_files`` in ``pyproject.toml``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from countonce.errors import CountOnceError, ErrorCode
from countonce.jobs.store import JobStore, SqliteJobStore
from countonce.jobs.worker import Worker
from countonce.types import CountResult, Interval, ModelVersions

PG_URL = os.environ.get("COUNTONCE_PG_URL")
PG_SKIP = "COUNTONCE_PG_URL not set: Postgres contract tests need a scratch database"
TABLES = (
    "upload_chunks, upload_sessions, rate_buckets, api_keys, share_tokens, feedback, results, "
    "job_files, jobs, workers, users"
)

StoreFactory = Callable[..., Any]


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _result(prompt: str = "p") -> CountResult:
    return CountResult(
        prompt=prompt,
        count=3,
        interval=Interval(2, 4, 0.9, "sensitivity"),
        model_versions=ModelVersions(detector="d", backbone="b", associator="a"),
        input_hash="h",
        recon_cache_hit=True,
        oom_fallback_rung=2,
        peak_vram_mb=1234.5,
    )


def _ok(job: dict[str, Any], inputs: Any, progress: Any) -> CountResult:
    progress("detect", 0.5)
    return _result(job["prompt"])


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def make_store(request: pytest.FixtureRequest, tmp_path: Path) -> StoreFactory:
    kind = request.param
    if kind == "postgres":
        if not PG_URL:
            pytest.skip(PG_SKIP)
        from countonce.jobs.pg_store import PostgresJobStore

        PostgresJobStore(PG_URL, data_dir=tmp_path)._query("SELECT 1")  # migrate once
        import psycopg

        with psycopg.connect(PG_URL.replace("+psycopg", "")) as conn:
            conn.execute(f"TRUNCATE {TABLES} CASCADE")  # type: ignore[arg-type]

        def pg(**kw: Any) -> PostgresJobStore:
            return PostgresJobStore(PG_URL, data_dir=tmp_path, **kw)

        return pg

    def sq(**kw: Any) -> SqliteJobStore:
        return SqliteJobStore(tmp_path / "q.db", **kw)

    return sq


def test_satisfies_protocol(make_store: StoreFactory) -> None:
    store: JobStore = make_store()
    assert store.counts()["queued"] == 0


def test_idempotency_key_queues_one_job(make_store: StoreFactory) -> None:
    store = make_store()
    a, created_a = store.submit("u1", "apple", ["a.jpg"], idempotency_key="k")
    b, created_b = store.submit("u1", "apple", ["a.jpg"], idempotency_key="k")
    c, created_c = store.submit("u2", "apple", ["a.jpg"], idempotency_key="k")  # other user
    assert (created_a, created_b, created_c) == (True, False, True)
    assert a == b != c
    assert len(store.list_jobs()) == 2
    assert len(store.job_files(a)) == 1


def test_no_key_never_dedupes(make_store: StoreFactory) -> None:
    store = make_store()
    ids = {store.submit("u", "apple", ["a.jpg"])[0] for _ in range(3)}
    assert len(ids) == 3


def test_claim_is_fifo_and_exclusive(make_store: StoreFactory) -> None:
    clock = Clock()
    store = make_store(clock=clock)
    ids = []
    for i in range(3):
        ids.append(store.submit("u", f"p{i}", [f"{i}.jpg"])[0])
        clock.t += 1
    got = [store.claim(f"w{i}") for i in range(4)]
    assert [g["id"] for g in got[:3]] == ids and got[3] is None
    assert all(g["status"] == "running" and g["attempts"] == 1 for g in got[:3])


def test_concurrent_claims_never_double_claim(make_store: StoreFactory) -> None:
    store = make_store()
    for i in range(30):
        store.submit("u", f"p{i}", [f"{i}.jpg"])
    claimed: list[str] = []
    lock = threading.Lock()

    def grab(name: str) -> None:
        while (job := store.claim(name)) is not None:
            with lock:
                claimed.append(job["id"])

    threads = [threading.Thread(target=grab, args=(f"w{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(claimed) == 30 and len(set(claimed)) == 30
    assert store.counts()["running"] == 30


def test_register_worker_and_heartbeat_ownership(make_store: StoreFactory) -> None:
    store = make_store()
    store.register_worker("w1", name="box", gpu_name="rtx", vram_mb=8000)
    store.register_worker("w1", name="box2")  # upsert
    (row,) = store._query("SELECT * FROM workers")
    assert row["name"] == "box2" and row["kind"] == "laptop"
    jid, _ = store.submit("u", "apple", ["a.jpg"])
    assert store.claim("w1") is not None
    assert store.heartbeat(jid, "w1") and not store.heartbeat(jid, "other")
    assert store.set_progress(jid, "w1", "detect", 7.0)
    assert store.get(jid)["progress"] == 1.0  # clamped


def test_done_writes_result_row(make_store: StoreFactory) -> None:
    store = make_store()
    jid, _ = store.submit("u", "apple", ["a.jpg"])
    w = Worker(store, _ok, worker_id="w1")
    assert w.step() and not w.step()
    job = store.get(jid)
    assert job["status"] == "done" and job["attempts"] == 1 and job["progress"] == 1
    res = store.get_result(jid)
    assert (res["count"], res["recon_cache_hit"], res["oom_fallback_rung"]) == (3, True, 2)
    assert res["peak_vram_mb"] == 1234.5
    assert json.loads(res["result_json"])["count"] == 3


def test_permanent_error_dead_letters_with_zero_retries(make_store: StoreFactory) -> None:
    store = make_store()
    jid, _ = store.submit("u", "apple", ["a.mp4"])
    calls: list[int] = []

    def bad(job: Any, inputs: Any, progress: Any) -> CountResult:
        calls.append(1)
        raise CountOnceError("codec", ErrorCode.ERR_UNSUPPORTED_CODEC)

    w = Worker(store, bad)
    w.step()
    job = store.get(jid)
    assert job["status"] == "dead_letter" and job["error_code"] == "ERR_UNSUPPORTED_CODEC"
    assert job["attempts"] == 1 and calls == [1]
    assert not w.step() and calls == [1]
    assert store.get_result(jid) is None


def test_retryable_error_backs_off_then_succeeds(make_store: StoreFactory) -> None:
    clock = Clock()
    store = make_store(clock=clock)
    jid, _ = store.submit("u", "apple", ["a.jpg"])
    n = {"calls": 0}

    def flaky(job: Any, inputs: Any, progress: Any) -> CountResult:
        n["calls"] += 1
        if n["calls"] < 3:
            raise CountOnceError("blip", ErrorCode.ERR_STORAGE_TRANSIENT)
        return _result()

    w = Worker(store, flaky)
    assert w.step()
    job = store.get(jid)
    assert job["status"] == "queued" and job["error_code"] == "ERR_STORAGE_TRANSIENT"
    assert job["next_attempt_at"] == clock.t + 2.0
    assert not w.step()  # still backing off
    clock.t += 2.0
    assert w.step()
    assert store.get(jid)["next_attempt_at"] == clock.t + 4.0  # doubled
    assert not w.step()
    clock.t += 4.0
    assert w.step()
    job = store.get(jid)
    assert job["status"] == "done" and job["attempts"] == 3


def test_retryable_error_dead_letters_after_max_attempts(make_store: StoreFactory) -> None:
    clock = Clock()
    store = make_store(clock=clock)
    jid, _ = store.submit("u", "apple", ["a.jpg"])

    def oom(job: Any, inputs: Any, progress: Any) -> CountResult:
        raise CountOnceError("oom", ErrorCode.ERR_GPU_OOM)

    w = Worker(store, oom)
    for _ in range(10):
        clock.t += 100
        w.step()
    job = store.get(jid)
    assert job["status"] == "dead_letter" and job["attempts"] == 4


def test_fail_from_non_owner_is_ignored(make_store: StoreFactory) -> None:
    store = make_store()
    jid, _ = store.submit("u", "apple", ["a.jpg"])
    store.claim("w1")
    assert store.fail(jid, "w2", ErrorCode.ERR_CORRUPT_MEDIA, "x").status == "ignored"
    assert store.get(jid)["status"] == "running"


def test_stale_heartbeat_requeues_then_fails_after_two_attempts(make_store: StoreFactory) -> None:
    clock = Clock()
    store = make_store(clock=clock)
    jid, _ = store.submit("u", "apple", ["a.jpg"])
    assert store.claim("dead1")["attempts"] == 1
    clock.t += 60
    assert store.requeue_stale() == (0, 0)
    assert store.heartbeat(jid, "dead1")
    clock.t += 121
    assert store.requeue_stale() == (1, 0)
    assert store.claim("dead2")["attempts"] == 2
    assert not store.heartbeat(jid, "dead1")  # no longer ours
    assert store.complete(jid, "dead1", {"count": 1}) is False
    clock.t += 121
    assert store.requeue_stale() == (0, 1)
    job = store.get(jid)
    assert job["status"] == "failed" and job["error_code"] == "ERR_WORKER_TIMEOUT"


def test_drafts_are_never_claimed(make_store: StoreFactory) -> None:
    store = make_store()
    jid, _ = store.create_draft("u", "apple", idempotency_key="d")
    assert store.create_draft("u", "apple", idempotency_key="d") == (jid, False)
    assert store.claim("w") is None
    assert store.queue_draft(jid) and not store.queue_draft(jid)
    assert store.claim("w")["id"] == jid


def test_rate_limit_token_bucket(make_store: StoreFactory) -> None:
    from countonce.jobs.store import RateLimitError

    clock = Clock()
    store = make_store(clock=clock, rate_limit=2, rate_window_s=60.0)
    store.submit("u", "a", ["a.jpg"])
    store.submit("u", "a", ["a.jpg"])
    with pytest.raises(RateLimitError):
        store.submit("u", "a", ["a.jpg"])
    store.submit("other", "a", ["a.jpg"])
    clock.t += 30
    store.submit("u", "a", ["a.jpg"])


def test_share_tokens_feedback_and_opt_in(make_store: StoreFactory) -> None:
    store = make_store()
    jid, _ = store.submit("u", "apple", ["a.jpg"])
    tok = store.create_share_token(jid)
    view = store.resolve_share_token(tok)
    assert view is not None and view["job"]["id"] == jid
    assert store.revoke_share_token(tok) and store.resolve_share_token(tok) is None
    store.add_feedback(jid, "u", True, 5, "ok")
    (fb,) = store._query("SELECT * FROM feedback WHERE job_id = ?", (jid,))
    assert fb["correct"] and fb["user_count"] == 5
    store.set_training_opt_in("u", True)
    store.set_training_opt_in("v", False)
    assert store.opted_in_users() == {"u"}


def test_api_key_roundtrip(make_store: StoreFactory) -> None:
    store = make_store()
    key_id, plaintext = store.create_api_key("u")
    assert store.verify_api_key(plaintext) == "u" and store.verify_api_key(plaintext + "x") is None
    assert store.revoke_api_key(key_id) and store.verify_api_key(plaintext) is None


def test_delete_job_revokes_and_hides(make_store: StoreFactory) -> None:
    store = make_store()
    jid, _ = store.submit("u", "apple", ["a.jpg"])
    tok = store.create_share_token(jid)
    assert store.delete_job(jid) and not store.delete_job(jid)
    assert store.get(jid)["status"] == "deleted" and store.resolve_share_token(tok) is None


class WorkerKilled(BaseException):
    """Simulates SIGKILL: not caught by the worker, so the job stays ``running``."""


def test_load_20_jobs_2_workers_one_killed(make_store: StoreFactory) -> None:
    store = make_store(heartbeat_timeout_s=0.6)
    ids = [store.submit("u", f"p{i}", [f"{i}.jpg"])[0] for i in range(20)]
    runs: dict[str, int] = {}
    lock = threading.Lock()
    killed = threading.Event()

    def runner_for(name: str) -> Callable[..., CountResult]:
        def run(job: Any, inputs: Any, progress: Any) -> CountResult:
            progress("detect", 0.5)
            with lock:
                runs[job["id"]] = runs.get(job["id"], 0) + 1
                die = name == "A" and runs[job["id"]] == 1 and not killed.is_set()
                if die:
                    killed.set()
            if die:
                time.sleep(0.05)
                raise WorkerKilled
            time.sleep(0.01)
            return _result(job["prompt"])

        return run

    def drive(name: str) -> None:
        w = Worker(store, runner_for(name), worker_id=name, heartbeat_interval_s=0.1)
        try:
            while True:
                c = store.counts()
                if c["queued"] + c["running"] == 0:
                    return
                if not w.step():
                    time.sleep(0.02)
        except WorkerKilled:
            return

    threads = [threading.Thread(target=drive, args=(n,)) for n in "AB"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert killed.is_set()
    counts = store.counts()
    assert counts["done"] == 20 and sum(counts.values()) == 20
    (row,) = store._query("SELECT COUNT(DISTINCT job_id) AS d, COUNT(*) AS n FROM results")
    assert (row["d"], row["n"]) == (20, 20)
    assert {j["id"] for j in store.list_jobs("done")} == set(ids)
    attempts = sorted(j["attempts"] for j in store.list_jobs())
    assert attempts.count(2) == 1 and attempts.count(1) == 19  # only the killed job reran


@pytest.mark.postgres
def test_cli_worker_and_api_on_postgres(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--store postgres`` end to end: a queued job (bogus image) is drained by the CLI worker."""
    if not PG_URL:
        pytest.skip(PG_SKIP)
    from countonce.cli import main
    from countonce.jobs.pg_store import PostgresJobStore

    store = PostgresJobStore(PG_URL, data_dir=tmp_path)
    import psycopg

    with psycopg.connect(PG_URL.replace("+psycopg", "")) as conn:
        conn.execute(f"TRUNCATE {TABLES} CASCADE")  # type: ignore[arg-type]
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    jid, _ = store.submit("u", "apple", [img])
    db = tmp_path / "q.db"
    assert (
        main(["worker", "--db", str(db), "--store", "postgres", "--pg-url", PG_URL, "--once"]) == 0
    )
    assert "processed 1 job" in capsys.readouterr().out
    assert store.get(jid)["status"] == "dead_letter"  # bogus image, permanent error


TP = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_trace_context_roundtrip(make_store: StoreFactory) -> None:
    store = make_store()
    plain, _ = store.submit("u", "apple", ["a.jpg"])
    traced, _ = store.submit("u", "apple", ["a.jpg"], trace_context=TP)
    draft, _ = store.create_draft("u", "apple")
    assert store.get(draft)["trace_context"] is None
    assert store.queue_draft(draft, TP)
    assert store.get(plain)["trace_context"] is None  # old jobs: nothing stored
    store.register_worker("w1")
    claimed = {j["id"]: j["trace_context"] for j in (store.claim("w1") for _ in range(3)) if j}
    assert claimed == {plain: None, traced: TP, draft: TP}


def test_queue_draft_without_context_keeps_none(make_store: StoreFactory) -> None:
    store = make_store()
    draft, _ = store.create_draft("u", "apple")
    assert store.queue_draft(draft)
    assert store.get(draft)["trace_context"] is None
