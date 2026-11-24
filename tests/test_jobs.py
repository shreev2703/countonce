"""Job queue: idempotency, error taxonomy routing, backoff, stale requeue, 2-worker load test."""

import json
import sqlite3
import threading
import time
from pathlib import Path

from countonce.cli import main
from countonce.errors import CountOnceError, ErrorCode
from countonce.jobs.store import SqliteJobStore
from countonce.jobs.worker import Worker, classify
from countonce.types import CountResult, Interval, ModelVersions


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


def _ok(job, inputs, progress):
    progress("detect", 0.5)
    return _result(job["prompt"])


def test_idempotency_key_queues_one_job(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "q.db")
    a, created_a = store.submit("u1", "apple", ["a.jpg"], idempotency_key="k")
    b, created_b = store.submit("u1", "apple", ["a.jpg"], idempotency_key="k")
    c, created_c = store.submit("u2", "apple", ["a.jpg"], idempotency_key="k")  # other user
    assert (created_a, created_b, created_c) == (True, False, True)
    assert a == b != c
    assert len(store.list_jobs()) == 2
    assert len(store.job_files(a)) == 1


def test_done_writes_result_row(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "q.db")
    jid, _ = store.submit("u", "apple", ["a.jpg"])
    w = Worker(store, _ok, worker_id="w1")
    assert w.step() and not w.step()
    job = store.get(jid)
    assert job["status"] == "done" and job["attempts"] == 1 and job["progress"] == 1
    res = store.get_result(jid)
    assert (res["count"], res["recon_cache_hit"], res["oom_fallback_rung"]) == (3, 1, 2)
    assert res["peak_vram_mb"] == 1234.5
    assert json.loads(res["result_json"])["count"] == 3


def test_permanent_error_dead_letters_with_zero_retries(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "q.db")
    jid, _ = store.submit("u", "apple", ["a.mp4"])
    calls = []

    def bad(job, inputs, progress):
        calls.append(1)
        raise CountOnceError("codec", ErrorCode.ERR_UNSUPPORTED_CODEC)

    w = Worker(store, bad)
    w.step()
    job = store.get(jid)
    assert job["status"] == "dead_letter" and job["error_code"] == "ERR_UNSUPPORTED_CODEC"
    assert job["attempts"] == 1 and calls == [1]
    assert not w.step() and calls == [1]
    assert store.get_result(jid) is None


def test_retryable_error_backs_off_then_succeeds(tmp_path: Path) -> None:
    clock = Clock()
    store = SqliteJobStore(tmp_path / "q.db", clock=clock)
    jid, _ = store.submit("u", "apple", ["a.jpg"])
    n = {"calls": 0}

    def flaky(job, inputs, progress):
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


def test_retryable_error_dead_letters_after_max_attempts(tmp_path: Path) -> None:
    clock = Clock()
    store = SqliteJobStore(tmp_path / "q.db", clock=clock)
    jid, _ = store.submit("u", "apple", ["a.jpg"])

    def oom(job, inputs, progress):
        raise CountOnceError("oom", ErrorCode.ERR_GPU_OOM)

    w = Worker(store, oom)
    for _ in range(10):
        clock.t += 100
        w.step()
    job = store.get(jid)
    assert job["status"] == "dead_letter" and job["attempts"] == 4


def test_stale_heartbeat_requeues_then_fails_after_two_attempts(tmp_path: Path) -> None:
    clock = Clock()
    store = SqliteJobStore(tmp_path / "q.db", clock=clock)
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


def test_classify() -> None:
    assert classify(FileNotFoundError("x")) is ErrorCode.ERR_INVALID_INPUT
    assert classify(TimeoutError()) is ErrorCode.ERR_WORKER_TIMEOUT
    assert classify(ConnectionError()) is ErrorCode.ERR_STORAGE_TRANSIENT
    assert classify(KeyError("x")) is ErrorCode.ERR_NO_RECON


class WorkerKilled(BaseException):
    """Simulates SIGKILL: not caught by the worker, so the job stays ``running``."""


def test_load_20_jobs_2_workers_one_killed(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "q.db", heartbeat_timeout_s=0.6)
    ids = [store.submit("u", f"p{i}", [f"{i}.jpg"])[0] for i in range(20)]
    runs: dict[str, int] = {}
    lock = threading.Lock()
    killed = threading.Event()

    def runner_for(name: str):
        def run(job, inputs, progress):
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
    conn = sqlite3.connect(tmp_path / "q.db")
    assert conn.execute("SELECT COUNT(DISTINCT job_id), COUNT(*) FROM results").fetchone() == (
        20,
        20,
    )
    assert {j["id"] for j in store.list_jobs("done")} == set(ids)
    attempts = sorted(j["attempts"] for j in store.list_jobs())
    assert attempts.count(2) == 1 and attempts.count(1) == 19  # only the killed job reran


def test_cli_submit_and_worker_once(tmp_path: Path, capsys) -> None:
    db = tmp_path / "q.db"
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    base = ["submit", "--db", str(db), "--prompt", "apple", "--images", str(img)]
    assert main([*base, "--idempotency-key", "k"]) == 0
    assert main([*base, "--idempotency-key", "k"]) == 0
    out = capsys.readouterr().out
    assert "queued" in out and "duplicate" in out
    assert main(["worker", "--db", str(db), "--once"]) == 0  # 1 queued job, bogus image
    store = SqliteJobStore(db)
    (job,) = store.list_jobs()
    assert job["status"] == "dead_letter" and job["error_code"] in {
        "ERR_INVALID_INPUT",
        "ERR_NO_RECON",
        "ERR_CORRUPT_MEDIA",
        "ERR_UNSUPPORTED_MEDIA",
    }
