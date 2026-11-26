"""M19: idle heartbeat, retention in the worker loop, UI-only options, dead letters, supervisor."""

import json
import signal
import sys
import threading
from pathlib import Path

from countonce.cli import main
from countonce.errors import CountOnceError, ErrorCode
from countonce.jobs.service import supervise, systemd_unit
from countonce.jobs.store import DAY_S, SqliteJobStore
from countonce.jobs.worker import Worker, pipeline_runner


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def _store(tmp_path, clock=None):
    return SqliteJobStore(tmp_path / "j.db", output_root=tmp_path / "out", clock=clock or Clock())


def _job(store, tmp_path, name="a", **kw):
    f = tmp_path / f"{name}.jpg"
    f.write_bytes(b"x")
    return store.submit("u", "apples", [f], **kw)[0]


def test_idle_worker_heartbeat_updates_workers_row(tmp_path):
    clock = Clock()
    store = _store(tmp_path, clock)
    w = Worker(store, lambda *a: None, worker_id="w1")  # type: ignore[arg-type]
    clock.t += 500
    w.housekeeping(now=0.0)
    row = store._query("SELECT last_heartbeat FROM workers WHERE id = 'w1'")[0]
    assert row["last_heartbeat"] == clock.t


def test_retention_runs_in_housekeeping_and_is_throttled(tmp_path):
    clock = Clock()
    store = _store(tmp_path, clock)
    jid = _job(store, tmp_path)
    w = Worker(store, lambda *a: None, worker_id="w1", retention_days=30)  # type: ignore[arg-type]
    store.claim("w1")
    store.complete(jid, "w1", {"count": 1})
    clock.t += 31 * DAY_S
    assert w.housekeeping(now=0.0) == [jid]
    assert w.housekeeping(now=10.0) == []  # throttled to retention_every_s
    assert store.get(jid)["status"] == "deleted"


def test_retention_disabled_by_zero(tmp_path):
    clock = Clock()
    store = _store(tmp_path, clock)
    jid = _job(store, tmp_path)
    w = Worker(store, lambda *a: None, worker_id="w1", retention_days=0)  # type: ignore[arg-type]
    store.claim("w1")
    store.complete(jid, "w1", {"count": 1})
    clock.t += 90 * DAY_S
    assert w.housekeeping(now=0.0) == []
    assert store.get(jid)["status"] == "done"


def test_example_box_option_is_not_passed_to_config(tmp_path, monkeypatch):
    seen = {}

    def fake_run(arg, prompt, cfg):
        seen["cfg"] = cfg
        return "result"

    monkeypatch.setattr("countonce.pipeline.run", fake_run)
    job = {
        "id": "j1",
        "prompt": "apples",
        "options_json": json.dumps({"example_box": {"file": "a.jpg", "x": 1}, "max_frames": 8}),
    }
    out = pipeline_runner(tmp_path)(job, [Path("a.jpg")], lambda *_: None)
    assert out == "result"
    assert seen["cfg"].max_frames == 8


def test_typed_error_dead_letters_then_operator_retry(tmp_path, capsys):
    store = _store(tmp_path)

    def boom(job, inputs, progress):
        raise CountOnceError("bad file", ErrorCode.ERR_UNSUPPORTED_MEDIA)

    w = Worker(store, boom, worker_id="w1")
    jid = _job(store, tmp_path)
    assert w.step()
    row = store.get(jid)
    assert row["status"] == "dead_letter"
    assert row["error_code"] == "ERR_UNSUPPORTED_MEDIA"
    db = str(tmp_path / "j.db")
    assert main(["jobs", "--db", db, "dead"]) == 0
    assert jid in capsys.readouterr().out
    assert main(["jobs", "--db", db, "retry", jid]) == 0
    assert store.get(jid)["status"] == "queued"
    assert main(["jobs", "--db", db, "retry", jid]) == 1  # no longer dead-lettered


def test_run_loop_stops_on_event(tmp_path):
    store = _store(tmp_path)
    w = Worker(store, lambda *a: None, worker_id="w1")  # type: ignore[arg-type]
    stop = threading.Event()
    stop.set()
    assert w.run_loop(stop=stop) == 0


def test_supervise_restarts_then_gives_up(tmp_path):
    sleeps: list[float] = []
    cmd = [sys.executable, "-c", "import sys; sys.exit(3)"]
    code = supervise(cmd, max_restarts=2, sleep=sleeps.append, log=lambda m: None)
    assert code == 3
    assert sleeps == [1.0, 2.0]


def test_supervise_clean_exit_does_not_restart():
    sleeps: list[float] = []
    code = supervise([sys.executable, "-c", "pass"], sleep=sleeps.append, log=lambda m: None)
    assert code == 0 and sleeps == []


def test_supervise_restores_signal_handlers():
    before = signal.getsignal(signal.SIGTERM)
    supervise([sys.executable, "-c", "pass"], log=lambda m: None)
    assert signal.getsignal(signal.SIGTERM) is before


def test_print_unit_has_restart_and_graceful_stop(capsys):
    assert main(["worker", "--db", "jobs.db", "--supervise", "--print-unit"]) == 0
    out = capsys.readouterr().out
    assert "Restart=on-failure" in out and "KillSignal=SIGINT" in out
    assert "--supervise" not in out and "countonce.cli worker" in out
    assert "ExecStart=" in systemd_unit(["--db", "x"], Path("/tmp"))
