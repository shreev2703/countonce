"""Worker error classification and the ``submit`` / ``worker`` CLI.

Store behaviour (idempotency, claim, retry, dead-letter, stale requeue, load) lives in
``tests/jobs_contract.py``, which runs it against every ``JobStore``.
"""

from pathlib import Path

from countonce.cli import main
from countonce.errors import ErrorCode
from countonce.jobs.store import SqliteJobStore
from countonce.jobs.worker import classify


def test_classify() -> None:
    assert classify(FileNotFoundError("x")) is ErrorCode.ERR_INVALID_INPUT
    assert classify(TimeoutError()) is ErrorCode.ERR_WORKER_TIMEOUT
    assert classify(ConnectionError()) is ErrorCode.ERR_STORAGE_TRANSIENT
    assert classify(KeyError("x")) is ErrorCode.ERR_NO_RECON


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


def test_store_postgres_without_url_is_a_clean_error(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("COUNTONCE_PG_URL", raising=False)
    assert main(["worker", "--db", str(tmp_path / "q.db"), "--once", "--store", "postgres"]) == 2
    assert "COUNTONCE_PG_URL" in capsys.readouterr().err
    assert main(["serve-api", "--db", str(tmp_path / "q.db"), "--store", "postgres"]) == 2
    assert not (tmp_path / "q.db").exists()


def test_store_flag_rejects_unknown_kind(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(SystemExit):
        main(["worker", "--db", str(tmp_path / "q.db"), "--store", "mysql"])
