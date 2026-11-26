import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from countonce.api.app import create_app  # noqa: E402
from countonce.jobs.store import SqliteJobStore  # noqa: E402


def test_status_is_public_and_carries_timings_only(tmp_path):
    clock = [1000.0]
    store = SqliteJobStore(tmp_path / "jobs.db", clock=lambda: clock[0])
    client = TestClient(create_app(store))
    assert client.get("/status").json()["recent"] == []

    job_id, _ = store.submit("alice", "secret prompt", [])
    clock[0] += 5
    store.register_worker("w1", name="my-laptop")
    assert store.claim("w1")
    clock[0] += 7
    assert store.fail(job_id, "w1", "ERR_NO_RECON", "boom")

    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["recent"] == [
        {"status": "dead_letter", "finished_at": 1012.0, "wait_s": 5.0, "run_s": 7.0}
    ]
    for private in ("secret prompt", "alice", "my-laptop", job_id):
        assert private not in r.text
