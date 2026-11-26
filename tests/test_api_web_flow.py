"""Backend pieces the signed-in web flow relies on: sign-in key exchange and corrections."""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from countonce.api.app import create_app  # noqa: E402
from countonce.jobs.store import SqliteJobStore  # noqa: E402
from countonce.jobs.uploads import UploadService  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "schemas" / "v1.1" / "result.json"


def _client(tmp_path: Path, secret: str | None = "s3cret") -> tuple[TestClient, SqliteJobStore]:
    store = SqliteJobStore(tmp_path / "jobs.db", output_root=tmp_path / "out")
    app = create_app(store, UploadService(store, tmp_path / "up"), auth_secret=secret)
    return TestClient(app), store


def test_exchange_mints_a_working_key(tmp_path: Path) -> None:
    c, store = _client(tmp_path)
    body = {"provider": "google", "subject": "123"}
    assert c.post("/auth/exchange", json=body).status_code == 401
    bad = c.post("/auth/exchange", json=body, headers={"X-Auth-Secret": "nope"})
    assert bad.status_code == 401
    ok = c.post("/auth/exchange", json=body, headers={"X-Auth-Secret": "s3cret"})
    assert ok.status_code == 200 and ok.json()["user_id"] == "google:123"
    me = c.get("/me", headers={"Authorization": f"Bearer {ok.json()['api_key']}"})
    assert me.json() == {"user_id": "google:123"}
    assert store.api_keys("google:123")


def test_exchange_disabled_without_secret(tmp_path: Path) -> None:
    c, _ = _client(tmp_path, secret=None)
    r = c.post(
        "/auth/exchange", json={"provider": "g", "subject": "1"}, headers={"X-Auth-Secret": "x"}
    )
    assert r.status_code == 404


def test_corrections_replay_and_validation(tmp_path: Path) -> None:
    c, store = _client(tmp_path)
    _, key = store.create_api_key("alice")
    h = {"Authorization": f"Bearer {key}"}
    raw = json.loads(FIXTURE.read_text())
    job_id, _ = store.create_draft("alice", "apples")
    store.record_file(job_id, "input", "k", 1, "0" * 64)
    store.queue_draft(job_id)
    store.register_worker("w", name="fake")
    store.claim("w")
    store.complete(job_id, "w", {"count": raw["count"], "result_json": json.dumps(raw)})
    det = raw["instances"][0]["members"][0]

    url = f"/jobs/{job_id}/corrections"
    assert c.get(url, headers=h).json()["corrected_count"] is None
    r = c.post(url, headers=h, json={"action": "reject", "detection_ids": [det]})
    assert r.status_code == 200, r.text
    assert r.json()["idempotent_replay"] is False
    again = c.post(url, headers=h, json={"action": "reject", "detection_ids": [det]})
    assert again.json()["idempotent_replay"] is True
    assert c.get(url, headers=h).json()["corrections"][0]["action"] == "reject"
    bad = c.post(url, headers=h, json={"action": "reject", "detection_ids": ["nope"]})
    assert bad.status_code == 422
    bad = c.post(url, headers=h, json={"action": "explode", "detection_ids": [det]})
    assert bad.status_code == 422
    other = SqliteJobStore(tmp_path / "jobs.db")
    _, k2 = other.create_api_key("bob")
    nope = c.post(
        url,
        headers={"Authorization": f"Bearer {k2}"},
        json={"action": "reject", "detection_ids": [det]},
    )
    assert nope.status_code == 404
