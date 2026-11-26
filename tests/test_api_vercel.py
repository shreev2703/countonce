"""M18: the API entrypoint is fit for a serverless host."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def asgi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("COUNTONCE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("COUNTONCE_PG_URL", raising=False)
    monkeypatch.setenv("COUNTONCE_SSE_MAX_S", "0.2")
    import countonce.api.asgi as mod

    return importlib.reload(mod)


def test_root_redirects_to_docs_without_web_url(asgi, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("COUNTONCE_WEB_URL", raising=False)
    c = TestClient(asgi.build_app(), follow_redirects=False)
    r = c.get("/")
    assert r.status_code == 307 and r.headers["location"] == "/docs"
    assert c.head("/").status_code == 307
    assert c.get("/health").status_code == 200


def test_root_redirects_to_the_site(asgi, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("COUNTONCE_WEB_URL", "https://countonce.example/")
    r = TestClient(asgi.build_app(), follow_redirects=False).get("/")
    assert r.headers["location"] == "https://countonce.example"


def test_sse_stream_ends_at_the_cap_when_no_worker_finishes_the_job(tmp_path: Path) -> None:
    """The stream closes without a "done" event; the web client then falls back to polling."""
    import time

    from countonce.api.app import create_app
    from countonce.jobs.store import SqliteJobStore
    from countonce.jobs.uploads import UploadService

    store = SqliteJobStore(tmp_path / "jobs.db", output_root=tmp_path / "out")
    _, key = store.create_api_key("alice")
    h = {"Authorization": f"Bearer {key}"}
    c = TestClient(
        create_app(store, UploadService(store, tmp_path / "up"), sse_poll_s=0.02, sse_max_s=0.3)
    )
    job = c.post("/jobs", headers=h, json={"prompt": "cup"})
    assert job.status_code == 201, job.text
    jid = job.json()["id"]
    start = time.monotonic()
    names: list[str] = []
    with c.stream("GET", f"/jobs/{jid}/events", headers=h) as r:
        for line in r.iter_lines():
            if line.startswith("event:"):
                names.append(line.split(":", 1)[1].strip())
    assert time.monotonic() - start < 3
    assert names and "done" not in names


def test_open_app_reads_sse_cap_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from countonce.api import app as appmod

    seen: dict[str, float] = {}
    real = appmod.create_app

    def spy(*a, **kw):  # type: ignore[no-untyped-def]
        seen["v"] = kw["sse_max_s"]
        return real(*a, **kw)

    monkeypatch.setattr(appmod, "create_app", spy)
    monkeypatch.setenv("COUNTONCE_SSE_MAX_S", "45")
    appmod.open_app(tmp_path / "jobs.db", tmp_path / "up")
    assert seen["v"] == 45.0


def test_api_import_does_not_load_the_model_stack() -> None:
    code = (
        "import sys, countonce.api.asgi\n"
        "bad = [m for m in ('torch','transformers','pycolmap','cv2') if m in sys.modules]\n"
        "assert not bad, bad"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "COUNTONCE_DATA_DIR": tempfile.mkdtemp()},
    )
    assert r.returncode == 0, r.stderr
