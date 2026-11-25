"""HTTP API with a fake worker: lifecycle, idempotency, SSE, shares, rate limits, dead letter."""

import hashlib
import io
import json
import random
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from countonce.api.app import HTTP_STATUS, create_app  # noqa: E402
from countonce.api.docs import render_api_md  # noqa: E402
from countonce.cli import main  # noqa: E402
from countonce.errors import ErrorCode  # noqa: E402
from countonce.jobs.store import SqliteJobStore  # noqa: E402
from countonce.jobs.uploads import UploadService  # noqa: E402

CHUNK = 1000


def _png() -> bytes:
    rng = random.Random(0)
    img = Image.new("RGB", (64, 64))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(4096)])
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class Env:
    def __init__(self, tmp_path: Path, **store_kw: object) -> None:
        self.store = SqliteJobStore(tmp_path / "jobs.db", output_root=tmp_path / "out", **store_kw)  # type: ignore[arg-type]
        self.uploads = UploadService(self.store, tmp_path / "uploads")
        self.tmp = tmp_path
        _, key = self.store.create_api_key("alice")
        self.h = {"Authorization": f"Bearer {key}"}
        _, key2 = self.store.create_api_key("bob")
        self.h2 = {"X-API-Key": key2}

    def client(self, **kw: object) -> TestClient:
        return TestClient(create_app(self.store, self.uploads, sse_poll_s=0.02, **kw))  # type: ignore[arg-type]

    def upload(self, c: TestClient, job_id: str, h: dict[str, str] | None = None) -> None:
        h = h or self.h
        data = _png()
        r = c.post(
            "/uploads",
            headers=h,
            json={
                "filename": "a.png",
                "total_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "chunk_size": CHUNK,
            },
        )
        assert r.status_code == 201, r.text
        sid = r.json()["session_id"]
        n = r.json()["n_chunks"]
        for i in reversed(range(n)):  # out of order on purpose
            piece = data[i * CHUNK : (i + 1) * CHUNK]
            r = c.put(
                f"/uploads/{sid}/chunks/{i}",
                headers={**h, "X-Chunk-SHA256": hashlib.sha256(piece).hexdigest()},
                content=piece,
            )
            assert r.status_code == 200, r.text
        assert c.get(f"/uploads/{sid}", headers=h).json()["complete"]
        r = c.post(f"/uploads/{sid}/finalize", headers=h, json={"job_id": job_id})
        assert r.status_code == 200, r.text
        assert "path" not in r.json()

    def queued_job(self, c: TestClient, key: str | None = None) -> str:
        hdr = {**self.h, **({"Idempotency-Key": key} if key else {})}
        job_id = c.post("/jobs", headers=hdr, json={"prompt": "apples"}).json()["id"]
        self.upload(c, job_id)
        assert c.post(f"/jobs/{job_id}/submit", headers=hdr).status_code == 200
        return str(job_id)

    def work(self, ok: bool = True, code: ErrorCode = ErrorCode.ERR_CORRUPT_MEDIA) -> str | None:
        """Fake worker: claim one job, report progress, then complete or fail it."""
        self.store.register_worker("w1", name="fake")
        job = self.store.claim("w1")
        if job is None:
            return None
        self.store.set_progress(job["id"], "w1", "detecting", 0.5)
        if ok:
            out = self.tmp / "out" / job["id"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "result.json").write_text("{}")
            self.store.complete(
                job["id"],
                "w1",
                {
                    "count": 7,
                    "interval_low": 6,
                    "interval_high": 8,
                    "result_json": json.dumps({"count": 7}),
                    "warnings_json": "[]",
                },
            )
            with self.store._tx() as conn:
                conn.execute(
                    "INSERT INTO job_files (id, job_id, kind, key, bytes) VALUES (?,?,?,?,?)",
                    ("f1", job["id"], "output", str(out / "result.json"), 2),
                )
        else:
            self.store.fail(job["id"], "w1", code, "bad media")
        return str(job["id"])


@pytest.fixture()
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def test_auth_required_and_health_open(env: Env) -> None:
    c = env.client()
    assert c.get("/jobs").status_code == 401
    assert c.get("/jobs", headers={"Authorization": "Bearer co_x_y"}).status_code == 401
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["processing_paused"] is True and r.json()["workers_online"] == 0
    env.store.register_worker("w1", name="fake")
    assert c.get("/health").json()["workers_online"] == 1


def test_full_lifecycle(env: Env) -> None:
    c = env.client()
    created = c.post("/jobs", headers=env.h, json={"prompt": "apples"})
    assert created.status_code == 201 and created.json()["status"] == "draft"
    job_id = created.json()["id"]
    # no files yet, and a worker can not claim a draft
    assert c.post(f"/jobs/{job_id}/submit", headers=env.h).status_code == 400
    assert env.store.claim("w1") is None
    env.upload(c, job_id)
    assert c.get(f"/jobs/{job_id}/result", headers=env.h).status_code == 409
    sub = c.post(f"/jobs/{job_id}/submit", headers=env.h)
    assert sub.json()["status"] == "queued"
    assert env.work() == job_id
    got = c.get(f"/jobs/{job_id}", headers=env.h).json()
    assert got["status"] == "done" and got["progress"] == 1
    res = c.get(f"/jobs/{job_id}/result", headers=env.h).json()
    assert res["result"]["count"] == 7 and res["result"]["result_json"] == {"count": 7}
    url = res["outputs"][0]["url"]
    assert c.get(url, headers=env.h).json() == {}
    fb = c.post(f"/jobs/{job_id}/feedback", headers=env.h, json={"correct": False, "user_count": 9})
    assert fb.status_code == 201
    assert env.store._query("SELECT user_count FROM feedback")[0]["user_count"] == 9
    assert [j["id"] for j in c.get("/jobs", headers=env.h).json()["jobs"]] == [job_id]
    assert c.get("/jobs?status=queued", headers=env.h).json()["jobs"] == []
    assert c.delete(f"/jobs/{job_id}", headers=env.h).json()["status"] == "deleted"
    assert c.get(f"/jobs/{job_id}", headers=env.h).status_code == 404
    assert c.get("/jobs", headers=env.h).json()["jobs"] == []


def test_other_users_jobs_are_404(env: Env) -> None:
    c = env.client()
    job_id = c.post("/jobs", headers=env.h, json={"prompt": "x"}).json()["id"]
    for method, path in [
        ("GET", f"/jobs/{job_id}"),
        ("POST", f"/jobs/{job_id}/submit"),
        ("DELETE", f"/jobs/{job_id}"),
        ("GET", f"/jobs/{job_id}/result"),
    ]:
        assert c.request(method, path, headers=env.h2).status_code == 404
    assert c.get("/jobs", headers=env.h2).json()["jobs"] == []


def test_same_idempotency_key_queues_one_job(env: Env) -> None:
    c = env.client()
    h = {**env.h, "Idempotency-Key": "uuid-1"}
    a = c.post("/jobs", headers=h, json={"prompt": "apples"})
    b = c.post("/jobs", headers=h, json={"prompt": "apples"})
    assert a.status_code == 201 and b.status_code == 200
    assert a.json()["id"] == b.json()["id"] and b.json()["idempotent_replay"]
    job_id = a.json()["id"]
    env.upload(c, job_id)
    s1 = c.post(f"/jobs/{job_id}/submit", headers=h).json()
    s2 = c.post(f"/jobs/{job_id}/submit", headers=h).json()
    assert not s1["idempotent_replay"] and s2["idempotent_replay"]
    assert env.store.counts()["queued"] == 1
    # another user's identical key is a different job
    other = c.post("/jobs", headers={**env.h2, "Idempotency-Key": "uuid-1"}, json={"prompt": "x"})
    assert other.json()["id"] != job_id


def test_concurrent_duplicate_submit_is_reported_as_replay(env: Env) -> None:
    """The route reads ``draft``, then a concurrent duplicate wins the draft -> queued transition.

    ``queue_draft`` then returns False for the loser, which must answer ``idempotent_replay: true``
    (it used to say false for both). Simulated deterministically by letting a "winner" call run
    between the route's status read and its own ``queue_draft``.
    """
    c = env.client()
    job_id = c.post("/jobs", headers=env.h, json={"prompt": "apples"}).json()["id"]
    env.upload(c, job_id)
    real = env.store.queue_draft
    calls: list[bool] = []

    def racing_queue_draft(jid: str, *a: object, **kw: object) -> bool:
        if not calls:
            assert real(jid, *a, **kw) is True  # the concurrent winner
        calls.append(True)
        return bool(real(jid, *a, **kw))  # the request under test lost the race

    env.store.queue_draft = racing_queue_draft  # type: ignore[method-assign]
    r = c.post(f"/jobs/{job_id}/submit", headers=env.h)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert r.json()["idempotent_replay"] is True
    assert env.store.counts()["queued"] == 1


def test_sse_stream_reaches_done(env: Env) -> None:
    c = env.client()
    job_id = env.queued_job(c)
    worker = threading.Thread(target=lambda: (time.sleep(0.2), env.work()))
    worker.start()
    events = []
    with c.stream("GET", f"/jobs/{job_id}/events", headers=env.h) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        name = ""
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line[5:])))
    worker.join()
    assert events[0][0] == "progress" and events[0][1]["status"] == "queued"
    assert events[-1][0] == "done" and events[-1][1]["status"] == "done"


def test_revoked_share_gives_404(env: Env) -> None:
    c = env.client()
    job_id = env.queued_job(c)
    env.work()
    tok = c.post(f"/jobs/{job_id}/share", headers=env.h).json()["token"]
    view = c.get(f"/share/{tok}")  # no auth
    assert view.status_code == 200
    assert view.json()["result"]["count"] == 7 and "user_id" not in view.json()["job"]
    assert c.delete(f"/share/{tok}", headers=env.h2).status_code == 404  # not the owner
    assert c.delete(f"/share/{tok}", headers=env.h).status_code == 200
    assert c.get(f"/share/{tok}").status_code == 404
    assert c.get("/share/st_nonsense").status_code == 404


def test_submit_rate_limit_gives_429(tmp_path: Path) -> None:
    env = Env(tmp_path, rate_limit=1, rate_window_s=3600.0)
    c = env.client()
    env.queued_job(c)
    second = c.post("/jobs", headers=env.h, json={"prompt": "x"}).json()["id"]
    env.upload(c, second)
    r = c.post(f"/jobs/{second}/submit", headers=env.h)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    assert r.json()["error"]["code"] == "ERR_RATE_LIMITED" and r.json()["error"]["retryable"]
    assert env.store.get(second)["status"] == "draft"  # type: ignore[index]


def test_request_rate_limit_gives_429(env: Env) -> None:
    c = env.client(request_rate_limit=3, request_rate_window_s=3600.0)
    codes = [c.get("/jobs", headers=env.h).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert c.get("/jobs", headers=env.h2).status_code == 200  # per user
    assert c.get("/health").status_code == 200  # unauthenticated routes are not limited


def test_permanent_error_lands_in_dead_letter(env: Env) -> None:
    c = env.client()
    job_id = env.queued_job(c)
    env.work(ok=False, code=ErrorCode.ERR_CORRUPT_MEDIA)
    job = c.get(f"/jobs/{job_id}", headers=env.h).json()
    assert job["status"] == "dead_letter" and job["error_code"] == "ERR_CORRUPT_MEDIA"
    assert job["tip"]
    r = c.get(f"/jobs/{job_id}/result", headers=env.h)
    assert r.status_code == HTTP_STATUS[ErrorCode.ERR_CORRUPT_MEDIA]
    assert r.json()["error"]["code"] == "ERR_CORRUPT_MEDIA" and not r.json()["error"]["retryable"]
    assert env.store.claim("w1") is None  # not retried
    with c.stream("GET", f"/jobs/{job_id}/events", headers=env.h) as s:
        assert "event: done" in "".join(s.iter_text())


def test_retryable_error_requeues(env: Env) -> None:
    c = env.client()
    job_id = env.queued_job(c)
    env.work(ok=False, code=ErrorCode.ERR_STORAGE_TRANSIENT)
    assert c.get(f"/jobs/{job_id}", headers=env.h).json()["status"] == "queued"


def test_every_error_code_has_an_http_status() -> None:
    assert set(HTTP_STATUS) == set(ErrorCode)


def test_upload_errors_map_to_status(env: Env) -> None:
    c = env.client()
    data = _png()
    base = {
        "total_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "chunk_size": 1000,
    }
    r = c.post("/uploads", headers=env.h, json={**base, "filename": "a.exe"})
    assert r.status_code == 415 and r.json()["error"]["code"] == "ERR_UNSUPPORTED_MEDIA"
    sid = c.post("/uploads", headers=env.h, json={**base, "filename": "a.png"}).json()["session_id"]
    bad = c.put(f"/uploads/{sid}/chunks/0", headers=env.h, content=b"x" * 1000)
    assert bad.status_code == 200  # no per-chunk hash given; length is right
    r = c.put(
        f"/uploads/{sid}/chunks/1",
        headers={**env.h, "X-Chunk-SHA256": "0" * 64},
        content=data[1000:2000],
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "ERR_CHUNK_CORRUPT"
    r = c.post(f"/uploads/{sid}/finalize", headers=env.h, json={"prompt": "x"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "ERR_UPLOAD_INCOMPLETE"
    # someone else's session is invisible
    assert c.get(f"/uploads/{sid}", headers=env.h2).status_code == 404
    assert c.get("/uploads/nope", headers=env.h).status_code == 404


def test_validation_error_is_typed_400(env: Env) -> None:
    r = env.client().post("/jobs", headers=env.h, json={"prompt": ""})
    assert r.status_code == 400 and r.json()["error"]["code"] == "ERR_INVALID_INPUT"


def test_delete_running_job_conflicts(env: Env) -> None:
    c = env.client()
    job_id = env.queued_job(c)
    env.store.claim("w1")
    assert c.delete(f"/jobs/{job_id}", headers=env.h).status_code == 409


def test_docs_and_cli(tmp_path: Path) -> None:
    out = tmp_path / "API.md"
    assert main(["serve-api", "--db", str(tmp_path / "j.db"), "--write-docs", str(out)]) == 0
    text = out.read_text()
    for route in ("/jobs/{job_id}/events", "/share/{token}", "/health", "/uploads"):
        assert route in text
    env = Env(tmp_path)
    assert render_api_md(create_app(env.store)) == text
    committed = Path("docs/API.md")
    if committed.exists():
        assert "/jobs/{job_id}/submit" in committed.read_text()
