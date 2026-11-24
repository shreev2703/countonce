"""Resumable chunked uploads: resume, corruption, duplicates, final hash, expiry, CLI."""

import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image

from countonce.cli import main
from countonce.errors import CountOnceError, ErrorCode
from countonce.jobs.store import DAY_S, SqliteJobStore
from countonce.jobs.uploads import (
    LocalTransport,
    UploadError,
    UploadLimits,
    UploadService,
    file_sha256,
    upload_file,
)

CHUNK = 1000


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def _png(tmp_path: Path, name: str = "photo.png") -> Path:
    import random

    rng = random.Random(0)
    img = Image.new("RGB", (64, 64))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(4096)])
    buf = io.BytesIO()
    img.save(buf, "PNG")
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    assert p.stat().st_size > 3 * CHUNK
    return p


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    store = SqliteJobStore(tmp_path / "j.db", clock=clock, output_root=tmp_path / "outputs")
    svc = UploadService(store, tmp_path / "uploads")
    return clock, store, svc, tmp_path


def _create(svc, path, chunk=CHUNK):
    return svc.create_session("u", path.name, path.stat().st_size, file_sha256(path), chunk)


def _code(exc) -> ErrorCode:
    return exc.value.code


def test_interrupted_upload_resumes_from_right_offset(env):
    _, store, svc, tmp = env
    src = _png(tmp)
    t = LocalTransport(svc)

    class Dropped(Exception):
        pass

    sent = []

    def drop_after_two(i, n):
        sent.append(i)
        if len(sent) == 2:
            raise Dropped

    sid_box = []
    with pytest.raises(Dropped):
        upload_file(
            t, "u", src, chunk_size=CHUNK, prompt="apples",
            on_session=sid_box.append, on_chunk=drop_after_two,
        )  # fmt: skip
    sid = sid_box[0]
    st = svc.status(sid)
    assert st["received"] == [0, 1]
    assert st["next_missing_index"] == 2 and st["next_offset"] == 2 * CHUNK
    assert not st["complete"]

    resumed = []
    out = upload_file(
        t, "u", src, resume=sid, prompt="apples", on_chunk=lambda i, n: resumed.append(i)
    )
    assert resumed == list(range(2, st["n_chunks"]))  # never re-sent 0 and 1
    final = Path(out["path"])
    # PNG re-encode keeps pixels; the *assembled* bytes were byte-identical before sanitizing.
    assert Image.open(final).tobytes() == Image.open(src).tobytes()
    files = store.job_files(out["job_id"])
    assert [f["key"] for f in files] == [str(final)]
    assert files[0]["sha256"] == file_sha256(final) and files[0]["exif_stripped"] == 1
    assert store.get(out["job_id"])["status"] == "queued"


def test_assembled_output_is_byte_identical_for_video_passthrough(env, monkeypatch):
    """Videos are validated, not re-encoded, so the stored file equals the upload."""
    _, store, svc, tmp = env
    import countonce.jobs.uploads as up

    monkeypatch.setattr(up, "validate_video", lambda *a, **k: {})
    src = tmp / "clip.mp4"
    src.write_bytes(bytes(range(256)) * 20)
    out = upload_file(LocalTransport(svc), "u", src, chunk_size=777, prompt="x")
    assert Path(out["path"]).read_bytes() == src.read_bytes()


def test_corrupt_chunk_rejected(env):
    _, _, svc, tmp = env
    src = _png(tmp)
    sid = _create(svc, src)["session_id"]
    data = src.read_bytes()[:CHUNK]
    bad = bytes([data[0] ^ 1]) + data[1:]
    with pytest.raises(UploadError) as e:
        svc.put_chunk(sid, 0, bad, hashlib.sha256(data).hexdigest())
    assert _code(e) is ErrorCode.ERR_CHUNK_CORRUPT and e.value.retryable
    with pytest.raises(UploadError) as e:  # wrong length
        svc.put_chunk(sid, 0, data[:-1])
    assert _code(e) is ErrorCode.ERR_CHUNK_CORRUPT
    assert svc.status(sid)["received"] == []  # nothing stored
    svc.put_chunk(sid, 0, data, hashlib.sha256(data).hexdigest())  # resend succeeds
    assert svc.status(sid)["received"] == [0]


def test_duplicate_chunk_is_noop(env):
    _, _, svc, tmp = env
    src = _png(tmp)
    sid = _create(svc, src)["session_id"]
    data = src.read_bytes()[:CHUNK]
    first = svc.put_chunk(sid, 0, data)
    part = Path(svc._session(sid)["dir"]) / "000000.part"
    mtime = part.stat().st_mtime_ns
    again = svc.put_chunk(sid, 0, data)
    assert first["duplicate"] is False and again["duplicate"] is True
    assert again["received"] == [0]
    assert part.stat().st_mtime_ns == mtime
    with pytest.raises(UploadError) as e:  # same index, different valid content
        svc.put_chunk(sid, 0, b"z" * CHUNK)
    assert _code(e) is ErrorCode.ERR_INVALID_INPUT


def test_wrong_final_hash_fails_finalize(env):
    _, store, svc, tmp = env
    src = _png(tmp)
    data = src.read_bytes()
    lie = hashlib.sha256(b"not the file").hexdigest()
    sid = svc.create_session("u", src.name, len(data), lie, CHUNK)["session_id"]
    for i in range(0, len(data), CHUNK):
        svc.put_chunk(sid, i // CHUNK, data[i : i + CHUNK])
    with pytest.raises(UploadError) as e:
        svc.finalize(sid, prompt="apples")
    assert _code(e) is ErrorCode.ERR_CORRUPT_MEDIA
    assert svc.status(sid)["status"] == "failed"
    assert store.list_jobs() == []  # nothing attached
    with pytest.raises(UploadError):
        svc.put_chunk(sid, 0, data[:CHUNK])  # failed sessions accept nothing


def test_finalize_incomplete_and_idempotent(env):
    _, store, svc, tmp = env
    src = _png(tmp)
    st = _create(svc, src)
    sid = st["session_id"]
    with pytest.raises(UploadError) as e:
        svc.finalize(sid, prompt="p")
    assert _code(e) is ErrorCode.ERR_UPLOAD_INCOMPLETE
    out = upload_file(LocalTransport(svc), "u", src, resume=sid, prompt="p")
    again = svc.finalize(sid, prompt="p")
    assert again["already_finalized"] and again["job_id"] == out["job_id"]
    assert len(store.list_jobs()) == 1


def test_attach_to_existing_job_and_ownership(env):
    _, store, svc, tmp = env
    src = _png(tmp)
    jid, _ = store.submit("u", "apples", [])
    out = upload_file(LocalTransport(svc), "u", src, job_id=jid)
    assert out["job_id"] == jid and len(store.job_files(jid)) == 1
    other, _ = store.submit("someone-else", "x", [])
    with pytest.raises(UploadError) as e:
        upload_file(LocalTransport(svc), "u", src, job_id=other)
    assert _code(e) is ErrorCode.ERR_INVALID_INPUT


def test_limits_and_typed_errors(env):
    _, store, svc, tmp = env
    small = UploadService(
        store, tmp / "u2", UploadLimits(max_bytes=5000, max_chunk_bytes=2000, max_chunks=4)
    )
    h = "a" * 64
    cases = [
        (("u", "a.png", 5001, h, 1000), ErrorCode.ERR_FILE_TOO_LARGE),
        (("u", "a.png", 4000, h, 2001), ErrorCode.ERR_INVALID_INPUT),
        (("u", "a.png", 4000, h, 500), ErrorCode.ERR_INVALID_INPUT),  # 8 chunks > 4
        (("u", "a.txt", 100, h, 100), ErrorCode.ERR_UNSUPPORTED_MEDIA),
        (("u", "a.png", 100, "xyz", 100), ErrorCode.ERR_INVALID_INPUT),
        (("u", "a.png", 0, h, 100), ErrorCode.ERR_INVALID_INPUT),
    ]
    for args, code in cases:
        with pytest.raises(CountOnceError) as e:
            small.create_session(*args)
        assert _code(e) is code, args
    sid = small.create_session("u", "../../evil.png", 100, h, 100)["session_id"]
    assert small.status(sid)["filename"] == "evil.png"
    with pytest.raises(UploadError):
        small.put_chunk(sid, 5, b"x" * 100)
    with pytest.raises(UploadError):
        small.status("nope")


def test_limits_from_packaged_yaml():
    from countonce.resources import resolve_config

    lim = UploadLimits.from_yaml(resolve_config(Path("configs/uploads.yaml")))
    assert lim.max_chunk_bytes == 64 * 1024**2 and lim.session_ttl_s == 86400


def test_expired_sessions_purged(env):
    clock, store, svc, tmp = env
    src = _png(tmp)
    abandoned = _create(svc, src)["session_id"]
    svc.put_chunk(abandoned, 0, src.read_bytes()[:CHUNK])
    adir = Path(svc._session(abandoned)["dir"])
    assert (adir / "000000.part").exists()
    clock.t += 0.5 * DAY_S
    fresh = _create(svc, src)["session_id"]  # younger, must survive
    clock.t += 0.75 * DAY_S  # abandoned is 1.25 d idle (> 1 d ttl); fresh is 0.75 d
    with pytest.raises(UploadError):
        svc.put_chunk(abandoned, 1, src.read_bytes()[CHUNK : 2 * CHUNK])  # expired: refused
    store.purge_expired()
    assert not adir.exists()
    rows = store._query("SELECT id FROM upload_sessions")
    assert [r["id"] for r in rows] == [fresh]
    assert store._query("SELECT * FROM upload_chunks") == []


def test_purge_keeps_finalized_file_until_job_retention(env):
    clock, store, svc, tmp = env
    src = _png(tmp)
    out = upload_file(LocalTransport(svc), "u", src, chunk_size=CHUNK, prompt="p")
    clock.t += 2 * DAY_S
    store.purge_expired()
    assert store._query("SELECT id FROM upload_sessions") == []
    assert Path(out["path"]).exists()  # job is queued, its input stays


def test_existing_db_migrates_additively(tmp_path):
    store = SqliteJobStore(tmp_path / "old.db")  # a db that never saw uploads
    assert store._query("SELECT name FROM sqlite_master WHERE name = 'upload_sessions'") == []
    store.purge_expired()  # creates the tables on demand, does not fail
    UploadService(store)
    assert store._query("SELECT name FROM sqlite_master WHERE name = 'upload_chunks'")


def test_cli_upload_and_resume(tmp_path, capsys):
    db = tmp_path / "j.db"
    src = _png(tmp_path)
    assert main(["upload", str(src), "--db", str(db), "--prompt", "apples"]) == 0
    out = capsys.readouterr().out
    assert "upload session " in out and "upload: job " in out
    jid = out.split("upload: job ")[1].split()[0]
    assert SqliteJobStore(db).get(jid)["status"] == "queued"
    # no prompt and no job -> usage error; missing file -> 1
    assert main(["upload", str(src), "--db", str(db)]) == 2
    assert main(["upload", str(tmp_path / "nope.png"), "--db", str(db), "--prompt", "p"]) == 1
    # resume against an unknown session fails with a typed message
    assert main(["upload", str(src), "--db", str(db), "--prompt", "p", "--resume", "bogus"]) == 1
    assert "ERR_INVALID_INPUT" in capsys.readouterr().err


def test_cli_resume_after_partial(tmp_path, capsys):
    db = tmp_path / "j.db"
    src = _png(tmp_path)
    svc = UploadService(SqliteJobStore(db), tmp_path / "uploads")
    # CLI default root is <db dir>/uploads, same as here.
    sid = _create(svc, src)["session_id"]
    svc.put_chunk(sid, 0, src.read_bytes()[:CHUNK])
    other = tmp_path / "other.png"
    other.write_bytes(_png(tmp_path, "x.png").read_bytes() + b"\0")
    assert main(["upload", str(other), "--db", str(db), "--prompt", "p", "--resume", sid]) == 1
    capsys.readouterr()
    code = main(
        ["upload", str(src), "--db", str(db), "--prompt", "p", "--resume", sid, "--chunk-mb", "9"]
    )
    out = capsys.readouterr().out
    assert code == 0 and "chunk 1/" not in out and "chunk 2/" in out  # offset 0 was kept
