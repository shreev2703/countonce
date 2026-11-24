"""BlobStore contract over the local and S3 (moto, no network) backends, plus the routed paths."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest

pytest.importorskip("boto3")
pytest.importorskip("moto")

import boto3  # noqa: E402
from moto import mock_aws  # noqa: E402

from countonce.detect.cache import CachedDetector, DetectionCache  # noqa: E402
from countonce.jobs.store import SqliteJobStore  # noqa: E402
from countonce.jobs.uploads import UploadService, upload_file  # noqa: E402
from countonce.jobs.worker import Worker  # noqa: E402
from countonce.recon.cache import CachedReconstructor, ReconCache  # noqa: E402
from countonce.storage import (  # noqa: E402
    BlobNotFound,
    BlobStore,
    LocalBlobStore,
    SignedURLError,
    delete_prefix,
    expire_prefix,
    open_blobstore,
)
from countonce.storage.factory import cache_kwargs  # noqa: E402
from countonce.storage.s3 import S3BlobStore  # noqa: E402
from countonce.types import Detection, Frame, Recon  # noqa: E402

BUCKET = "countonce-test"


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "testing")  # moto only; not a real credential
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(params=["local", "s3"])
def blobs(request: pytest.FixtureRequest, tmp_path: Path) -> BlobStore:
    if request.param == "local":
        yield LocalBlobStore(tmp_path / "blobs", secret="s3cret")  # type: ignore[misc]
        return
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield S3BlobStore(BUCKET, client=client, prefix="t")  # type: ignore[misc]


# -- contract -------------------------------------------------------------------------------
def test_put_get_exists_delete(blobs: BlobStore) -> None:
    assert not blobs.exists("a/b.bin")
    blobs.put("a/b.bin", b"hello")
    assert blobs.exists("a/b.bin")
    assert blobs.get("a/b.bin") == b"hello"
    blobs.put("a/b.bin", b"replaced")  # overwrite
    assert blobs.get("a/b.bin") == b"replaced"
    assert blobs.delete("a/b.bin") is True
    assert blobs.delete("a/b.bin") is False
    assert not blobs.exists("a/b.bin")


def test_get_missing_raises(blobs: BlobStore) -> None:
    with pytest.raises(BlobNotFound):
        blobs.get("nope/missing")


def test_list_prefix(blobs: BlobStore) -> None:
    for k in ("x/1", "x/2", "x/sub/3", "y/1"):
        blobs.put(k, k.encode())
    assert [b.key for b in blobs.list_prefix("x/")] == ["x/1", "x/2", "x/sub/3"]
    assert [b.key for b in blobs.list_prefix("y")] == ["y/1"]
    assert blobs.list_prefix("zzz") == []
    info = blobs.list_prefix("y/")[0]
    assert info.size == 3 and info.modified > 0
    assert delete_prefix(blobs, "x/") == 3
    assert [b.key for b in blobs.list_prefix("")] == ["y/1"]


@pytest.mark.parametrize("bad", ["", "/abs", "a/../b", "a//b", "a/./b", "a\\b"])
def test_bad_keys_rejected(blobs: BlobStore, bad: str) -> None:
    with pytest.raises(ValueError):
        blobs.put(bad, b"x")


def test_presign_urls(blobs: BlobStore) -> None:
    blobs.put("p/q.txt", b"data")
    get_url = blobs.presign_get("p/q.txt", 60)
    put_url = blobs.presign_put("p/new.txt", 60)
    assert get_url != put_url
    assert "p/q.txt" in get_url and "p/new.txt" in put_url
    assert "sig=" in get_url or "Signature" in get_url or "X-Amz-Signature" in get_url


def test_protocol(blobs: BlobStore) -> None:
    assert isinstance(blobs, BlobStore)


# -- local signed URLs ----------------------------------------------------------------------
def test_local_signed_url_roundtrip_and_expiry(tmp_path: Path) -> None:
    clock = Clock()
    st = LocalBlobStore(tmp_path, secret="k", clock=clock)
    url = st.presign_get("a/b c.txt", 60)
    assert st.verify_url("GET", url) == "a/b c.txt"
    clock.t += 59
    st.verify_url("GET", url)
    clock.t += 2
    with pytest.raises(SignedURLError, match="expired"):
        st.verify_url("GET", url)


def test_local_signed_url_tamper_and_method(tmp_path: Path) -> None:
    clock = Clock()
    st = LocalBlobStore(tmp_path, secret="k", clock=clock)
    url = st.presign_get("a.txt", 60)
    with pytest.raises(SignedURLError):
        st.verify_url("PUT", url)  # a GET URL cannot write
    with pytest.raises(SignedURLError):
        st.verify_url("GET", url.replace("a.txt", "other.txt"))
    q = urlsplit(url).query
    forged = url.replace(q.split("&")[1], "expires=9999999999")  # extend expiry
    with pytest.raises(SignedURLError):
        st.verify_url("GET", forged)
    other = LocalBlobStore(tmp_path, secret="different", clock=clock)
    with pytest.raises(SignedURLError):
        other.verify_url("GET", url)  # signed with another secret
    with pytest.raises(SignedURLError):
        st.verify_url("GET", "/blobs/a.txt")  # unsigned


def test_s3_presigned_get_is_real_and_expiring() -> None:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        st = S3BlobStore(BUCKET, region="us-east-1")  # built-in client: SigV4
        st.put("k", b"v")
        q = urlsplit(st.presign_get("k", 120)).query
        assert "X-Amz-Signature=" in q and "X-Amz-Expires=120" in q


# -- factory --------------------------------------------------------------------------------
def test_factory_local_and_s3(tmp_path: Path) -> None:
    assert isinstance(open_blobstore(env={}, local_root=tmp_path), LocalBlobStore)
    with mock_aws():
        env = {
            "COUNTONCE_STORAGE": "s3",
            "COUNTONCE_S3_BUCKET": BUCKET,
            "COUNTONCE_S3_ENDPOINT_URL": "https://example.r2.cloudflarestorage.com",
            "COUNTONCE_S3_REGION": "auto",
        }
        st = open_blobstore(env=env)
        assert isinstance(st, S3BlobStore) and st.bucket == BUCKET
    with pytest.raises(ValueError, match="BUCKET"):
        open_blobstore("s3", env={})
    with pytest.raises(ValueError, match="unknown storage"):
        open_blobstore("ftp", env={})
    assert cache_kwargs("recon", env={}) == {}


# -- routed paths ---------------------------------------------------------------------------
def _recon() -> Recon:
    n = 2
    return Recon(
        backbone="fake",
        intrinsics=np.tile(np.eye(3), (n, 1, 1)),
        extrinsics=np.tile(np.eye(4)[:3], (n, 1, 1)),
        depth=np.ones((n, 4, 4)),
        depth_conf=np.ones((n, 4, 4)),
        world_points=np.zeros((n, 4, 4, 3)),
        world_points_conf=np.ones((n, 4, 4)),
        scene_scale=1.0,
        runtime_s=0.5,
        peak_vram_mb=0.0,
    )


class _FakeRecon:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def config(self) -> dict[str, object]:
        return {}

    def reconstruct(self, frames: list[Frame]) -> Recon:
        self.calls += 1
        return _recon()


def _frames() -> list[Frame]:
    return [Frame(id=i, path=Path(f"{i}.jpg"), width=4, height=4, sha256=f"h{i}") for i in range(2)]


def test_recon_cache_hit_over_blobstore(blobs: BlobStore) -> None:
    inner = _FakeRecon()
    cr = CachedReconstructor(inner, ReconCache(blobs=blobs, prefix="cache/recon"), "v1")  # type: ignore[arg-type]
    first = cr.reconstruct_cached(_frames())
    second = cr.reconstruct_cached(_frames())
    assert (first.hit, second.hit) == (False, True)
    assert inner.calls == 1
    assert blobs.exists(f"cache/recon/{first.key}.npz")
    assert np.allclose(second.recon.depth, first.recon.depth)


def test_recon_cache_corrupt_blob_is_a_miss(blobs: BlobStore) -> None:
    cache = ReconCache(blobs=blobs, prefix="r")
    blobs.put("r/k.npz", b"not an npz")
    assert cache.get("k") is None


def test_detection_cache_hit_over_blobstore(blobs: BlobStore) -> None:
    class Det:
        name, version = "d", "1"
        calls = 0

        def config(self) -> dict[str, object]:
            return {}

        def detect(self, frames, prompt, exemplars=None):  # type: ignore[no-untyped-def]
            Det.calls += 1
            return [
                Detection(
                    id=Detection.make_id(f.id, 0),
                    frame_id=f.id,
                    score=0.9,
                    box_xyxy=np.array([0, 0, 1, 1.0]),
                    point_xy=None,
                )
                for f in frames
            ]

    cd = CachedDetector(Det(), DetectionCache(blobs=blobs, prefix="cache/detect"))  # type: ignore[arg-type]
    a = cd.detect(_frames(), "apple")
    b = cd.detect(_frames(), "apple")
    assert len(a) == len(b) == 2 and Det.calls == 1 and cd.hits == 2
    assert len(blobs.list_prefix("cache/detect/")) == 2


def test_cache_ttl_expiry(blobs: BlobStore) -> None:
    cache = ReconCache(blobs=blobs, prefix="c", ttl_s=100)
    cache.put("old", _recon())
    modified = blobs.list_prefix("c/")[0].modified
    assert cache.expire(now=modified + 50) == 0
    assert cache.get("old") is not None
    assert cache.expire(now=modified + 101) == 1
    assert cache.get("old") is None
    assert ReconCache(blobs=blobs, prefix="c").expire() == 0  # no TTL: never expires
    assert expire_prefix(blobs, "zz/", 1) == 0


def test_cache_kwargs_s3_env() -> None:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        kw = cache_kwargs("recon", env={"COUNTONCE_STORAGE": "s3", "COUNTONCE_S3_BUCKET": BUCKET})
        cache = ReconCache(Path("/unused"), **kw)  # type: ignore[arg-type]
        cache.put("k", _recon())
        assert isinstance(cache.blobs, S3BlobStore)
        assert cache.get("k") is not None


def _png(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (32, 32), (200, 10, 10)).save(path)


def test_upload_assembly_and_purge_over_blobstore(blobs: BlobStore, tmp_path: Path) -> None:
    clock = Clock()
    store = SqliteJobStore(tmp_path / "jobs.db", clock=clock, blobs=blobs)
    svc = UploadService(store, tmp_path / "uploads", blobs=blobs)
    img = tmp_path / "in.png"
    _png(img)
    from countonce.jobs.uploads import LocalTransport

    out = upload_file(LocalTransport(svc), "u1", img, chunk_size=64, prompt="red")
    job = out["job_id"]
    (f,) = store.job_files(job, "input")
    assert blobs.exists(f["key"]) or (Path(f["key"]).is_file())
    assert blobs.list_prefix("sessions/") == []  # chunks removed after assembly
    assert len(blobs.list_prefix("files/")) == 1

    # worker download goes through the store too
    seen: list[Path] = []

    def runner(job_row, inputs, progress):  # type: ignore[no-untyped-def]
        seen.extend(inputs)
        assert all(p.is_file() for p in inputs)
        out_dir = tmp_path / "out" / job_row["id"]
        out_dir.mkdir(parents=True)
        (out_dir / "result.json").write_text(json.dumps({"ok": 1}))
        from countonce.types import CountResult  # noqa: F401

        raise RuntimeError("stop after upload check")

    w = Worker(store, runner, blobs=blobs, out_root=tmp_path / "out")
    w.step()
    assert len(seen) == 1 and seen[0].suffix == ".png"

    # outputs path: upload then purge removes every blob
    w._upload_outputs(job)
    assert blobs.exists(f"outputs/{job}/result.json")
    assert [r["kind"] for r in store.job_files(job, "output")] == ["output"]
    done_at = clock.t
    with store._tx() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'done', finished_at = ? WHERE id = ?", (done_at, job)
        )
    clock.t += 31 * 86400
    assert store.purge_expired() == [job]
    assert blobs.list_prefix("files/") == []
    assert blobs.list_prefix("outputs/") == []


def test_expired_upload_session_chunks_purged(blobs: BlobStore, tmp_path: Path) -> None:
    clock = Clock()
    store = SqliteJobStore(tmp_path / "jobs.db", clock=clock, blobs=blobs)
    svc = UploadService(store, tmp_path / "uploads", blobs=blobs)
    data = b"x" * 10
    st = svc.create_session("u", "a.png", 10, hashlib.sha256(data).hexdigest(), 5)
    svc.put_chunk(st["session_id"], 0, data[:5])
    assert len(blobs.list_prefix("sessions/")) == 1
    clock.t += 2 * 86400
    store.purge_expired()
    assert blobs.list_prefix("sessions/") == []


def test_worker_downloads_from_s3_into_scratch(tmp_path: Path) -> None:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        st = S3BlobStore(BUCKET)
        store = SqliteJobStore(tmp_path / "j.db", blobs=st)
        st.put("files/s/pic.png", b"PNGDATA")
        jid, _ = store.submit("u", "red", ["files/s/pic.png"])
        got: list[bytes] = []

        def runner(job, inputs, progress):  # type: ignore[no-untyped-def]
            got.append(inputs[0].read_bytes())
            raise RuntimeError("done")

        Worker(store, runner, blobs=st).step()
        assert got == [b"PNGDATA"]
        assert io  # keep import used
