"""WebDataset-layout shards for rendered scenes (PLAN §16B.1, Layout).

One scene is one sample: ``<key>.json`` (metadata) and ``<key>.npz`` (images, depth, masks,
cameras). Samples are packed into plain ``.tar`` shards with stdlib ``tarfile`` (no webdataset
dependency; any WebDataset reader groups members by the shared key prefix). A shard is closed
before its payload would exceed ``max_bytes``; the file itself is padded by tar to a 10 KiB record
(a single oversized sample still gets its own shard).

Output is deterministic: fixed mtime/owner in tar headers, a fixed zip timestamp in each npz, so
identical scenes give byte-identical shards and the DAG's hash-based skipping works. ``index.json``
lists every shard with its sha256, size and sample keys; ``read_shards`` verifies the hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

INDEX_NAME = "index.json"
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class ShardError(ValueError):
    pass


@dataclass
class Sample:
    key: str
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def scene_sample(key: str, scene: Any, extra_meta: dict[str, Any] | None = None) -> Sample:
    """Sample from a `synth.validate.RenderedScene`."""
    arrays = {
        "images": scene.images,
        "depth": scene.depth,
        "instance_masks": scene.instance_masks,
        "camera_centers": scene.camera_centers,
    }
    if scene.camera_forward is not None:
        arrays["camera_forward"] = scene.camera_forward
    return Sample(key, {**scene.metadata, **(extra_meta or {})}, arrays)


def encode_npz(arrays: dict[str, np.ndarray]) -> bytes:
    """Deterministic .npz bytes (`np.savez` stamps the current time into the zip entries)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(arrays[name]), allow_pickle=False)
    return buf.getvalue()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_shards(
    samples: Iterable[Sample],
    out_dir: Path | str,
    *,
    max_bytes: int = 1 << 30,
    prefix: str = "shard",
) -> dict[str, Any]:
    """Pack `samples` into ``<out_dir>/<prefix>-NNNNN.tar`` and write ``index.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    tar: tarfile.TarFile | None = None
    cur: dict[str, Any] = {}
    seen: set[str] = set()

    def close() -> None:
        nonlocal tar
        if tar is None:
            return
        tar.close()
        tar = None
        path = out / cur["name"]
        cur["bytes"] = path.stat().st_size
        cur["sha256"] = _sha256(path)
        shards.append(dict(cur))

    try:
        for s in samples:
            if s.key in seen or not s.key or "/" in s.key or "." in s.key:
                raise ShardError(f"invalid or duplicate sample key {s.key!r}")
            seen.add(s.key)
            meta = json.dumps(s.metadata, sort_keys=True, default=str).encode()
            blob = encode_npz(s.arrays)
            need = len(meta) + len(blob) + 2048  # two 512 B headers plus block padding
            if tar is not None and cur["bytes_est"] + need > max_bytes and cur["keys"]:
                close()
            if tar is None:
                cur = {"name": f"{prefix}-{len(shards):05d}.tar", "keys": [], "bytes_est": 1024}
                tar = tarfile.open(out / cur["name"], "w", format=tarfile.USTAR_FORMAT)
            _add(tar, f"{s.key}.json", meta)
            _add(tar, f"{s.key}.npz", blob)
            cur["keys"].append(s.key)
            cur["bytes_est"] += need
        close()
    except BaseException:
        if tar is not None:
            tar.close()
            (out / cur["name"]).unlink(missing_ok=True)
        for sh in shards:  # no index.json is written, so earlier shards would be orphans
            (out / sh["name"]).unlink(missing_ok=True)
        raise
    for sh in shards:
        sh.pop("bytes_est", None)
    index = {
        "format": "webdataset-tar",
        "max_bytes": max_bytes,
        "n_samples": len(seen),
        "shards": shards,
    }
    (out / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def read_shards(shard_dir: Path | str, verify: bool = True) -> Iterator[Sample]:
    """Yield samples in index order; raises `ShardError` on a hash mismatch or a broken pair."""
    root = Path(shard_dir)
    index = json.loads((root / INDEX_NAME).read_text())
    for sh in index["shards"]:
        path = root / sh["name"]
        if verify and _sha256(path) != sh["sha256"]:
            raise ShardError(f"sha256 mismatch for {sh['name']}")
        pending: dict[str, dict[str, Any]] = {}
        with tarfile.open(path, "r") as tar:
            for member in tar:
                f = tar.extractfile(member)
                if f is None:
                    continue
                key, _, ext = member.name.rpartition(".")
                pending.setdefault(key, {})[ext] = f.read()
                got = pending[key]
                if "json" in got and "npz" in got:
                    del pending[key]
                    with np.load(io.BytesIO(got["npz"])) as z:
                        arrays = {k: z[k] for k in z.files}
                    yield Sample(key, json.loads(got["json"]), arrays)
        if pending:
            raise ShardError(f"{sh['name']}: incomplete samples {sorted(pending)}")
