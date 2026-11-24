"""Parquet feature store: one row per detection (PLAN §16B.1, Feature store).

Rows come from synthetic episodes (`synth.geometry.Episode`). Lifted xyz uses the same unprojection
as `countonce.match.model.episode_inputs` but is left in world units (no per-episode
normalisation). The generator has no per-object size or box, so ``size`` is the nominal object size
and ``box`` is the nominal pixel box centred on the detection; both are placeholders until real
detector boxes exist. ``embedding`` is null unless an embedding function is supplied.

The Parquet schema metadata carries ``countonce.data_version``; `feature_store_version` reads it so
a matcher checkpoint can record which feature store it trained on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from countonce.match.model import NOMINAL_OBJECT_SIZE
from synth.geometry import Episode, load_episode

VERSION_KEY = b"countonce.data_version"
COLUMNS = (
    "scene_id",
    "split",
    "view_id",
    "det_id",
    "x",
    "y",
    "z",
    "size",
    "box_cx",
    "box_cy",
    "box_w",
    "box_h",
    "score",
    "embedding",
    "gt_instance_id",
    "recipe_version",
    "seed",
    "data_version",
)

EmbedFn = Callable[[Episode], np.ndarray]


def _pa() -> Any:
    import pyarrow as pa

    return pa


def schema() -> Any:
    pa = _pa()
    return pa.schema(
        [
            ("scene_id", pa.string()),
            ("split", pa.string()),
            ("view_id", pa.int32()),
            ("det_id", pa.int32()),
            ("x", pa.float32()),
            ("y", pa.float32()),
            ("z", pa.float32()),
            ("size", pa.float32()),
            ("box_cx", pa.float32()),
            ("box_cy", pa.float32()),
            ("box_w", pa.float32()),
            ("box_h", pa.float32()),
            ("score", pa.float32()),
            ("embedding", pa.list_(pa.float32())),
            ("gt_instance_id", pa.int32()),  # -1 = false positive
            ("recipe_version", pa.string()),
            ("seed", pa.int64()),
            ("data_version", pa.string()),
        ],
        metadata={VERSION_KEY: b""},
    )


def lift_world(ep: Episode) -> np.ndarray:
    """(n, 3) world xyz of each detection from uv, depth, K, R, t."""
    v = ep.view
    pix = np.concatenate([ep.uv, np.ones((ep.n_detections, 1))], axis=1)
    ray = np.einsum("nij,nj->ni", np.linalg.inv(ep.K[v]), pix)
    cam = ray * ep.depth[:, None]
    return np.asarray(np.einsum("nji,nj->ni", ep.R[v], cam - ep.t[v]), dtype=np.float64)


def episode_rows(
    ep: Episode,
    scene_id: str,
    split: str,
    *,
    recipe_version: str,
    data_version: str,
    embed: EmbedFn | None = None,
) -> dict[str, list[Any]]:
    n = ep.n_detections
    xyz = lift_world(ep) if n else np.zeros((0, 3))
    half = ep.K[ep.view, 0, 0] * NOMINAL_OBJECT_SIZE / np.maximum(ep.depth, 1e-6)
    emb: list[Any]
    if embed is None:
        emb = [None] * n
    else:
        e = np.asarray(embed(ep), dtype=np.float32)
        if e.ndim != 2 or e.shape[0] != n:
            raise ValueError(f"embedding must be (n_detections={n}, d), got {e.shape}")
        emb = [row.tolist() for row in e]
    return {
        "scene_id": [scene_id] * n,
        "split": [split] * n,
        "view_id": ep.view.astype(int).tolist(),
        "det_id": list(range(n)),
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "size": [NOMINAL_OBJECT_SIZE] * n,
        "box_cx": ep.uv[:, 0].tolist(),
        "box_cy": ep.uv[:, 1].tolist(),
        "box_w": (2 * half).tolist(),
        "box_h": (2 * half).tolist(),
        "score": ep.score.tolist(),
        "embedding": emb,
        "gt_instance_id": ep.identity.astype(int).tolist(),
        "recipe_version": [recipe_version] * n,
        "seed": [int(ep.seed)] * n,
        "data_version": [data_version] * n,
    }


def write_features(
    episodes: Sequence[tuple[str, str, Path]],
    out_path: Path | str,
    *,
    recipe_version: str,
    data_version: str,
    embed: EmbedFn | None = None,
) -> int:
    """Write (scene_id, split, episode .npz) triples to one Parquet file; returns the row count."""
    import pyarrow.parquet as pq

    pa = _pa()
    cols: dict[str, list[Any]] = {c: [] for c in COLUMNS}
    for scene_id, split, path in episodes:
        rows = episode_rows(
            load_episode(path),
            scene_id,
            split,
            recipe_version=recipe_version,
            data_version=data_version,
            embed=embed,
        )
        for c in COLUMNS:
            cols[c].extend(rows[c])
    sch = schema().with_metadata({VERSION_KEY: data_version.encode()})
    table = pa.Table.from_pydict(cols, schema=sch)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # fixed compression and no statistics keep the bytes stable for the DAG's output hashing
    pq.write_table(table, out, compression="zstd", write_statistics=False)
    return int(table.num_rows)


def feature_store_version(path: Path | str) -> str:
    """The data version stored in the Parquet schema metadata ('' when absent)."""
    import pyarrow.parquet as pq

    md = pq.read_schema(path).metadata or {}
    return str(md.get(VERSION_KEY, b"").decode())
