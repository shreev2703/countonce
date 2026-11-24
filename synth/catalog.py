"""DuckDB catalog over scene metadata and the feature store (PLAN §16B.1, Catalog).

Scene metadata is kept as a small Parquet file (``scenes.parquet``, one row per recipe including
quarantined ones). `connect` opens an in-memory DuckDB with a ``scenes`` view over it and, when a
feature store is given, a ``detections`` view. `query_slice` picks training/eval slices with
parameterised filters; `run_query` runs arbitrary SQL (the ``countonce catalog query`` command).

``occlusion`` is a recipe-level proxy: occluders per object. ``baseline_spread`` is the mean
distance of the camera centres from their centroid. Per-pixel occlusion needs real renders.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from synth.recipe import SceneRecipe

SCENE_COLUMNS = (
    "scene_id",
    "split",
    "seed",
    "recipe_version",
    "arrangement",
    "renderer",
    "n_objects",
    "n_views",
    "n_occluders",
    "occlusion",
    "baseline_spread",
    "data_version",
    "quarantined",
    "quarantine_reasons",
)


class CatalogError(ValueError):
    pass


def _deps() -> tuple[Any, Any]:
    try:
        import duckdb
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise CatalogError("the catalog needs the data extra: uv pip install -e .[data]") from exc
    return duckdb, pa


def baseline_spread(recipe: SceneRecipe) -> float:
    c = np.array([cam.position for cam in recipe.rig.cameras], dtype=float)
    return float(np.linalg.norm(c - c.mean(axis=0), axis=1).mean())


def scene_row(
    recipe: SceneRecipe,
    *,
    scene_id: str,
    split: str,
    renderer: str,
    data_version: str,
    quarantine_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "split": split,
        "seed": recipe.seed,
        "recipe_version": recipe.recipe_version,
        "arrangement": recipe.arrangement,
        "renderer": renderer,
        "n_objects": recipe.n_objects,
        "n_views": len(recipe.rig.cameras),
        "n_occluders": len(recipe.occluders),
        "occlusion": len(recipe.occluders) / max(recipe.n_objects, 1),
        "baseline_spread": baseline_spread(recipe),
        "data_version": data_version,
        "quarantined": bool(quarantine_reasons),
        "quarantine_reasons": ",".join(sorted(quarantine_reasons)),
    }


def write_scenes(rows: Sequence[dict[str, Any]], path: Path | str) -> int:
    import pyarrow.parquet as pq

    _, pa = _deps()
    types = {
        "seed": pa.int64(),
        "n_objects": pa.int32(),
        "n_views": pa.int32(),
        "n_occluders": pa.int32(),
        "occlusion": pa.float64(),
        "baseline_spread": pa.float64(),
        "quarantined": pa.bool_(),
    }
    sch = pa.schema([(c, types.get(c, pa.string())) for c in SCENE_COLUMNS])
    ordered = sorted(rows, key=lambda r: (r["split"], r["seed"]))
    table = pa.Table.from_pydict({c: [r[c] for r in ordered] for c in SCENE_COLUMNS}, schema=sch)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd", write_statistics=False)
    return int(table.num_rows)


def _lit(path: Path | str) -> str:
    """SQL string literal (DDL cannot take prepared parameters)."""
    return "'" + str(path).replace("'", "''") + "'"


def connect(scenes: Path | str, features: Path | str | None = None) -> Any:
    """In-memory DuckDB with a ``scenes`` view (and ``detections`` when `features` is given)."""
    duckdb, _ = _deps()
    if not Path(scenes).exists():
        raise CatalogError(f"scene catalog not found: {scenes}")
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE VIEW scenes AS SELECT * FROM read_parquet({_lit(scenes)})")
    if features is not None and Path(features).exists():
        con.execute(f"CREATE VIEW detections AS SELECT * FROM read_parquet({_lit(features)})")
    return con


def run_query(con: Any, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    duckdb, _ = _deps()
    try:
        cur = con.execute(sql)
        return [d[0] for d in cur.description or []], cur.fetchall()
    except duckdb.Error as exc:
        raise CatalogError(str(exc)) from exc


def query_slice(
    con: Any,
    *,
    min_objects: int | None = None,
    max_objects: int | None = None,
    arrangement: str | None = None,
    renderer: str | None = None,
    split: str | None = None,
    min_baseline: float | None = None,
    min_occlusion: float | None = None,
    include_quarantined: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Scenes matching every given filter, ordered by (split, seed). Quarantined scenes are
    excluded unless `include_quarantined`."""
    where: list[str] = []
    args: list[Any] = []

    def add(clause: str, value: Any) -> None:
        if value is not None:
            where.append(clause)
            args.append(value)

    add("n_objects >= ?", min_objects)
    add("n_objects <= ?", max_objects)
    add("arrangement = ?", arrangement)
    add("renderer = ?", renderer)
    add("split = ?", split)
    add("baseline_spread >= ?", min_baseline)
    add("occlusion >= ?", min_occlusion)
    if not include_quarantined:
        where.append("NOT quarantined")
    sql = "SELECT * FROM scenes"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY split, seed"
    if limit is not None:
        sql += " LIMIT ?"
        args.append(int(limit))
    cur = con.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def format_table(cols: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(c) for c in cols]] + [[str(v) for v in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(cols))]
    return "\n".join("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in cells)
