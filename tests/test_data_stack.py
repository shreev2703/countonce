"""Track J data stack: shards, Parquet feature store, DuckDB catalog, DAG lineage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("duckdb")

from synth import catalog, features, shards  # noqa: E402
from synth.dag import Dag, DagConfig, proxy_scene  # noqa: E402
from synth.geometry import GeometryConfig, generate_episode, save_episode  # noqa: E402
from synth.recipe import make_recipe  # noqa: E402


def _samples(n: int) -> list[shards.Sample]:
    return [
        shards.scene_sample(f"scene_{s:06d}", proxy_scene(make_recipe(s, n_objects=6, n_views=3)))
        for s in range(n)
    ]


def test_shard_round_trip_and_size_cap(tmp_path: Path) -> None:
    samples = _samples(5)
    idx = shards.pack_shards(samples, tmp_path, max_bytes=8000)
    assert len(idx["shards"]) > 1 and idx["n_samples"] == 5
    for sh in idx["shards"]:
        assert (tmp_path / sh["name"]).stat().st_size <= 8000 + 10240
        assert len(sh["sha256"]) == 64
    back = list(shards.read_shards(tmp_path))
    assert [b.key for b in back] == [s.key for s in samples]
    for a, b in zip(samples, back, strict=True):
        assert json.loads(json.dumps(a.metadata, default=str)) == b.metadata
        for k, arr in a.arrays.items():
            np.testing.assert_array_equal(arr, b.arrays[k])


def test_shards_are_deterministic_and_tamper_detected(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    shards.pack_shards(_samples(3), a)
    shards.pack_shards(_samples(3), b)
    assert (a / "shard-00000.tar").read_bytes() == (b / "shard-00000.tar").read_bytes()
    with (a / "shard-00000.tar").open("r+b") as f:
        f.seek(700)
        f.write(b"\xff")
    with pytest.raises(shards.ShardError, match="sha256"):
        list(shards.read_shards(a))
    with pytest.raises(shards.ShardError, match="duplicate"):
        shards.pack_shards(_samples(1) * 2, tmp_path / "c")


def test_failed_pack_leaves_no_partial_shards(tmp_path: Path) -> None:
    out = tmp_path / "d"
    with pytest.raises(shards.ShardError, match="duplicate"):
        shards.pack_shards([*_samples(3), *_samples(1)], out, max_bytes=8000)
    assert not list(out.glob("*.tar")) and not (out / shards.INDEX_NAME).exists()


def test_feature_schema_and_row_count(tmp_path: Path) -> None:
    eps = []
    for seed in (1, 2):
        ep = generate_episode(seed, GeometryConfig(n_views=(3, 3), n_instances=(8, 8)))
        p = tmp_path / f"ep_{seed:06d}.npz"
        save_episode(ep, p)
        eps.append((f"scene_{seed:06d}", "train", p, ep))
    out = tmp_path / "f.parquet"
    n = features.write_features(
        [(s, sp, p) for s, sp, p, _ in eps], out, recipe_version="1.0", data_version="abc123"
    )
    assert n == sum(e.n_detections for *_, e in eps)
    table = pq.read_table(out)
    assert table.num_rows == n
    assert tuple(table.schema.names) == features.COLUMNS
    assert table.schema.field("embedding").type == pa.list_(pa.float32())
    assert features.feature_store_version(out) == "abc123"
    rows = table.to_pydict()
    assert set(rows["data_version"]) == {"abc123"} and set(rows["seed"]) == {1, 2}
    assert all(e is None for e in rows["embedding"])
    first = eps[0][3]
    assert rows["gt_instance_id"][: first.n_detections] == first.identity.tolist()
    # lifted xyz of a true detection lands near its GT 3D point (pixel + depth noise only)
    i = int(np.flatnonzero(first.identity >= 0)[0])
    xyz = np.array([rows[c][i] for c in "xyz"])
    assert np.linalg.norm(xyz - first.points3d[first.identity[i]]) < 1.0


def test_feature_embeddings(tmp_path: Path) -> None:
    ep = generate_episode(3, GeometryConfig(n_views=(3, 3), n_instances=(5, 5)))
    p = tmp_path / "ep_000003.npz"
    save_episode(ep, p)
    out = tmp_path / "f.parquet"
    features.write_features(
        [("scene_000003", "val", p)],
        out,
        recipe_version="1.0",
        data_version="v",
        embed=lambda e: np.ones((e.n_detections, 4)),
    )
    assert pq.read_table(out).to_pydict()["embedding"][0] == [1.0] * 4
    with pytest.raises(ValueError, match="embedding"):
        features.write_features(
            [("scene_000003", "val", p)],
            out,
            recipe_version="1.0",
            data_version="v",
            embed=lambda e: np.ones((2, 4)),
        )


def _rows() -> list[dict]:
    out = []
    for seed, (n, arr) in enumerate(
        [(10, "canopy"), (90, "canopy"), (95, "piled"), (120, "canopy")]
    ):
        r = make_recipe(seed, n_objects=n, n_views=4, arrangement=arr)
        out.append(
            catalog.scene_row(
                r,
                scene_id=f"scene_{seed:06d}",
                split="train",
                renderer="proxy",
                data_version="v1",
                quarantine_reasons=["all_black_frame"] if seed == 3 else (),
            )
        )
    return out


def test_catalog_slice_query(tmp_path: Path) -> None:
    path = tmp_path / "scenes.parquet"
    assert catalog.write_scenes(_rows(), path) == 4
    con = catalog.connect(path)
    got = catalog.query_slice(con, min_objects=80, arrangement="canopy")
    assert [r["scene_id"] for r in got] == ["scene_000001"]  # 3 is quarantined, 2 is piled
    both = catalog.query_slice(con, min_objects=80, include_quarantined=True, arrangement="canopy")
    assert [r["scene_id"] for r in both] == ["scene_000001", "scene_000003"]
    assert catalog.query_slice(con, min_baseline=1e9) == []
    cols, rows = catalog.run_query(con, "SELECT count(*) AS n FROM scenes WHERE quarantined")
    assert cols == ["n"] and rows == [(1,)]
    with pytest.raises(catalog.CatalogError):
        catalog.connect(tmp_path / "missing.parquet")


def _dag_cfg(tmp_path: Path) -> DagConfig:
    return DagConfig(
        work_dir=str(tmp_path / "dag"),
        splits={"train": 4, "val": 2, "test": 2},
        recipe={"n_objects": 6, "n_views": 3},
        train={"epochs": 1},
        shards={"max_bytes": 6000},
    )


def _noop_train(dag: Dag) -> None:
    from synth.dag import _feature_store_version

    d = dag.stage_dir("train")
    (d / "lineage.json").write_text(
        json.dumps({"feature_store_version": _feature_store_version(dag)})
    )


def test_dag_lineage_shard_features_catalog(tmp_path: Path) -> None:
    def quarantine_one(recipe):  # seed 1 is all-black, so it is quarantined
        scene = proxy_scene(recipe)
        if recipe.seed == 1:
            scene.images[:] = 0
        return scene

    dag = Dag(
        _dag_cfg(tmp_path),
        scene_loader=quarantine_one,
        executors={"train": _noop_train, "evaluate": lambda d: d.stage_dir("evaluate")},
    )
    assert set(dag.run().values()) == {"ran"}
    work = dag.work
    index = json.loads((work / "shard" / "index.json").read_text())
    assert index["n_samples"] == 7 and len(index["shards"]) > 1  # 8 scenes, 1 quarantined
    keys = {s.key for s in shards.read_shards(work / "shard")}
    assert "scene_000001" not in keys

    feat = work / "features" / "features.parquet"
    stamp = dag.read_stamp("features")
    assert stamp and stamp["dataset_version"] == features.feature_store_version(feat)
    n_rows = json.loads((work / "features" / "summary.json").read_text())["n_rows"]
    assert pq.read_table(feat).num_rows == n_rows > 0

    con = catalog.connect(work / "catalog" / "scenes.parquet", feat)
    assert catalog.run_query(con, "SELECT count(*) FROM scenes")[1] == [(8,)]
    assert len(catalog.query_slice(con)) == 7  # quarantine status excludes seed 1
    assert catalog.run_query(con, "SELECT count(*) FROM detections")[1] == [(n_rows,)]
    q = catalog.run_query(con, "SELECT scene_id, quarantine_reasons FROM scenes WHERE quarantined")
    assert q[1] == [("scene_000001", "all_black_frame")]

    lineage = json.loads((work / "train" / "lineage.json").read_text())
    assert lineage["feature_store_version"] == features.feature_store_version(feat)

    # second run is a no-op; a shard-size change reruns only shard (nothing downstream reads it)
    assert set(dag.run().values()) == {"skipped"}
    cfg = _dag_cfg(tmp_path)
    cfg.shards = {"max_bytes": 1 << 20}
    res = Dag(cfg, scene_loader=quarantine_one, executors=dag.executors).run()
    assert res["shard"] == "ran"


def test_cli_catalog_query(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from countonce.cli import main

    catalog.write_scenes(_rows(), tmp_path / "catalog" / "scenes.parquet")
    argv = ["countonce", "catalog", "query", "SELECT scene_id FROM scenes WHERE n_objects > 100"]
    argv += ["--dir", str(tmp_path)]
    old, sys.argv = sys.argv, argv
    try:
        assert main(argv[1:]) == 0
    finally:
        sys.argv = old
    assert "scene_000003" in capsys.readouterr().out
    assert main(["catalog", "query", "SELEC nope", "--dir", str(tmp_path)]) == 2


def test_train_records_feature_store_version(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from countonce.match.model import checkpoint_meta
    from countonce.match.train import TrainConfig, train

    cfg = _dag_cfg(tmp_path)
    cfg.train = {"epochs": 1, "model": {"width": 16, "layers": 1, "heads": 2}}
    dag = Dag(cfg, executors={"evaluate": lambda d: d.stage_dir("evaluate")})
    dag.run()
    best = dag.work / "train" / "run" / "best.pt"
    assert checkpoint_meta(best)["feature_store_version"] == features.feature_store_version(
        dag.work / "features" / "features.parquet"
    )
    assert TrainConfig().feature_store == "" and train is not None
