from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from synth.dag import STAGES, Dag, DagConfig, DagError, dataset_version, proxy_scene
from synth.recipe import SceneRecipe


def _cfg(tmp_path: Path, **kw) -> DagConfig:
    base = dict(
        work_dir=str(tmp_path / "dag"),
        splits={"train": 4, "val": 2, "test": 2},
        recipe={"n_objects": 6, "n_views": 3},
        train={"epochs": 1},
    )
    return DagConfig(**{**base, **kw})


def _counting(calls: list[str]):
    def mk(stage: str):
        def run(dag: Dag) -> None:
            calls.append(stage)
            d = dag.stage_dir(stage)
            (d / "out.txt").write_text(json.dumps(dag.cfg.slice(stage), sort_keys=True))

        return run

    return {"train": mk("train"), "evaluate": mk("evaluate")}


def test_second_run_is_noop(tmp_path: Path) -> None:
    calls: list[str] = []
    dag = Dag(_cfg(tmp_path), executors=_counting(calls))
    first = dag.run()
    assert set(first.values()) == {"ran"}
    calls.clear()
    second = Dag(_cfg(tmp_path), executors=_counting(calls)).run()
    assert set(second.values()) == {"skipped"} and calls == []
    assert all(r.state == "fresh" for r in dag.status())
    stamp = dag.read_stamp("episodes")
    assert stamp and stamp["seed"] == 0 and stamp["recipe_version"]
    assert {"inputs", "config_hash", "outputs", "git_sha"} <= set(stamp)


def test_config_change_reruns_only_downstream(tmp_path: Path) -> None:
    calls: list[str] = []
    Dag(_cfg(tmp_path), executors=_counting(calls)).run()
    calls.clear()
    changed = _cfg(tmp_path, train={"epochs": 2})
    dag = Dag(changed, executors=_counting(calls))
    assert [r.state for r in dag.status()] == ["fresh"] * 6 + ["stale"] * 2
    res = dag.run()
    assert res == {
        "recipe": "skipped",
        "validate": "skipped",
        "shard": "skipped",
        "episodes": "skipped",
        "features": "skipped",
        "catalog": "skipped",
        "train": "ran",
        "evaluate": "ran",
    }
    assert calls == ["train", "evaluate"]


def test_geometry_change_reruns_episodes_onward(tmp_path: Path) -> None:
    Dag(_cfg(tmp_path), executors=_counting([])).run()
    res = Dag(_cfg(tmp_path, geometry={"pixel_noise": 3.0}), executors=_counting([])).run()
    assert [s for s in STAGES if res[s] == "ran"] == [
        "episodes",
        "features",
        "catalog",
        "train",
        "evaluate",
    ]


def test_from_and_force(tmp_path: Path) -> None:
    Dag(_cfg(tmp_path), executors=_counting([])).run()
    dag = Dag(_cfg(tmp_path), executors=_counting([]))
    res = dag.run(from_stage="train")
    assert [s for s in STAGES if res[s] == "ran"] == ["train", "evaluate"]
    assert set(dag.run(force=True).values()) == {"ran"}
    with pytest.raises(DagError):
        dag.run(from_stage="nope")


def test_tampered_output_is_detected(tmp_path: Path) -> None:
    dag = Dag(_cfg(tmp_path), executors=_counting([]))
    dag.run()
    victim = next((dag.work / "episodes" / "train").glob("*.npz"))
    victim.write_bytes(b"x")
    assert dag.status()[3].state == "stale"
    assert dag.run()["episodes"] == "ran"


def test_quarantined_scene_never_reaches_episodes(tmp_path: Path) -> None:
    bad_seed = 1

    def loader(recipe: SceneRecipe):
        scene = proxy_scene(recipe)
        if recipe.seed == bad_seed:
            scene.images[:] = 0  # all-black frames
        return scene

    dag = Dag(_cfg(tmp_path), scene_loader=loader, executors=_counting([]))
    dag.run()
    q = json.loads((dag.work / "validate" / "quarantine.json").read_text())
    assert list(q["quarantined"]) == ["train/recipe_000001"]
    names = {p.name for p in (dag.work / "episodes").rglob("*.npz")}
    assert "ep_000001.npz" not in names
    assert len(names) == 7  # 8 scenes, 1 quarantined


def test_proxy_degenerate_rig_quarantined(tmp_path: Path) -> None:
    def loader(recipe: SceneRecipe):
        scene = proxy_scene(recipe)
        scene.camera_centers[:] = scene.camera_centers[0]
        return scene

    dag = Dag(_cfg(tmp_path), scene_loader=loader, executors=_counting([]))
    dag.run()
    assert not list((dag.work / "episodes").rglob("*.npz"))


def test_unknown_config_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("bogus: 1\n")
    with pytest.raises(DagError):
        DagConfig.from_yaml(p)


def test_checkpoint_records_dataset_version(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from countonce.match.model import checkpoint_meta

    cfg = _cfg(
        tmp_path,
        train={"epochs": 1, "batch_size": 4, "model": {"width": 16, "layers": 1, "heads": 2}},
    )
    dag = Dag(cfg)
    assert dag.run()["evaluate"] == "ran"
    lineage = json.loads((dag.work / "train" / "lineage.json").read_text())
    version = dataset_version([dag.work / "episodes" / s for s in ("train", "val")])
    assert lineage["data_version"] == version
    assert checkpoint_meta(dag.work / "train" / "run" / "best.pt")["data_version"] == version
    ev = json.loads((dag.work / "evaluate" / "eval.json").read_text())
    assert ev["train_data_version"] == version
    assert set(Dag(cfg).run().values()) == {"skipped"}
