"""Rerunnable synthetic-data DAG with lineage (PLAN §16B.1, pipeline orchestration and versioning).

Stages: ``recipe`` -> ``validate`` (quarantine) -> ``shard`` (WebDataset tars) -> ``episodes`` ->
``features`` (Parquet) -> ``catalog`` (DuckDB-queryable) -> ``train`` -> ``evaluate``.

Each stage writes its outputs under ``<work_dir>/<stage>/`` plus ``<work_dir>/stamps/<stage>.json``
recording the input hashes, the config-slice hash, the recipe version, the seed, the git sha and
the output hashes. A stage is skipped when its stamp matches the current key and its outputs still
hash the same. The key uses the upstream *output* hash, so a rerun that reproduces identical output
does not cascade. The git sha is recorded but is not part of the key (every commit would rerun
everything). A scene quarantined by ``validate`` never reaches ``episodes``. The dataset version is
a content hash of the episode set; ``train`` stores it in the checkpoint and in ``lineage.json``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from synth.geometry import GeometryConfig, generate_episode, save_episode
from synth.recipe import RECIPE_VERSION, SceneRecipe, make_recipe
from synth.validate import QuarantineReport, RenderedScene, validate_scene

STAGES = ("recipe", "validate", "shard", "episodes", "features", "catalog", "train", "evaluate")
_RECIPE_KEYS = {"n_objects", "n_views", "arrangement", "rig_kind"}

SceneLoader = Callable[[SceneRecipe], RenderedScene]
Executor = Callable[["Dag"], None]


class DagError(ValueError):
    pass


@dataclass
class DagConfig:
    work_dir: str = "data/dag"
    seed: int = 0
    splits: dict[str, int] = field(default_factory=lambda: {"train": 40, "val": 8, "test": 8})
    recipe: dict[str, Any] = field(default_factory=dict)  # pins passed to make_recipe
    geometry: dict[str, Any] = field(default_factory=dict)  # GeometryConfig fields (not counts)
    train: dict[str, Any] = field(default_factory=dict)  # TrainConfig overrides
    ablate: dict[str, Any] = field(default_factory=dict)  # AblationConfig overrides; {} = skip
    shards: dict[str, Any] = field(default_factory=dict)  # max_bytes (default 1 GiB per shard)

    @classmethod
    def from_yaml(cls, path: Path | str) -> DagConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        bad = sorted(set(raw) - {f.name for f in fields(cls)})
        if bad:
            raise DagError(f"unknown dag config keys: {bad}")
        cfg = cls(**raw)
        cfg.check()
        return cfg

    def check(self) -> None:
        if {"train", "val", "test"} - set(self.splits):
            raise DagError("splits must define train, val and test")
        bad = sorted(set(self.recipe) - _RECIPE_KEYS)
        if bad:
            raise DagError(f"unknown recipe pins: {bad}")
        bad = sorted(set(self.shards) - {"max_bytes"})
        if bad:
            raise DagError(f"unknown shards keys: {bad}")

    def slice(self, stage: str) -> dict[str, Any]:
        """The config values a stage depends on (changing anything else leaves it fresh)."""
        if stage == "recipe":
            return {"splits": self.splits, "recipe": self.recipe}
        if stage == "validate":
            return {}
        if stage == "shard":
            return {"shards": self.shards}
        if stage in ("features", "catalog"):
            return {}
        if stage == "episodes":
            return {"geometry": self.geometry}
        if stage == "train":
            return {"train": self.train}
        return {"ablate": self.ablate, "train": {"threshold": self.train.get("match_threshold")}}


# ---------------------------------------------------------------- hashing helpers


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(obj: Any) -> str:
    return _sha(json.dumps(obj, sort_keys=True, default=str).encode())[:16]


def hash_files(root: Path, files: list[Path] | None = None) -> dict[str, str]:
    """{relative path: sha256 prefix} for `files` (default: every file under `root`)."""
    paths = files if files is not None else sorted(p for p in root.rglob("*") if p.is_file())
    return {str(p.relative_to(root)): _sha(p.read_bytes())[:16] for p in sorted(paths)}


def _combine(hashes: dict[str, str]) -> str:
    return hash_json(sorted(hashes.items()))


def dataset_version(episode_dirs: list[Path]) -> str:
    """Content hash of the episode set in `episode_dirs` (names and bytes)."""
    h = hashlib.sha256()
    for d in episode_dirs:
        for p in sorted(d.glob("*.npz")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _git_sha() -> str | None:
    from countonce.provenance import git_commit

    return git_commit()


# ---------------------------------------------------------------- scene proxy


def proxy_scene(recipe: SceneRecipe) -> RenderedScene:
    """Renderer-free stand-in: tiny constant frames, the target visible in every view and the
    recipe's real camera centres. Real renders plug in through `Dag(scene_loader=...)`."""
    v = len(recipe.rig.cameras)
    meta = {
        "scene_id": f"scene_{recipe.seed:06d}",
        "recipe_version": recipe.recipe_version,
        "seed": recipe.seed,
        "renderer": "proxy",
        "prompt": recipe.prompt,
        "n_objects": recipe.n_objects,
        "n_views": v,
        "target_ids": [1],
        "object_positions": [o.position for o in recipe.objects],
        "resolution": (4, 4),
    }
    return RenderedScene(
        metadata=meta,
        images=np.full((v, 4, 4, 3), 128, dtype=np.uint8),
        depth=np.ones((v, 4, 4)),
        instance_masks=np.ones((v, 4, 4), dtype=int),
        camera_centers=np.array([c.position for c in recipe.rig.cameras], dtype=float),
    )


# ---------------------------------------------------------------- stage executors


def _split_seeds(cfg: DagConfig) -> dict[str, list[int]]:
    out, offset = {}, 0
    for split, n in cfg.splits.items():
        out[split] = [cfg.seed + offset + i for i in range(int(n))]
        offset += int(n)
    return out


def _exec_recipe(dag: Dag) -> None:
    out = dag.stage_dir("recipe")
    for split, seeds in _split_seeds(dag.cfg).items():
        for s in seeds:
            r = make_recipe(s, **dag.cfg.recipe)
            p = out / split / f"recipe_{s:06d}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(r.to_json())


def _exec_validate(dag: Dag) -> None:
    rdir, out = dag.stage_dir("recipe", create=False), dag.stage_dir("validate")
    report = QuarantineReport()
    clean: list[str] = []
    renderers: dict[str, str] = {}
    for p in sorted(rdir.glob("*/recipe_*.json")):
        key = f"{p.parent.name}/{p.stem}"
        scene = dag.scene_loader(SceneRecipe.from_json(p.read_text()))
        renderers[key] = str(scene.metadata.get("renderer", "unknown"))
        reasons = validate_scene(scene)
        report.n_scenes += 1
        if reasons:
            report.quarantined[key] = reasons
        else:
            clean.append(key)
    report.write(out)
    (out / "clean.json").write_text(json.dumps({"clean": clean}, indent=2))
    (out / "renderers.json").write_text(json.dumps(renderers, indent=2, sort_keys=True))


def scene_id_for(recipe: SceneRecipe) -> str:
    """Stable id shared by shards, features and the catalog (renderer metadata may differ)."""
    return f"scene_{recipe.seed:06d}"


def _need_data_stack() -> None:
    try:
        import duckdb  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise DagError("the shard/features/catalog stages need: uv pip install -e .[data]") from exc


def _exec_shard(dag: Dag) -> None:
    from synth.shards import pack_shards, scene_sample

    out = dag.stage_dir("shard")
    verdict = json.loads((dag.stage_dir("validate", create=False) / "clean.json").read_text())

    def samples() -> Iterator[Any]:
        for key in verdict["clean"]:
            recipe = SceneRecipe.from_json(
                (dag.stage_dir("recipe", create=False) / f"{key}.json").read_text()
            )
            sid = scene_id_for(recipe)
            yield scene_sample(
                sid,
                dag.scene_loader(recipe),
                {"split": key.split("/")[0], "scene_id": sid, "recipe_version": RECIPE_VERSION},
            )

    pack_shards(samples(), out, max_bytes=int(dag.cfg.shards.get("max_bytes", 1 << 30)))


def _exec_episodes(dag: Dag) -> None:
    out = dag.stage_dir("episodes")
    verdict = json.loads((dag.stage_dir("validate", create=False) / "clean.json").read_text())
    geo = {k: (tuple(v) if isinstance(v, list) else v) for k, v in dag.cfg.geometry.items()}
    for key in verdict["clean"]:
        split = key.split("/")[0]
        recipe = SceneRecipe.from_json(
            (dag.stage_dir("recipe", create=False) / f"{key}.json").read_text()
        )
        gcfg = GeometryConfig(
            **{
                **geo,
                "n_views": (len(recipe.rig.cameras),) * 2,
                "n_instances": (recipe.n_objects,) * 2,
            }  # type: ignore[arg-type]
        )
        path = out / split / f"ep_{recipe.seed:06d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_episode(generate_episode(recipe.seed, gcfg), path)


def _episode_triples(dag: Dag) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for split in ("train", "val", "test"):
        for path in sorted((dag.work / "episodes" / split).glob("ep_*.npz")):
            out.append((f"scene_{int(path.stem[3:]):06d}", split, path))
    return out


def _episodes_version(dag: Dag) -> str:
    return dataset_version([dag.work / "episodes" / s for s in ("train", "val", "test")])


def _exec_features(dag: Dag) -> None:
    from synth.features import write_features

    _need_data_stack()
    out = dag.stage_dir("features")
    n = write_features(
        _episode_triples(dag),
        out / "features.parquet",
        recipe_version=RECIPE_VERSION,
        data_version=_episodes_version(dag),
    )
    (out / "summary.json").write_text(json.dumps({"n_rows": n}, indent=2))


def _exec_catalog(dag: Dag) -> None:
    from synth.catalog import scene_row, write_scenes

    _need_data_stack()
    out = dag.stage_dir("catalog")
    vdir = dag.stage_dir("validate", create=False)
    quarantined = json.loads((vdir / "quarantine.json").read_text())["quarantined"]
    renderers = json.loads((vdir / "renderers.json").read_text())
    version = _episodes_version(dag)
    rows = []
    for p in sorted(dag.stage_dir("recipe", create=False).glob("*/recipe_*.json")):
        key = f"{p.parent.name}/{p.stem}"
        recipe = SceneRecipe.from_json(p.read_text())
        rows.append(
            scene_row(
                recipe,
                scene_id=scene_id_for(recipe),
                split=p.parent.name,
                renderer=renderers.get(key, "unknown"),
                data_version=version,
                quarantine_reasons=quarantined.get(key, []),
            )
        )
    write_scenes(rows, out / "scenes.parquet")


def _feature_store_path(dag: Dag) -> Path:
    return dag.work / "features" / "features.parquet"


def _train_config(dag: Dag) -> Any:
    from countonce.match.train import TrainConfig

    work = dag.work
    known = {f.name for f in fields(TrainConfig)}
    bad = sorted(set(dag.cfg.train) - known)
    if bad:
        raise DagError(f"unknown train keys: {bad}")
    base = {
        "train_dir": str(work / "episodes" / "train"),
        "val_dir": str(work / "episodes" / "val"),
        "out_dir": str(work / "train" / "run"),
        "results_csv": str(work / "train" / "history.csv"),
        "seed": dag.cfg.seed,
        "resume": False,
        "device": "cpu",
        "data_version": dataset_version([work / "episodes" / "train", work / "episodes" / "val"]),
        "feature_store": str(_feature_store_path(dag)),
    }
    return TrainConfig(**{**base, **dag.cfg.train})


def _feature_store_version(dag: Dag) -> str | None:
    path = _feature_store_path(dag)
    if not path.exists():
        return None
    from synth.features import feature_store_version

    return feature_store_version(path)


def _exec_train(dag: Dag) -> None:
    from countonce.match.train import train

    tcfg = _train_config(dag)
    out = Path(tcfg.out_dir)
    shutil.rmtree(out, ignore_errors=True)
    history = train(tcfg)
    best = out / "best.pt"
    lineage = {
        "data_version": tcfg.data_version,
        "feature_store_version": _feature_store_version(dag),
        "checkpoint": "run/best.pt",
        "checkpoint_sha256": _sha(best.read_bytes()) if best.exists() else None,
        "recipe_version": RECIPE_VERSION,
        "seed": dag.cfg.seed,
        "epochs_logged": len(history),
    }
    (dag.work / "train" / "lineage.json").write_text(json.dumps(lineage, indent=2))


def _exec_evaluate(dag: Dag) -> None:
    from countonce.match.model import load_checkpoint
    from countonce.match.train import evaluate, load_split

    tcfg = _train_config(dag)
    out = dag.stage_dir("evaluate")
    lineage = json.loads((dag.work / "train" / "lineage.json").read_text())
    model = load_checkpoint(Path(tcfg.out_dir) / "best.pt")
    adim = int(tcfg.model.get("appearance_dim", 0))
    test = load_split(dag.work / "episodes" / "test", tcfg.feature_group, adim)
    metrics = evaluate(model, test, tcfg)
    record = {
        "metrics": metrics,
        "n_test_episodes": len(test),
        "train_data_version": lineage["data_version"],
        "feature_store_version": lineage.get("feature_store_version"),
        "test_data_version": dataset_version([dag.work / "episodes" / "test"]),
        "checkpoint_sha256": lineage["checkpoint_sha256"],
    }
    if dag.cfg.ablate:
        from bench.synth_ablation import AblationConfig, run_ablation

        ab = AblationConfig(
            **{
                "train_dir": str(dag.work / "episodes" / "train"),
                "val_dir": str(dag.work / "episodes" / "val"),
                "test_dir": str(dag.work / "episodes" / "test"),
                "work_dir": str(out / "ablation"),
                "out_csv": str(out / "ablation.csv"),
                "per_seed_csv": str(out / "ablation_per_seed.csv"),
                **dag.cfg.ablate,
            }
        )
        summary, per_seed = run_ablation(ab)
        record["ablation"] = {"summary": summary.name, "per_seed": per_seed.name}
    (out / "eval.json").write_text(json.dumps(record, indent=2))


EXECUTORS: dict[str, Executor] = {
    "recipe": _exec_recipe,
    "validate": _exec_validate,
    "shard": _exec_shard,
    "features": _exec_features,
    "catalog": _exec_catalog,
    "episodes": _exec_episodes,
    "train": _exec_train,
    "evaluate": _exec_evaluate,
}

# run state that is not a deterministic output (logs embed git state; last.pt holds the optimiser)
_UNSTAMPED = {"train": ("run/log.jsonl", "run/last.pt"), "evaluate": ("ablation/",)}


# ---------------------------------------------------------------- the DAG


@dataclass
class StageStatus:
    stage: str
    state: str  # fresh | stale | missing
    reason: str = ""


class Dag:
    def __init__(
        self,
        cfg: DagConfig,
        *,
        scene_loader: SceneLoader = proxy_scene,
        executors: dict[str, Executor] | None = None,
    ) -> None:
        cfg.check()
        self.cfg = cfg
        self.work = Path(cfg.work_dir)
        self.scene_loader = scene_loader
        self.executors = {**EXECUTORS, **(executors or {})}

    # paths
    def stage_dir(self, stage: str, create: bool = True) -> Path:
        d = self.work / stage
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def stamp_path(self, stage: str) -> Path:
        return self.work / "stamps" / f"{stage}.json"

    def read_stamp(self, stage: str) -> dict[str, Any] | None:
        p = self.stamp_path(stage)
        return json.loads(p.read_text()) if p.exists() else None

    # keys
    def _key(self, stage: str, upstream: str | None) -> dict[str, Any]:
        return {
            "inputs": {"upstream": upstream, "recipe_version": RECIPE_VERSION},
            "config_hash": hash_json(self.cfg.slice(stage)),
            "recipe_version": RECIPE_VERSION,
            "seed": self.cfg.seed,
        }

    def _output_hashes(self, stage: str) -> dict[str, str]:
        root = self.stage_dir(stage, create=False)
        if not root.exists():
            return {}
        skip = _UNSTAMPED.get(stage, ())
        files = [
            p
            for p in root.rglob("*")
            if p.is_file() and not any(str(p.relative_to(root)).startswith(s) for s in skip)
        ]
        return hash_files(root, files)

    def _upstream_hash(self, stage: str, hashes: dict[str, str | None]) -> str | None:
        i = STAGES.index(stage)
        if i == 0:
            return None
        parts = {STAGES[j]: hashes.get(STAGES[j]) for j in range(i)}
        return hash_json(parts)

    # status
    def status(self) -> list[StageStatus]:
        out: list[StageStatus] = []
        hashes: dict[str, str | None] = {}
        broken = False
        for st in STAGES:
            stamp = self.read_stamp(st)
            if stamp is None:
                out.append(StageStatus(st, "missing", "never run"))
                broken = True
                continue
            if broken:
                out.append(StageStatus(st, "stale", "upstream stage is not fresh"))
                continue
            want = self._key(st, self._upstream_hash(st, hashes))
            have = {k: stamp.get(k) for k in want}
            if have != want:
                diff = [k for k in want if have[k] != want[k]]
                out.append(StageStatus(st, "stale", f"changed: {', '.join(diff)}"))
                broken = True
            elif self._output_hashes(st) != stamp.get("outputs"):
                out.append(StageStatus(st, "stale", "outputs modified or missing"))
                broken = True
            else:
                out.append(StageStatus(st, "fresh"))
                hashes[st] = stamp.get("output_hash")
        return out

    # run
    def run(self, from_stage: str | None = None, force: bool = False) -> dict[str, str]:
        """Run stale stages in order. Returns {stage: 'ran' | 'skipped'}."""
        if from_stage is not None and from_stage not in STAGES:
            raise DagError(f"unknown stage {from_stage!r}; stages are {', '.join(STAGES)}")
        forced = force
        result: dict[str, str] = {}
        hashes: dict[str, str | None] = {}
        for st in STAGES:
            if from_stage == st:
                forced = True
            stamp = self.read_stamp(st)
            key = self._key(st, self._upstream_hash(st, hashes))
            fresh = (
                not forced
                and stamp is not None
                and {k: stamp.get(k) for k in key} == key
                and self._output_hashes(st) == stamp.get("outputs")
            )
            if fresh and stamp is not None:
                result[st] = "skipped"
                hashes[st] = stamp.get("output_hash")
                continue
            self._clear(st)
            t0 = time.time()
            self.executors[st](self)
            outputs = self._output_hashes(st)
            new = {
                "stage": st,
                **key,
                "git_sha": _git_sha(),
                "outputs": outputs,
                "output_hash": _combine(outputs),
                "seconds": round(time.time() - t0, 3),
            }
            if st in ("episodes", "features", "catalog", "train", "evaluate"):
                new["dataset_version"] = dataset_version(
                    [self.work / "episodes" / s for s in ("train", "val", "test")]
                )
            self.stamp_path(st).parent.mkdir(parents=True, exist_ok=True)
            self.stamp_path(st).write_text(json.dumps(new, indent=2))
            hashes[st] = new["output_hash"]
            result[st] = "ran"
        return result

    def _clear(self, stage: str) -> None:
        shutil.rmtree(self.work / stage, ignore_errors=True)
        self.stamp_path(stage).unlink(missing_ok=True)


def format_status(rows: list[StageStatus]) -> str:
    return "\n".join(f"{r.stage:<9} {r.state:<8} {r.reason}".rstrip() for r in rows)
