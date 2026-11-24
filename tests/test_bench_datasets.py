"""Video / multi-view dataset loaders on tiny synthetic fixtures in the ASSUMED layouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from bench.datasets import load_dataset
from bench.datasets.common import subsample_frames, to_gt_doc, to_sweep_scenes
from bench.evaluate import load_gt
from bench.sweep import BenchConfig, build_jobs, dataset_scenes, discover_scenes
from countonce.ingest.frames import list_images

CFG = Path(__file__).parents[1] / "configs" / "bench.yaml"


def _frames(d: Path, n: int) -> None:
    """Real, mutually distinct tiny JPEGs so the default data-quality gates pass."""
    d.mkdir(parents=True)
    for i in range(n):
        arr = np.random.default_rng(abs(hash((str(d), i))) % 2**32).integers(
            0, 255, (8, 8, 3), dtype=np.uint8
        )
        Image.fromarray(arr).save(d / f"{i:04d}.jpg")


def _rgbd(root: Path) -> Path:
    for clip in ("clipA", "clipB"):
        _frames(root / "images" / clip, 30)
    (root / "count_annotations").mkdir()
    (root / "count_annotations" / "clipA.json").write_text(json.dumps({"can": 7, "box": 3}))
    (root / "count_annotations" / "clipB.json").write_text(json.dumps({"bottle": 5}))
    (root / "dataset_split.json").write_text(json.dumps({"train": ["clipA"], "test": ["clipB"]}))
    return root


def _videocount(root: Path) -> Path:
    _frames(root / "frames" / "pen1", 12)
    root.joinpath("annotations.json").write_text(
        json.dumps({"pen1": {"count": 9, "prompt": "penguin", "source": "penguins"}})
    )
    return root


def _onescene(root: Path) -> Path:
    _frames(root / "scenes" / "s1" / "images", 6)
    root.joinpath("counting.json").write_text(
        json.dumps(
            {"scenes": [{"scene_id": "s1", "category": "chair", "count": 4, "split": "test"}]}
        )
    )
    return root


def test_subsample_frames_even_and_deterministic():
    fr = [Path(f"{i}.jpg") for i in range(100)]
    a = subsample_frames(fr, 10)
    assert len(a) == 10 and a[0] == fr[0] and a[-1] == fr[-1]
    assert a == subsample_frames(fr, 10)
    assert len(subsample_frames(fr, None, stride=10)) == 10
    assert subsample_frames(fr[:3], 10) == fr[:3]


def test_rgbd_videocount_scenes_counts_and_subsampling(tmp_path):
    scenes = load_dataset("rgbd_videocount", _rgbd(tmp_path / "r"), max_frames=8)
    by = {s.scene_id: s for s in scenes}
    assert set(by) == {"rgbdvc_clipA_can", "rgbdvc_clipA_box", "rgbdvc_clipB_bottle"}
    assert by["rgbdvc_clipA_can"].count == 7 and by["rgbdvc_clipA_can"].prompt == "can"
    assert all(len(s.frames) == 8 for s in scenes)
    assert by["rgbdvc_clipB_bottle"].split == "test"


def test_videocount_and_onescene(tmp_path):
    (vc,) = load_dataset("videocount", _videocount(tmp_path / "v"), max_frames=5)
    assert vc.count == 9 and vc.prompt == "penguin" and len(vc.frames) == 5
    (os_,) = load_dataset("onescene", _onescene(tmp_path / "o"))
    assert os_.count == 4 and len(os_.frames) == 6 and os_.split == "test"


def test_missing_root_and_layout_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_onescene"):
        load_dataset("onescene", tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="TODO: verify"):
        load_dataset("rgbd_videocount", tmp_path / "empty")
    with pytest.raises(ValueError):
        load_dataset("bogus", tmp_path)


def test_adapters_feed_sweep_and_evaluate(tmp_path):
    scenes = load_dataset("rgbd_videocount", _rgbd(tmp_path / "r"), max_frames=8)
    out = tmp_path / "out"
    sw = to_sweep_scenes(scenes, out, split="test")
    assert [s["id"] for s in sw] == ["rgbdvc_clipB_bottle"]
    assert len(list_images(sw[0]["dir"])) == 8
    assert sw[0]["gt"]["count"] == 5 and sw[0]["gt"]["prompt"] == "bottle"
    assert [s["id"] for s in discover_scenes(out)] == ["rgbdvc_clipB_bottle"]
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(to_gt_doc(scenes, "test")))
    assert load_gt(gt)["scenes"] == {"rgbdvc_clipB_bottle": {"count": 5}}


def test_bench_config_opt_in(tmp_path):
    cfg = BenchConfig.from_yaml(CFG)
    assert [d.name for d in cfg.datasets] == ["rgbd_videocount", "videocount", "onescene"]
    assert not any(d.enabled for d in cfg.datasets)
    assert dataset_scenes(cfg) == []  # nothing loaded, no download needed

    doc = yaml.safe_load(CFG.read_text())
    doc["csv"] = str(tmp_path / "res" / "bench.csv")
    doc["split"] = "test"
    doc["datasets"] = [
        {"name": "onescene", "root": str(_onescene(tmp_path / "o")), "enabled": True},
    ]
    p = tmp_path / "bench.yaml"
    p.write_text(yaml.safe_dump(doc))
    cfg2 = BenchConfig.from_yaml(p)
    scenes = dataset_scenes(cfg2)
    assert [s["id"] for s in scenes] == ["onescene_s1"]
    assert build_jobs(cfg2, scenes)
