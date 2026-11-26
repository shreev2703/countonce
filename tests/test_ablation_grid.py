"""§9.6 ablation grid: every ablation expands to a valid Config, the new switches change
behaviour on crafted fixtures, and the v3 bench row is skipped (not failed) without a checkpoint."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from test_associators import SIZE, Z, _lifted, make_recon

from bench.runner import read_rows
from bench.sweep import BenchConfig, discover_scenes, run_bench
from countonce.config import Config
from countonce.lift import apply_lift_region
from countonce.merge.appearance import CachedExtractor
from countonce.merge.cluster import cluster_detections
from countonce.pipeline import _make_associator
from countonce.types import Detection, Frame

BENCH = Path(__file__).parents[1] / "configs" / "bench.yaml"


def test_every_ablation_expands_to_valid_config():
    cfg = BenchConfig.from_yaml(BENCH)
    names = {m.name for m in cfg.all_methods()}
    for expected in (
        "co_v2_vggt+cannot_link=False",
        "co_v2_vggt+appearance=False",
        "co_v2_vggt+hallucination=off",
        "co_v2_vggt+hallucination=drop",
        "co_v2_vggt+lift_region=box",
        "co_v2_vggt+assoc=v1",
        "co_v3_vggt",
    ):
        assert expected in names, expected
    for m in cfg.all_methods():
        if m.kind == "countonce":
            Config(**m.config)  # extra="forbid": a typo'd field would raise
    v3 = next(m for m in cfg.all_methods() if m.name == "co_v3_vggt")
    assert v3.config["assoc"] == "v3" and v3.config["v3_checkpoint"]


def _same_frame_pair():
    a = _lifted("0:0", 0, [0, 0, Z], 0.0)
    b = _lifted("0:1", 0, [0.1 * SIZE, 0, Z], 0.0)  # same frame, almost the same place
    return [a, b]


def test_cannot_link_off_merges_same_frame_v1_and_on_is_unviolated():
    lifted = _same_frame_pair()
    on = _make_associator(Config(assoc="v1"), 0.5).associate(lifted)
    off = _make_associator(Config(assoc="v1", cannot_link=False), 0.5).associate(lifted)
    assert len(on.instances) == 2
    assert all(len({m.split(":")[0] for m in i.members}) == len(i.members) for i in on.instances)
    assert len(off.instances) == 1 and off.instances[0].n_views == 1
    assert len(cluster_detections(lifted, 0.5).instances) == 2  # default stays on


def test_cannot_link_off_merges_same_frame_v2():
    recon = make_recon(2)
    lifted = [  # one object seen in frame 0, double-detected in frame 1
        _lifted("0:0", 0, [0, 0, Z], 0.0),
        _lifted("1:0", 1, [0, 0, Z], 0.2),
        _lifted("1:1", 1, [0.1 * SIZE, 0, Z], 0.2),
    ]
    on = _make_associator(Config(assoc="v2"), 0.5).associate(lifted, recon)
    off = _make_associator(Config(assoc="v2", cannot_link=False), 0.5).associate(lifted, recon)
    assert len(on.instances) == 2
    for inst in on.instances:
        frames = [m.split(":")[0] for m in inst.members]
        assert len(frames) == len(set(frames))
    assert len(off.instances) == 1 and len(off.instances[0].members) == 3


def test_hallucination_toggle_changes_count_on_one_view_ghost():
    recon = make_recon(4)
    real = [_lifted(f"{v}:0", v, [-3, 0, Z], 0.2 * v) for v in range(4)]
    lifted = [*real, _lifted("0:1", 0, [3, 0, Z], 0.0)]  # visible in all views, seen in one
    counts = {
        mode: len(
            _make_associator(Config(assoc="v2", hallucination=mode), 0.5)
            .associate(lifted, recon)
            .instances
        )
        for mode in ("off", "downweight", "drop")
    }
    assert counts == {"off": 2, "downweight": 2, "drop": 1}


def test_appearance_off_ignores_features():
    recon = make_recon(2)
    a = _lifted("0:0", 0, [0, 0, Z], 0.0)
    b = _lifted("1:0", 1, [0, 0, Z], 0.2)
    a.det.feature = np.array([1.0, 0.0])
    b.det.feature = np.array([-1.0, 0.0])  # opposite embeddings
    cfg = {"assoc": "v2", "hallucination": "off"}
    on = _make_associator(Config(**cfg), 0.5).associate([a, b], recon)
    off = _make_associator(Config(**cfg, appearance=False), 0.5).associate([a, b], recon)
    assert len(on.instances) == 2 and len(off.instances) == 1


def test_lift_region_box_drops_masks_and_records_fallbacks():
    box = np.array([10.0, 10.0, 30.0, 30.0])
    mask = np.zeros((50, 50), bool)
    mask[10:30, 10:30] = True
    with_both = Detection("0:0", 0, 0.9, box_xyxy=box, mask=mask)
    mask_only = Detection("0:1", 0, 0.9, mask=mask)
    box_only = Detection("0:2", 0, 0.9, box_xyxy=box)
    dets = [with_both, mask_only, box_only]

    out, fb = apply_lift_region(dets, "box")
    assert out[0].mask is None and out[1].mask is not None and out[2].mask is None
    assert fb == 1  # mask-only detection cannot be box-lifted
    out, fb = apply_lift_region(dets, "mask")
    assert out == dets and fb == 1  # the box-only detection falls back to its box
    assert replace(with_both, mask=None).box_xyxy is not None


def test_v3_row_skipped_cleanly_in_e2e_sweep(tmp_path):
    from test_pipeline import OBJECTS, PlaneRecon, make_scene

    from countonce.detect.base import FakeDetector

    src = tmp_path / "src"
    src.mkdir()
    img_dir, script = make_scene(src)
    scene = tmp_path / "scenes" / "s1"
    scene.mkdir(parents=True)
    shutil.move(str(img_dir), scene / "images")
    (scene / "gt.json").write_text(json.dumps({"count": len(OBJECTS), "tier": "easy"}))
    cfg = BenchConfig(
        scenes_dir=tmp_path / "scenes",
        csv=tmp_path / "bench.csv",
        runs_dir=tmp_path / "runs",
        cache_dir=None,
        views=[4],
        order=["ordered"],
        random_samples=0,
        even_sample=True,
        methods=[
            {"name": "co_v1_colmap", "kind": "countonce",
             "config": {"assoc": "v1", "backbone": "colmap", "max_frames": 0}},
            {"name": "co_v3_vggt", "kind": "countonce",
             "config": {"assoc": "v3", "backbone": "vggt",
                        "v3_checkpoint": str(tmp_path / "missing.pt")}},
        ],
    )  # fmt: skip
    ran, skipped, errored = run_bench(
        cfg,
        discover_scenes(cfg.scenes_dir),
        detector_factory=lambda conf: FakeDetector(script),
        reconstructor_factory=lambda conf: PlaneRecon(),
    )
    assert (ran, skipped, errored) == (1, 1, 0)
    status = {r["method"]: r["status"] for r in read_rows(cfg.csv)}
    assert status == {"co_v1_colmap": "ok", "co_v3_vggt": "skipped_no_checkpoint"}


def test_appearance_model_wires_a_shared_extractor():
    off = _make_associator(Config(assoc="v2"), 0.5)
    on = _make_associator(Config(assoc="v2", appearance_model="dinov2_vits14"), 0.5)
    again = _make_associator(Config(assoc="v2", appearance_model="dinov2_vits14"), 1.0)
    gated = Config(assoc="v2", appearance_model="dinov2_vits14", appearance=False)
    assert off.extractor is None
    assert on.extractor is not None and on.extractor is again.extractor
    assert _make_associator(gated, 0.5).extractor is None


def test_cached_extractor_embeds_each_crop_once(tmp_path):
    seen: list[str] = []

    class Fake:
        def extract(self, frame, dets):
            seen.extend(d.id for d in dets)
            return np.array([[d.box_xyxy[0], 1.0] for d in dets])

    frame = Frame(id=0, path=tmp_path / "a.jpg", width=10, height=10, sha256="abc")
    a, b = (
        Detection(id=n, frame_id=0, score=1.0, box_xyxy=np.array([x, 0.0, x + 1, 1.0]))
        for n, x in (("a", 1.0), ("b", 2.0))
    )
    ex = CachedExtractor(Fake())
    ex.extract(frame, [a])
    out = ex.extract(frame, [b, a])
    assert seen == ["a", "b"]
    assert out[:, 0].tolist() == [2.0, 1.0]
