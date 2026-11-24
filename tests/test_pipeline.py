"""End-to-end CPU integration test on a generated synthetic scene (PLAN section 18)."""

import json

import numpy as np
import pytest
from PIL import Image

from countonce import schemas
from countonce.cli import main
from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.pipeline import run
from countonce.types import Frame, Recon

W, H, F, DEPTH = 64, 48, 50.0, 5.0
OBJECTS = np.array([[-1.0, -0.5], [0.0, 0.0], [1.2, 0.4], [0.6, -0.8], [-0.7, 0.7]])  # x, y
CAM_X = [0.0, 0.3, 0.6, 0.9]


class PlaneRecon:
    """Fake reconstructor: every view looks at the plane z=DEPTH, cameras slide along x."""

    name = "fake"

    def config(self):
        return {}

    def reconstruct(self, frames: list[Frame]) -> Recon:
        n = len(frames)
        k = np.tile(np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]]), (n, 1, 1))
        ext = np.tile(np.eye(3, 4), (n, 1, 1))
        wp = np.zeros((n, H, W, 3))
        ys, xs = np.mgrid[0:H, 0:W]
        for i in range(n):
            cx = CAM_X[i]
            wp[i, ..., 0] = (xs + 0.5 - W / 2) * DEPTH / F + cx
            wp[i, ..., 1] = (ys + 0.5 - H / 2) * DEPTH / F
            wp[i, ..., 2] = DEPTH
            ext[i, 0, 3] = -cx
        return Recon(
            backbone="fake",
            intrinsics=k,
            extrinsics=ext,
            depth=np.full((n, H, W), DEPTH),
            depth_conf=np.ones((n, H, W)),
            world_points=wp,
            world_points_conf=np.full((n, H, W), 2.0),
        )


def make_scene(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(len(CAM_X)):
        arr = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"f{i}.png")
    script = {}
    for i, cx in enumerate(CAM_X):
        boxes = []
        for ox, oy in OBJECTS:
            u = F * (ox - cx) / DEPTH + W / 2
            v = F * oy / DEPTH + H / 2
            if 4 <= u < W - 4 and 4 <= v < H - 4:
                boxes.append((u - 3, v - 3, u + 3, v + 3, 0.9))
        script[i] = boxes
    return img_dir, script


def test_end_to_end_counts_each_object_once(tmp_path):
    img_dir, script = make_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "out", cache_dir=tmp_path / "cache", max_frames=0)
    res = run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    total_dets = sum(len(v) for v in script.values())
    assert total_dets > len(OBJECTS)  # the scene really has duplicates to remove
    assert res.count == len(OBJECTS)
    assert not res.recon_cache_hit

    raw = json.loads((tmp_path / "out" / "result.json").read_text())
    loaded = schemas.load_result(raw)
    assert loaded.count == len(OBJECTS)
    assert "git_commit" in loaded.config and loaded.config["config"]["assoc"] == "v1"
    for name in ("markers.json", "counts.csv"):
        assert (tmp_path / "out" / name).exists()

    again = run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    assert again.recon_cache_hit and again.count == res.count


def test_zero_detections_gives_zero_count(tmp_path):
    img_dir, _ = make_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "out", max_frames=0)
    res = run(img_dir, "nothing", cfg, detector=FakeDetector({}), reconstructor=PlaneRecon())
    assert res.count == 0 and res.zero_count


@pytest.mark.parametrize("assoc", ["v1", "v2"])
def test_empty_scene_counts_zero_with_interval_containing_zero(tmp_path, assoc):
    img_dir, _ = make_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "out", max_frames=0, assoc=assoc)
    res = run(img_dir, "nothing", cfg, detector=FakeDetector({}), reconstructor=PlaneRecon())
    assert res.count == 0 and res.zero_count and not res.instances
    assert res.interval.low == 0 and res.interval.low <= 0 <= res.interval.high
    assert res.interval.high >= res.count


def test_end_to_end_v2_counts_each_object_once(tmp_path):
    img_dir, script = make_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "out", max_frames=0, assoc="v2")
    res = run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    assert res.count == len(OBJECTS)
    assert any(e.merged and e.reproj_error_px >= 0 for e in res.match_evidence)


def test_v3_without_checkpoint_raises_clear_error(tmp_path):
    img_dir, _ = make_scene(tmp_path)
    with pytest.raises(FileNotFoundError, match="v3_checkpoint"):
        run(img_dir, "x", Config(assoc="v3", out_dir=tmp_path / "o"))


def test_config_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("detector: gdino\nfps: 4\nscore_threshold: 0.5\n")
    cfg = Config.from_yaml(p)
    assert (cfg.detector, cfg.fps, cfg.score_threshold) == ("gdino", 4.0, 0.5)
    assert cfg.with_overrides(fps=None, backbone="colmap").backbone == "colmap"
    p.write_text("bogus: 1\n")
    with pytest.raises(ValueError):
        Config.from_yaml(p)


def test_cli_reports_missing_v3_checkpoint_as_error(tmp_path, capsys):
    img_dir, _ = make_scene(tmp_path)
    code = main(["run", "--images", str(img_dir), "--prompt", "x", "--assoc", "v3",
                 "--out", str(tmp_path / "o")])  # fmt: skip
    assert code == 2
    assert "v3_checkpoint" in capsys.readouterr().err


def test_run_reports_sensitivity_interval_and_warnings(tmp_path):
    img_dir, script = make_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "out", max_frames=0)
    res = run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    assert res.confidence_method == "sensitivity" and res.interval.method == "sensitivity"
    assert res.interval.low <= res.count <= res.interval.high
    assert {"code", "message", "fix"} <= set(res.warnings[0])  # PlaneRecon depth_conf is 1.0
    assert any(w["code"] == "low_depth_confidence" for w in res.warnings)
    raw = json.loads((tmp_path / "out" / "result.json").read_text())
    assert schemas.load_result(raw).warnings
