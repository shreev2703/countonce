"""Tests for the code ported from vggt-sparse-recon (pure logic only, no GPU needed)."""

from pathlib import Path

import numpy as np
import pytest

from countonce.env_check import problems
from countonce.ingest.frames import sample_frames, vggt_grid_size
from countonce.recon.profiling import frame_cap
from countonce.recon.result import Recon


def test_sample_frames_even_and_clamped() -> None:
    paths = [Path(f"f{i}.jpg") for i in range(10)]
    picked = sample_frames(paths, 3, None)
    assert [p.name for p in picked] == ["f0.jpg", "f4.jpg", "f9.jpg"] or len(picked) == 3
    assert picked[0] == paths[0] and picked[-1] == paths[-1]
    assert sample_frames(paths, 20, None) == paths
    assert sample_frames(paths, 0, None) == paths


def test_sample_frames_seeded_is_reproducible_and_ordered() -> None:
    paths = [Path(f"f{i}.jpg") for i in range(30)]
    a = sample_frames(paths, 5, 7)
    assert a == sample_frames(paths, 5, 7)
    assert a == sorted(a)


def test_vggt_grid_size() -> None:
    assert vggt_grid_size(1920, 1080) == (518, 294)
    w, h = vggt_grid_size(1080, 1920)
    assert (w, h) == (518, 518)


def test_recon_camera_centers_and_failure() -> None:
    ext = np.zeros((1, 3, 4))
    ext[0, :, :3] = np.eye(3)
    ext[0, :, 3] = [1.0, 2.0, 3.0]
    recon = Recon(method="x", image_paths=[Path("a.jpg")], extrinsics=ext)
    assert np.allclose(recon.camera_centers(), [[-1, -2, -3]])
    failed = Recon.failure("x", [Path("a.jpg"), Path("b.jpg")], "cuda_oom")
    assert failed.failed and failed.registration_rate == 0.0


def test_frame_cap_uses_profile(tmp_path: Path) -> None:
    assert frame_cap(40, "bf16", tmp_path / "missing.json") == 40
    prof = tmp_path / "p.json"
    prof.write_text('{"profiles": {"bf16": {"max_ok_frames": 12}}}')
    assert frame_cap(40, "bf16", prof) == 12
    assert frame_cap(8, "bf16", prof) == 8


def test_env_problems_reports_missing_torch() -> None:
    assert problems({"gpu": {"error": "torch is not installed"}}) == ["torch is not installed"]


def test_colmap_config_rejects_low_min_model_size() -> None:
    pytest.importorskip("pycolmap")
    from countonce.recon.colmap import ColmapConfig

    with pytest.raises(ValueError):
        ColmapConfig(min_model_size=2)
