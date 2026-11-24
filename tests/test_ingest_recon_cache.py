"""Ingest filters, recon adapter, cache and scene.ply. CPU only: no torch, no pycolmap."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from countonce.export import write_scene_ply
from countonce.ingest.frames import drop_blurry, laplacian_variance
from countonce.ingest.select import cap_frames, drop_near_duplicates
from countonce.recon.adapter import MethodReconstructor, ReconAdaptError, to_dense_recon
from countonce.recon.cache import CachedReconstructor, ReconCache, input_hash, recon_cache_key
from countonce.recon.result import Recon as SparseRecon
from countonce.types import Frame, Recon


def _save(path: Path, arr: np.ndarray) -> Path:
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return path


def _noise(seed: int, size: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3))


# --- ingest ---------------------------------------------------------------------------------


def test_blur_filter_drops_blurred_keeps_sharp(tmp_path):
    sharp = _save(tmp_path / "sharp.png", _noise(0))
    blurred_img = Image.open(sharp).filter(ImageFilter.GaussianBlur(4))
    blurred = tmp_path / "blur.png"
    blurred_img.save(blurred)
    assert laplacian_variance(sharp) > 10 * laplacian_variance(blurred)
    thr = (laplacian_variance(sharp) + laplacian_variance(blurred)) / 2
    kept, scores = drop_blurry([sharp, blurred], thr)
    assert kept == [sharp]
    assert set(scores) == {"sharp.png", "blur.png"}


def test_near_duplicates_removed_against_last_kept(tmp_path):
    a = _noise(1)
    paths = [
        _save(tmp_path / "a.png", a),
        _save(tmp_path / "a2.png", np.clip(a + 1, 0, 255)),  # near duplicate
        _save(tmp_path / "b.png", _noise(2)),
        _save(tmp_path / "b2.png", _noise(2)),  # exact duplicate
    ]
    kept, dists = drop_near_duplicates(paths)
    assert [p.name for p in kept] == ["a.png", "b.png"]
    assert dists["a.png"] == float("inf")
    assert dists["b2.png"] == 0.0


def test_cap_frames_even_spacing():
    paths = [Path(f"f{i}.png") for i in range(10)]
    assert cap_frames(paths, 20) == paths
    capped = cap_frames(paths, 4)
    assert len(capped) == 4
    assert capped[0] == paths[0] and capped[-1] == paths[-1]
    gaps = np.diff([int(p.stem[1:]) for p in capped])
    assert gaps.max() - gaps.min() <= 1


# --- adapter --------------------------------------------------------------------------------


def _sparse(n=3, h=8, w=10) -> SparseRecon:
    k = np.tile(np.array([[10.0, 0, w / 2], [0, 10.0, h / 2], [0, 0, 1]]), (n, 1, 1))
    ext = np.zeros((n, 3, 4))
    ext[:, :, :3] = np.eye(3)
    ext[:, 0, 3] = -np.arange(n)  # camera i sits at x = i
    return SparseRecon(
        method="vggt",
        image_paths=[Path(f"{i}.png") for i in range(n)],
        extrinsics=ext,
        intrinsics=k,
        registered=np.ones(n, bool),
        depth=np.full((n, h, w), 2.0),
        depth_conf=np.full((n, h, w), 5.0),
        runtime_s=1.5,
    )


def test_adapter_unprojects_depth_to_world():
    res = _sparse()
    dense = to_dense_recon(res)
    assert isinstance(dense, Recon) and dense.n_views == 3
    # principal-point pixel of view 1 (camera at x=1) looks down +z: world = (1, 0, 2)
    assert np.allclose(dense.world_points[1, 4, 5], [1.0, 0.0, 2.0], atol=1e-5)
    assert dense.runtime_s == 1.5 and dense.scene_scale > 0


def test_adapter_unregistered_view_gets_zero_conf():
    res = _sparse()
    res.registered[2] = False
    res.extrinsics[2] = np.nan
    dense = to_dense_recon(res)
    assert (dense.world_points_conf[2] == 0).all()
    assert np.isfinite(dense.world_points).all()


def test_adapter_rejects_depthless_and_failed():
    res = _sparse()
    res.depth = None
    with pytest.raises(ReconAdaptError, match="depth"):
        to_dense_recon(res)
    with pytest.raises(ReconAdaptError, match="failed"):
        to_dense_recon(SparseRecon.failure("colmap", [Path("a.png")], "no pairs"))


class FakeMethod:
    name = "vggt"

    def __init__(self):
        self.calls = 0

    def reconstruct(self, image_paths):
        self.calls += 1
        return _sparse(len(image_paths))

    def config(self):
        return {"size": 518}


def _frames(n=3) -> list[Frame]:
    return [Frame(i, Path(f"{i}.png"), 10, 8, sha256=f"hash{i}") for i in range(n)]


def test_heavy_deps_not_imported():
    import countonce.recon.adapter  # noqa: F401

    # importing the adapter must not itself pull in the heavy backends
    assert "countonce.recon.vggt" not in sys.modules
    assert "countonce.recon.colmap" not in sys.modules


# --- cache ----------------------------------------------------------------------------------


def test_cache_hit_miss_and_key_sensitivity(tmp_path):
    method = FakeMethod()
    cr = CachedReconstructor(MethodReconstructor(method), ReconCache(tmp_path), "v1")
    frames = _frames()
    first = cr.reconstruct_cached(frames)
    second = cr.reconstruct_cached(frames)
    assert not first.hit and second.hit and method.calls == 1
    assert first.key == second.key
    assert np.array_equal(first.recon.world_points, second.recon.world_points)
    assert second.recon.backbone == "vggt"
    # different order, version or config -> different key
    assert cr.key_for(list(reversed(frames))) != first.key
    assert recon_cache_key(frames, "vggt", "v2", {"size": 518}) != first.key
    assert recon_cache_key(frames, "vggt", "v1", {"size": 336}) != first.key
    assert input_hash(frames) == input_hash(_frames())


def test_corrupt_cache_entry_is_a_miss(tmp_path):
    cache = ReconCache(tmp_path)
    cr = CachedReconstructor(MethodReconstructor(FakeMethod()), cache, "v1")
    frames = _frames()
    key = cr.key_for(frames)
    tmp_path.mkdir(exist_ok=True)
    cache.path(key).write_bytes(b"not an npz")
    assert not cr.reconstruct_cached(frames).hit
    assert cr.reconstruct_cached(frames).hit


# --- scene.ply ------------------------------------------------------------------------------


def test_scene_ply_downsamples_and_has_valid_header(tmp_path):
    dense = to_dense_recon(_sparse(n=3, h=8, w=10))  # 240 points
    out = tmp_path / "scene.ply"
    n = write_scene_ply(dense, out, max_points=50, conf_percentile=0.0)
    assert n == 50
    raw = out.read_bytes()
    header, _, body = raw.partition(b"end_header\n")
    assert b"element vertex 50" in header
    assert len(body) == 50 * (12 + 3)
    # deterministic given the seed
    out2 = tmp_path / "scene2.ply"
    write_scene_ply(dense, out2, max_points=50, conf_percentile=0.0)
    assert out2.read_bytes() == raw


def test_scene_ply_skips_zero_conf_points(tmp_path):
    res = _sparse()
    res.registered[2] = False
    res.extrinsics[2] = np.nan
    n = write_scene_ply(to_dense_recon(res), tmp_path / "s.ply", conf_percentile=0.0)
    assert n == 2 * 8 * 10
