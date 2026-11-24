"""Chunked reconstruction: fake backbone returning each chunk in its own random Sim(3) frame."""

import numpy as np
import pytest

from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.pipeline import _chunk_cap, run
from countonce.recon.align import Sim3, transform_extrinsics
from countonce.recon.chunked import (
    ChunkAlignmentError,
    ChunkedReconstructor,
    chunk_ranges,
)
from countonce.types import Frame, Recon
from tests.test_pipeline import OBJECTS, PlaneRecon, make_scene

H, W = 4, 5


def _rand_rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _rand_sim(rng):
    return Sim3(float(rng.uniform(0.5, 2.0)), _rand_rot(rng), rng.normal(size=3) * 3)


def _truth(n, rng):
    ext = np.zeros((n, 3, 4))
    for i in range(n):
        R = _rand_rot(rng)
        C = np.array([i * 0.5, np.sin(i), 0.3 * i * i % 1.7])
        ext[i, :3, :3] = R
        ext[i, :3, 3] = -R @ C
    k = np.tile(np.array([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]]), (n, 1, 1))
    wp = rng.normal(size=(n, H, W, 3)) * 2
    return k, ext, wp, rng.uniform(1, 5, (n, H, W)), rng.uniform(1, 3, (n, H, W))


class SimFake:
    """Returns each requested frame range in a fresh random Sim(3) frame of the true world."""

    name = "fake"

    def __init__(self, n, seed=0, perfect_overlap=True):
        self.rng = np.random.default_rng(seed)
        self.k, self.ext, self.wp, self.depth, self.conf = _truth(n, self.rng)
        self.calls = []

    def config(self):
        return {}

    def reconstruct(self, frames):
        ids = [f.id for f in frames]
        self.calls.append(ids)
        sim = _rand_sim(self.rng)  # true world -> chunk frame
        ext = transform_extrinsics(self.ext[ids], sim)
        return Recon(
            backbone="fake",
            intrinsics=self.k[ids],
            extrinsics=ext,
            depth=self.depth[ids] * sim.s,
            depth_conf=self.conf[ids],
            world_points=sim.apply(self.wp[ids]),
            world_points_conf=self.conf[ids],
            runtime_s=1.0,
            peak_vram_mb=10.0 * len(ids),
        )


def _frames(n):
    return [Frame(id=i, path=f"f{i}.png", width=W, height=H, sha256=str(i)) for i in range(n)]


def test_chunk_ranges():
    assert chunk_ranges(10, 4, 2) == [(0, 4), (2, 6), (4, 8), (6, 10)]
    assert chunk_ranges(5, 10, 2) == [(0, 5)]
    with pytest.raises(ValueError, match="overlap"):
        chunk_ranges(10, 4, 1)
    with pytest.raises(ValueError, match="exceed"):
        chunk_ranges(10, 3, 3)


@pytest.mark.parametrize("overlap", [2, 3, 5])
def test_stitched_matches_ground_truth(overlap):
    n = 23
    fake = SimFake(n, seed=overlap)
    out = ChunkedReconstructor(fake, chunk_size=8, overlap=overlap).reconstruct(_frames(n))
    assert len(fake.calls) > 2 and max(map(len, fake.calls)) <= 8
    assert out.n_views == n
    # The first chunk defines the frame; map ground truth into it with the same Sim(3).
    # Fit it from view 0..: compare up to one global Sim(3) via pairwise structure instead.
    from countonce.recon.align import align_by_camera_centers, camera_centers

    sim = align_by_camera_centers(fake.ext, out.extrinsics)
    np.testing.assert_allclose(
        camera_centers(out.extrinsics), sim.apply(camera_centers(fake.ext)), atol=1e-6
    )
    np.testing.assert_allclose(out.extrinsics, transform_extrinsics(fake.ext, sim), atol=1e-6)
    np.testing.assert_allclose(out.world_points, sim.apply(fake.wp), atol=1e-6)
    np.testing.assert_allclose(out.depth, fake.depth * sim.s, atol=1e-6)
    np.testing.assert_array_equal(out.depth_conf, fake.conf)
    np.testing.assert_array_equal(out.world_points_conf, fake.conf)
    np.testing.assert_array_equal(out.intrinsics, fake.k)
    assert out.peak_vram_mb == 80.0


def test_shared_frames_keep_first_chunk_copy():
    fake = SimFake(12, seed=1)
    chunked = ChunkedReconstructor(fake, chunk_size=6, overlap=2)
    first = SimFake(12, seed=1).reconstruct(_frames(12)[:6])  # same rng stream -> same first chunk
    out = chunked.reconstruct(_frames(12))
    np.testing.assert_allclose(out.extrinsics[:6], first.extrinsics, atol=1e-9)


def test_small_input_passes_through():
    fake = SimFake(5)
    out = ChunkedReconstructor(fake, chunk_size=8, overlap=2).reconstruct(_frames(5))
    assert fake.calls == [[0, 1, 2, 3, 4]] and out.n_views == 5


def test_too_little_overlap_is_a_clear_error():
    fake = SimFake(10)
    with pytest.raises(ValueError, match="overlap must be >= 2"):
        ChunkedReconstructor(fake, chunk_size=5, overlap=1)
    with pytest.raises(ValueError, match="chunk_overlap"):
        Config(chunk_overlap=1)


def test_coincident_shared_cameras_raise():
    fake = SimFake(10)
    fake.ext[:] = fake.ext[0]  # all cameras at one place: scale is unobservable
    with pytest.raises(ChunkAlignmentError, match="coincide"):
        ChunkedReconstructor(fake, chunk_size=5, overlap=2).reconstruct(_frames(10))


def test_chunk_cap_from_config():
    assert _chunk_cap(Config(chunked_recon=True, chunk_size=7)) == 7


class _IdPlaneRecon(PlaneRecon):
    """PlaneRecon whose camera depends on the frame id, so any chunk is consistent."""

    def reconstruct(self, frames):
        full = super().reconstruct([frames[0]] * 4)  # reuse geometry builder for 4 slots
        keep = [f.id for f in frames]
        return Recon(
            backbone="fake",
            intrinsics=full.intrinsics[keep],
            extrinsics=full.extrinsics[keep],
            depth=full.depth[keep],
            depth_conf=full.depth_conf[keep],
            world_points=full.world_points[keep],
            world_points_conf=full.world_points_conf[keep],
        )


def test_run_with_chunked_flag_counts_same(tmp_path):
    img_dir, script = make_scene(tmp_path)
    cfg = Config(
        out_dir=tmp_path / "out",
        max_frames=0,
        chunked_recon=True,
        chunk_size=3,
        chunk_overlap=2,
    )
    res = run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=_IdPlaneRecon())
    assert res.count == len(OBJECTS)
