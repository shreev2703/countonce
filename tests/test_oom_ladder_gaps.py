"""OOM ladder on a VGGT-like backbone and on the detector stage. CPU only, no torch import."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.ingest.frames import backbone_grid_map, scaled_target
from countonce.oom import GpuOomError, run_with_oom_ladder
from countonce.pipeline import make_frames, regrid_frames, run
from countonce.recon.adapter import MethodReconstructor
from countonce.recon.cache import CachedReconstructor, ReconCache
from countonce.recon.chunked import ChunkedReconstructor
from countonce.recon.result import Recon
from tests.test_pipeline import CAM_X, DEPTH, F, H, W, make_scene


def _oom():
    return type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory")


class VGGTLike:
    """Path-based method with VGGT's knobs: ``input_size``, OOM raising, ``with_*`` copies."""

    name = "vggt"
    preprocess_mode = "crop"

    def __init__(self, fail_first=0, max_frames=10**9, max_size=518, input_size=518, raises=False):
        self.calls: list[tuple[int, int]] = []  # shared across copies: (n_images, input_size)
        self.fail_first, self.max_frames, self.max_size = fail_first, max_frames, max_size
        self.input_size, self.raises = input_size, raises

    def _copy(self, **kw):
        import copy

        other = copy.copy(self)
        other.__dict__.update(kw)
        return other

    def with_resolution_scale(self, scale):
        return self._copy(input_size=scaled_target(scale))

    def with_oom_raising(self):
        return self._copy(raises=True)

    def config(self):
        return {"input_size": self.input_size}

    def reconstruct(self, image_paths):
        n = len(image_paths)
        self.calls.append((n, self.input_size))
        if (
            len(self.calls) <= self.fail_first
            or n > self.max_frames
            or self.input_size > self.max_size
        ):
            if self.raises:
                raise _oom()
            return Recon.failure(self.name, list(image_paths), "cuda_oom")  # the old path
        k = np.tile(np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]]), (n, 1, 1))
        ext = np.tile(np.eye(3, 4), (n, 1, 1))
        for i in range(n):
            ext[i, 0, 3] = -CAM_X[i]
        return Recon(
            method=self.name,
            image_paths=list(image_paths),
            extrinsics=ext,
            intrinsics=k,
            registered=np.ones(n, bool),
            depth=np.full((n, H, W), DEPTH),
            depth_conf=np.ones((n, H, W)),
        )


def _run(tmp_path, method, detector=None, scene=None):
    img_dir, script = scene or make_scene(tmp_path)
    rec = MethodReconstructor(method, "vggt")
    res = run(
        img_dir,
        "thing",
        Config(out_dir=tmp_path / "out", dedup_threshold=0),
        detector=detector or FakeDetector(script),
        reconstructor=rec,
        export=False,
    )
    return res


def test_ladder_path_does_not_import_torch():
    code = (
        "import sys; import countonce.pipeline, countonce.recon.adapter, countonce.recon.chunked;"
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("kw", "rung"),
    [
        ({"fail_first": 1}, 1),
        ({"max_frames": 3}, 2),
        ({"max_frames": 3, "max_size": 400}, 3),
    ],
)
def test_vggt_like_reaches_each_rung(tmp_path, kw, rung):
    method = VGGTLike(**kw)
    res = _run(tmp_path, method)
    assert res.oom_fallback_rung == rung
    if rung == 3:  # resolution rung succeeded at the 0.75 tier instead of RungUnavailable
        assert method.calls[-1][1] == scaled_target(0.75)
    if rung >= 2:
        assert method.calls[-1][0] <= 3


def test_ladder_order_on_real_wrapper(tmp_path):
    method = VGGTLike(max_frames=0, max_size=0)  # never succeeds
    with pytest.raises(GpuOomError):
        _run(tmp_path, method)
    sizes = [s for _, s in method.calls]
    assert sizes[:3] == [518, 518, 518]  # rungs 0, 1, 2 at native size
    assert sizes[3:] == [scaled_target(0.75), scaled_target(0.5)]


def test_unladdered_call_keeps_failure_path():
    m = VGGTLike(max_size=0)
    assert m.reconstruct([Path("a.png")]).failure_reason == "cuda_oom"
    assert m.with_oom_raising().raises


def test_resolution_scale_passthrough():
    m = MethodReconstructor(VGGTLike(), "vggt")
    scaled = m.with_resolution_scale(0.75)
    assert scaled.grid_target == 392 and m.grid_target == 518
    chunked = ChunkedReconstructor(m, 4, 2).with_resolution_scale(0.75)
    assert chunked.inner.grid_target == 392 and chunked.chunk_size == 4
    cached = CachedReconstructor(m, ReconCache(Path("/nonexistent")), "0")
    assert cached.with_resolution_scale(0.5).inner.grid_target == scaled_target(0.5)


def test_scaled_target_is_patch_multiple():
    for s in (0.9, 0.75, 0.5, 0.33, 0.01):
        t = scaled_target(s)
        assert t % 14 == 0 and 14 <= t < 518
    assert scaled_target(0.75) == 392
    assert scaled_target(1.0) == 518


@pytest.mark.parametrize("mode", ["crop", "pad"])
@pytest.mark.parametrize("size", [(640, 480), (480, 640), (1000, 400), (300, 900)])
def test_grid_map_roundtrip_at_075(mode, size):
    w, h = size
    gm = backbone_grid_map(w, h, mode, scaled_target(0.75))
    assert gm.grid_width <= 392 and gm.grid_height <= 392
    for x, y in [(0.0, 0.0), (w / 2, h / 2), (w - 1.0, h - 1.0), (13.3, 7.7)]:
        rx, ry = gm.to_original(*gm.to_grid(x, y))
        assert abs(rx - x) < 0.5 and abs(ry - y) < 0.5


def test_regrid_frames_updates_grid_map(tmp_path):
    img_dir, _ = make_scene(tmp_path)
    frames = make_frames(sorted(img_dir.glob("*.png")), "crop")
    scaled = MethodReconstructor(VGGTLike(), "vggt").with_resolution_scale(0.75)
    new = regrid_frames(frames, "crop", scaled)
    assert new[0].grid_map == backbone_grid_map(frames[0].width, frames[0].height, "crop", 392)
    assert new[0].grid_map != frames[0].grid_map
    assert regrid_frames(frames, None, scaled) is frames


def test_cache_key_differs_per_scale(tmp_path):
    img_dir, _ = make_scene(tmp_path)
    frames = make_frames(sorted(img_dir.glob("*.png")), "crop")
    base = MethodReconstructor(VGGTLike(), "vggt")
    cache = ReconCache(tmp_path / "c")
    keys = {
        CachedReconstructor(r, cache, "0").key_for(frames)
        for r in (base, base.with_resolution_scale(0.75), base.with_resolution_scale(0.5))
    }
    assert len(keys) == 3


def test_pipeline_cache_dir_scaled_rung(tmp_path):
    method = VGGTLike(max_size=400)
    img_dir, script = make_scene(tmp_path)
    res = run(
        img_dir,
        "thing",
        Config(out_dir=tmp_path / "out", cache_dir=tmp_path / "cache", dedup_threshold=0),
        detector=FakeDetector(script),
        reconstructor=MethodReconstructor(method, "vggt"),
        export=False,
    )
    assert res.oom_fallback_rung == 3
    assert res.recon_cache_key


class OomDetector(FakeDetector):
    """Raises OOM for batches larger than ``max_batch`` (and for the first ``fail_first`` calls)."""

    def __init__(self, script, max_batch=10**9, fail_first=0):
        super().__init__(script)
        self.max_batch, self.fail_first, self.batches = max_batch, fail_first, []

    def detect(self, frames, prompt, exemplars=None):
        self.batches.append(len(frames))
        if len(self.batches) <= self.fail_first or len(frames) > self.max_batch:
            raise _oom()
        return super().detect(frames, prompt, exemplars)


def test_detector_ladder_smaller_batch(tmp_path):
    scene = make_scene(tmp_path)
    det = OomDetector(scene[1], max_batch=3)
    res = _run(tmp_path, VGGTLike(), detector=det, scene=scene)
    assert det.batches[:2] == [len(CAM_X), len(CAM_X)]  # initial, then free-cache retry
    assert max(det.batches[2:]) == max(3, len(CAM_X) // 2) or max(det.batches[2:]) <= 3
    assert res.count >= 0


def test_detector_ladder_free_cache_rung(tmp_path):
    scene = make_scene(tmp_path)
    det = OomDetector(scene[1], fail_first=1)
    _run(tmp_path, VGGTLike(), detector=det, scene=scene)
    assert det.batches == [len(CAM_X)] * 2


def test_detector_ladder_exhausted_raises(tmp_path):
    scene = make_scene(tmp_path)
    det = OomDetector(scene[1], max_batch=0)
    with pytest.raises(GpuOomError) as ei:
        _run(tmp_path, VGGTLike(), detector=det, scene=scene)
    assert ei.value.code == "ERR_GPU_OOM"
    assert ei.value.attempts == ["initial", "free_cache", "smaller_chunks"]


def test_ladder_without_resolution_tiers():
    seen = []

    def stage(att):
        seen.append(att.name)
        raise _oom()

    with pytest.raises(GpuOomError):
        run_with_oom_ladder(stage, n_frames=8, resolution_tiers=())
    assert seen == ["initial", "free_cache", "smaller_chunks"]
