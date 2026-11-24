"""COLMAP sparse-to-dense lifting (PLAN §9.3 item 4); hand-built SparseRecon, no pycolmap."""

from pathlib import Path

import numpy as np
import pytest

from countonce.config import Config
from countonce.lift import lift_detections
from countonce.pipeline import lift_kwargs
from countonce.recon.adapter import ReconAdaptError, to_dense_recon
from countonce.recon.result import Recon as SparseRecon
from countonce.types import Detection

H, W, F = 60, 80, 100.0


def _sparse(points, n_views=2, registered=None, **kw):
    k = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    ext = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (n_views, 1, 1))
    reg = np.ones(n_views, bool) if registered is None else np.asarray(registered)
    ext[~reg] = np.nan
    ki = np.tile(k, (n_views, 1, 1))
    ki[~reg] = np.nan
    return SparseRecon(
        method="colmap",
        image_paths=[Path(f"{i}.jpg") for i in range(n_views)],
        image_hw=(H, W),
        extrinsics=ext,
        intrinsics=ki,
        registered=reg,
        points=np.asarray(points, dtype=np.float32),
        **kw,
    )


def _blob(center, n=6, spread=0.05):
    rng = np.random.default_rng(0)
    return np.asarray(center) + rng.uniform(-spread, spread, size=(n, 3))


def _box_det(frame_id=0):
    return Detection(
        id=f"{frame_id}:0", frame_id=frame_id, score=0.9,
        box_xyxy=np.array([20.0, 10.0, 60.0, 50.0]),
    )


def test_sparse_maps_zero_conf_elsewhere_and_unregistered():
    res = _sparse(_blob([0.0, 0.0, 5.0]), registered=[True, False])
    dense = to_dense_recon(res)
    assert dense.world_points_conf[0].max() > 0
    assert (dense.world_points_conf[0] > 0).sum() <= 6
    assert (dense.world_points_conf[1] == 0).all()
    assert dense.depth[0][dense.world_points_conf[0] > 0].min() > 4.5


def test_lift_sparse_recovers_center():
    res = _sparse(
        _blob([0.0, 0.0, 5.0]),
        points_error=np.full(6, 0.5, np.float32),
        points_track_len=np.full(6, 3, np.int32),
    )
    dense = to_dense_recon(res)
    kw = lift_kwargs(Config(backbone="colmap"), dense)
    assert kw == {"min_pixels": 3, "conf_percentile": 0.0}
    (ld,) = lift_detections([_box_det()], dense, **kw)
    assert ld.ok, ld.fail_reason
    np.testing.assert_allclose(ld.xyz, [0, 0, 5.0], atol=0.1)
    assert 0 < ld.lift_conf <= 1


def test_dense_min_pixels_rejects_sparse_input():
    dense = to_dense_recon(_sparse(_blob([0.0, 0.0, 5.0], n=3)))
    (ld,) = lift_detections([_box_det()], dense)  # dense default: 5 pixels
    assert ld.fail_reason == "too_few_pixels"


def test_no_points_in_box_fails():
    far = _blob([1.5, 0.0, 5.0])  # projects right of the box (x ~ 70 px)
    dense = to_dense_recon(_sparse(far))
    (ld,) = lift_detections([_box_det()], dense, min_pixels=3, conf_percentile=0.0)
    assert not ld.ok and ld.fail_reason is not None


def test_track_views_restrict_projection():
    res = _sparse(_blob([0.0, 0.0, 5.0]), points_views=[np.array([0])] * 6)
    dense = to_dense_recon(res)
    assert dense.world_points_conf[0].max() > 0
    assert (dense.world_points_conf[1] == 0).all()


def test_missing_points_raises():
    res = _sparse(_blob([0.0, 0.0, 5.0]))
    res.points = None
    with pytest.raises(ReconAdaptError):
        to_dense_recon(res)
