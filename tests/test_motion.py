"""Dynamic-object residual filter (PLAN 8.3b): ghost duplicates from a moving object."""

import numpy as np

from bench.metrics import ghost_duplicate_rate
from countonce.config import Config
from countonce.merge.motion import MotionConfig, flag_moving
from countonce.pipeline import _make_associator
from tests.test_associators import Z, _lifted, make_recon

N_VIEWS = 4


def _scene(moving: bool):
    """Two static objects plus a third that jumps 4 units in x after view 1 (if ``moving``)."""
    recon = make_recon(N_VIEWS)
    lifted, truth = [], {}
    static = [np.array([-4.0, 0, Z]), np.array([4.0, 2.0, Z])]
    for v in range(N_VIEWS):
        for o, p in enumerate(static):
            did = f"{v}:{o}"
            lifted.append(_lifted(did, v, p, 0.2 * v))
            truth[did] = o
        pos = np.array([0.0 + (4.0 if moving and v >= 2 else 0.0), -2.0, Z])
        did = f"{v}:2"
        lifted.append(_lifted(did, v, pos, 0.2 * v))
        truth[did] = 2
    return recon, lifted, truth


def test_static_scene_flags_nothing():
    recon, lifted, _ = _scene(moving=False)
    out = flag_moving(lifted, recon)
    assert not any(ld.possibly_moving for ld in out)
    assert all(ld.motion_residual is not None and ld.motion_residual < 0.5 for ld in out)
    for assoc in ("v1", "v2"):
        res = _make_associator(Config(assoc=assoc, motion_filter=True), 0.5).associate(
            lifted, recon
        )
        assert len(res.instances) == 3
        assert not any(i.possibly_moving for i in res.instances)


def test_moving_object_is_flagged_and_ghost_merged():
    recon, lifted, truth = _scene(moving=True)
    flagged = {ld.det.id for ld in flag_moving(lifted, recon) if ld.possibly_moving}
    assert flagged == {"0:2", "1:2", "2:2", "3:2"}
    for assoc in ("v1", "v2"):
        off = _make_associator(Config(assoc=assoc), 0.5).associate(lifted, recon)
        assert len(off.instances) == 4  # ghost duplicate
        on = _make_associator(Config(assoc=assoc, motion_filter=True), 0.5).associate(lifted, recon)
        assert len(on.instances) == 3
        moved = [i for i in on.instances if i.possibly_moving]
        assert len(moved) == 1 and sorted(moved[0].members) == sorted(flagged)
        assert moved[0].n_views == N_VIEWS
        assert sum(s == "new" for s in on.detection_status.values()) == 3
        for res, expect in ((off, 1 / 4), (on, 0.0)):
            clusters = {m: i.id for i in res.instances for m in i.members}
            ids = sorted(clusters)
            rate = ghost_duplicate_rate([clusters[d] for d in ids], [truth[d] for d in ids], [2])
            assert rate == expect


def test_too_few_views_never_flags():
    recon, lifted, _ = _scene(moving=True)
    cfg = MotionConfig(min_views=10)
    assert not any(ld.possibly_moving for ld in flag_moving(lifted, recon, cfg=cfg))


def test_ghost_rate_nan_without_moving_objects():
    assert np.isnan(ghost_duplicate_rate([0, 1], ["a", "a"], []))
