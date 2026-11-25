"""True photo-order invariance (PLAN 0.2 M4): permute photo-to-frame assignment AND cameras."""

from functools import cache

import numpy as np
import pytest

from bench.order_invariance import measure, pair_agreement, partition, permute_photos
from countonce.match.model import make_model
from countonce.merge.cluster import V1Associator
from countonce.merge.match import V2Associator
from countonce.merge.sync import SyncAssociator
from countonce.merge.v3 import V3Associator
from tests.test_associators import make_scene

N_PERMS = 12


@cache
def _v3_model():
    pytest.importorskip("torch")  # only the v3 cases need it
    return make_model(width=32, layers=2).eval()  # architecture invariance; weights irrelevant


FACTORIES = {
    "v1": lambda: V1Associator(k=0.5),
    "v2": V2Associator,
    "sync": SyncAssociator,
    "v3": lambda: V3Associator(model=_v3_model()),
    "v3_sync": lambda: V3Associator(model=_v3_model(), synchronize=True),
}
SCENES = [(0, 0.03), (1, 0.15), (2, 0.4), (3, 0.15)]


def _scene(seed, noise):
    recon, lifted, _ = make_scene(seed, n_obj=15, n_views=6, noise=noise)
    return recon, lifted


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize(("seed", "noise"), SCENES)
def test_count_and_partition_invariant_to_photo_order(name, seed, noise):
    recon, lifted = _scene(seed, noise)
    r = measure(FACTORIES[name](), lifted, recon, n_perms=N_PERMS, seed=seed)
    assert r.n_perms >= 10
    assert r.count_range == 0, r.counts
    assert r.invariant and r.min_agreement == 1.0


def test_permutation_moves_cameras_with_photos():
    recon, lifted = _scene(0, 0.03)
    sigma = np.array([3, 0, 5, 1, 4, 2])
    lf, rc = permute_photos(lifted, recon, sigma)
    for old, new in zip(lifted, lf, strict=True):
        assert new.det.id == old.det.id and new.det.frame_id == sigma[old.det.frame_id]
    for p in range(6):
        assert np.array_equal(rc.extrinsics[sigma[p]], recon.extrinsics[p])
        assert np.array_equal(rc.depth[sigma[p]], recon.depth[p])


def test_harness_detects_frame_id_order_dependence(monkeypatch):
    """With the pre-M4 visit order (ascending frame id) v2 varies, so the test has teeth."""
    import countonce.merge.match as match

    monkeypatch.setattr(match, "canonical_frame_order", lambda ids, recon, f2v=None: sorted(ids))
    recon, lifted = _scene(1, 0.15)
    r = measure(V2Associator(), lifted, recon, n_perms=N_PERMS, seed=1)
    assert r.n_partition_changed > 0 and r.min_agreement < 1.0


def test_pair_agreement_basics():
    a = frozenset({frozenset({"x", "y"}), frozenset({"z"})})
    b = frozenset({frozenset({"x"}), frozenset({"y", "z"})})
    assert pair_agreement(a, a) == 1.0 and pair_agreement(a, b) < 1.0
    assert partition  # exported helper
