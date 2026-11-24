"""Spectral permutation synchronization baseline (PLAN §9.3 item 9, §18 cycle consistency)."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from countonce.merge.base import Associator
from countonce.merge.sync import SyncAssociator, synchronize
from tests.test_associators import make_scene


def _check(labels, views):
    n = len(views)
    same = labels[:, None] == labels[None, :]
    # cannot-link: no two distinct detections of one view share an instance
    for i in range(n):
        for j in range(i + 1, n):
            if views[i] == views[j]:
                assert not same[i, j]
    # transitivity of the induced match relation (cycle consistency)
    for i in range(n):
        for j in range(n):
            if same[i, j]:
                assert np.array_equal(same[i], same[j])


@st.composite
def noisy_matrices(draw):
    n_views = draw(st.integers(2, 5))
    sizes = draw(st.lists(st.integers(0, 4), min_size=n_views, max_size=n_views))
    views = [v for v, s in enumerate(sizes) for _ in range(s)]
    n = len(views)
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    a = np.triu(a, 1)
    return a + a.T, views


@settings(max_examples=60, deadline=None)
@given(noisy_matrices())
def test_synchronized_matches_are_transitive_and_cannot_link(data):
    a, views = data
    labels = synchronize(a, views)
    assert labels.shape == (len(views),)
    _check(labels, np.asarray(views))


def test_sync_recovers_planted_permutations():
    rng = np.random.default_rng(0)
    n_obj, n_views = 6, 4
    views = [v for v in range(n_views) for _ in range(n_obj)]
    truth = np.array([o for _ in range(n_views) for o in range(n_obj)])
    a = (truth[:, None] == truth[None, :]).astype(float)
    flip = rng.random(a.shape) < 0.08
    flip = np.triu(flip, 1)
    a = np.where(flip | flip.T, 1 - a, a)
    labels = synchronize(a, views)
    _check(labels, np.asarray(views))
    assert len(set(labels.tolist())) == n_obj


def test_sync_is_associator_and_recovers_n_on_geometry_fixture():
    recon, lifted, truth = make_scene(seed=3, n_obj=12, n_views=5, drop=0.2, fp=0)
    assoc = SyncAssociator()
    assert isinstance(assoc, Associator)
    res = assoc.associate(lifted, recon)
    n_true = len(set(truth.values()))
    assert len(res.instances) == n_true
    frame = {ld.det.id: ld.det.frame_id for ld in lifted}
    for inst in res.instances:
        frames = [frame[m] for m in inst.members]
        assert len(frames) == len(set(frames))  # never violates cannot-link


def test_sync_requires_recon():
    import pytest

    with pytest.raises(ValueError):
        SyncAssociator().associate([], None)
