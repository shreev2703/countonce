"""CPU tests of the permutation helpers used by the Fuji test-tree order-invariance run."""

import numpy as np

from bench.order_invariance import (
    common_agreement,
    distinct_permutations,
    fuji_window_row,
    reorder_by_frame,
    to_photo_ids,
)
from countonce.merge.cluster import V1Associator
from tests.test_associators import make_scene


def test_distinct_permutations_are_unique_and_never_identity():
    perms = distinct_permutations(5, 10, seed=3)
    keys = {tuple(p.tolist()) for p in perms}
    assert len(perms) == 10 and len(keys) == 10 and tuple(range(5)) not in keys


def test_distinct_permutations_capped_by_factorial():
    assert len(distinct_permutations(3, 10)) == 5  # 3! - identity


def test_to_photo_ids_inverts_the_upload_order():
    inv = np.array([2, 0, 1])  # position 0 holds photo 2, ...
    part = frozenset({frozenset({"0:1", "1:4"}), frozenset({"2:0"})})
    assert to_photo_ids(part, inv) == frozenset({frozenset({"2:1", "0:4"}), frozenset({"1:0"})})


def test_common_agreement_ignores_detections_only_one_side_has():
    a = frozenset({frozenset({"0:0", "1:0"}), frozenset({"2:0"})})
    b = frozenset({frozenset({"0:0", "1:0"}), frozenset({"2:0", "3:0"})})
    assert common_agreement(a, b) == 1.0


def test_explicit_sigmas_and_list_reorder_keep_v1_invariant():
    recon, lifted, _ = make_scene(1, n_obj=8, n_views=5, noise=0.15)
    sigmas = distinct_permutations(5, 4, seed=1)
    row = fuji_window_row(
        "v1", V1Associator(k=0.5), lifted, recon, tree="T", side="east", k=5,
        sigmas=sigmas, reorder_list=True,
    )  # fmt: skip
    assert row["n_perms"] == 4 and row["n_partition_changed"] == 0
    ids = [ld.det.frame_id for ld in reorder_by_frame(lifted[::-1])]
    assert ids == sorted(ids)
