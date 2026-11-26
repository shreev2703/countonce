"""CPU test of the Fuji adapter of the order-invariance harness on a tiny synthetic window."""

import numpy as np

from bench.order_invariance import FUJI_CSV_FIELDS, fuji_window_row
from countonce.merge.cluster import V1Associator
from countonce.merge.match import V2Associator
from tests.test_associators import make_scene


def _row(assoc, name):
    recon, lifted, _ = make_scene(1, n_obj=8, n_views=5, noise=0.15)
    return fuji_window_row(
        name, assoc, lifted, recon, tree="Tree09", side="east", k=5, n_perms=10, seed=0
    )


def test_row_has_all_columns_and_is_invariant_for_v1_v2():
    for name, assoc in (("v1", V1Associator(k=0.5)), ("v2", V2Associator())):
        row = _row(assoc, name)
        assert set(row) == set(FUJI_CSV_FIELDS)
        assert row["n_perms"] == 10
        assert row["count_range"] == 0 and row["n_partition_changed"] == 0
        assert row["min_rand_index"] == 1.0
        assert row["roi_count_range"] == 0  # no ROI callbacks: repeats the raw counts


def test_roi_callbacks_filter_the_count():
    recon, lifted, _ = make_scene(1, n_obj=8, n_views=5, noise=0.15)
    seen = []

    def in_roi(x):
        seen.append(x.shape)
        return x[:, 0] < np.median(x[:, 0])

    row = fuji_window_row(
        "v1", V1Associator(k=0.5), lifted, recon, tree="T", side="west", k=5, n_perms=3,
        to_gt=lambda x: x, in_roi=in_roi,
    )  # fmt: skip
    assert seen and row["roi_count_max"] < row["base_count"]
