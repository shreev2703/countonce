"""True photo-order invariance harness (PLAN section 0.2, M4).

The existing property tests shuffle the *list* of detections or the order of whole-frame blocks
while every frame keeps its camera. That is not photo-order invariance. Here the same photos are
fed in a different order: photo ``p`` is re-assigned to frame id ``sigma[p]`` and the camera arrays
(intrinsics, extrinsics, depth, ...) are permuted the same way, so every photo still sees its own
camera but sits at a different position in the frame sequence. Detection ids stay tied to the
photo, so partitions are comparable across permutations.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from countonce.merge.base import Associator, ClusterResult
from countonce.types import LiftedDetection, Recon

_ARRAYS = ("intrinsics", "extrinsics", "depth", "depth_conf", "world_points", "world_points_conf")


def permute_photos(
    lifted: Sequence[LiftedDetection], recon: Recon, sigma: np.ndarray
) -> tuple[list[LiftedDetection], Recon]:
    """Move photo ``p`` to frame id ``sigma[p]`` and permute the camera arrays to match."""
    inv = np.argsort(sigma)  # new position q holds old photo inv[q]
    new_recon = dataclasses.replace(recon, **{k: getattr(recon, k)[inv] for k in _ARRAYS})
    out = [
        dataclasses.replace(
            ld, det=dataclasses.replace(ld.det, frame_id=int(sigma[ld.det.frame_id]))
        )
        for ld in lifted
    ]
    return out, new_recon


def partition(res: ClusterResult) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(i.members) for i in res.instances)


def pair_agreement(a: frozenset[frozenset[str]], b: frozenset[frozenset[str]]) -> float:
    """Rand index of two partitions of the same detections (1.0 = identical)."""
    ids = sorted({m for g in a for m in g})
    la = {m: i for i, g in enumerate(a) for m in g}
    lb = {m: i for i, g in enumerate(b) for m in g}
    n = len(ids)
    if n < 2:
        return 1.0
    agree = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            agree += (la[ids[i]] == la[ids[j]]) == (lb[ids[i]] == lb[ids[j]])
            tot += 1
    return agree / tot


@dataclass
class InvarianceResult:
    n_perms: int
    base_count: int
    counts: list[int]
    count_range: int  # max - min over the permutations (identity included)
    count_std: float
    n_partition_changed: int  # permutations whose partition differs from the identity one
    min_agreement: float  # worst Rand index against the identity partition

    @property
    def invariant(self) -> bool:
        return self.n_partition_changed == 0


def measure(
    assoc: Associator,
    lifted: Sequence[LiftedDetection],
    recon: Recon,
    n_perms: int = 10,
    seed: int = 0,
) -> InvarianceResult:
    """Run ``assoc`` on the identity assignment and ``n_perms`` random photo permutations."""
    base = assoc.associate(list(lifted), recon)
    base_part = partition(base)
    rng = np.random.default_rng(seed)
    counts, changed, worst = [len(base.instances)], 0, 1.0
    for _ in range(n_perms):
        sigma = rng.permutation(recon.n_views)
        lf, rc = permute_photos(lifted, recon, sigma)
        part = partition(assoc.associate(lf, rc))
        counts.append(len(part))
        if part != base_part:
            changed += 1
            worst = min(worst, pair_agreement(base_part, part))
    return InvarianceResult(
        n_perms=n_perms,
        base_count=counts[0],
        counts=counts,
        count_range=max(counts) - min(counts),
        count_std=float(np.std(counts)),
        n_partition_changed=changed,
        min_agreement=worst,
    )


def sweep(
    assocs: dict[str, Callable[[], Associator]],
    scenes: Sequence[tuple[Recon, Sequence[LiftedDetection]]],
    n_perms: int = 10,
) -> dict[str, list[InvarianceResult]]:
    return {
        name: [measure(make(), lf, rc, n_perms, seed=i) for i, (rc, lf) in enumerate(scenes)]
        for name, make in assocs.items()
    }


# ----------------------------------------------------------------------------- Fuji adapter

FUJI_CSV_FIELDS = (
    "assoc", "tree", "side", "k", "n_perms", "n_dets", "base_count", "count_min", "count_max",
    "count_range", "roi_count_min", "roi_count_max", "roi_count_range", "n_partition_changed",
    "min_rand_index",
)  # fmt: skip


def fuji_window_row(
    name: str,
    assoc: Associator,
    lifted: Sequence[LiftedDetection],
    recon: Recon,
    *,
    tree: str,
    side: str,
    k: int,
    n_perms: int = 10,
    seed: int = 0,
    to_gt: Callable[[np.ndarray], np.ndarray] | None = None,
    in_roi: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, object]:
    """Photo-order invariance of one Fuji window: count range and partition variation.

    ``to_gt`` maps points from the window's reconstruction frame to the GT frame and ``in_roi``
    tells which of those lie in the row's region of interest. Together they give the ROI count
    the Fuji benchmark reports; without them the ROI columns repeat the raw counts. The raw count
    and the partition are over all instances of the window.
    """

    def roi_count(res: ClusterResult) -> int:
        if to_gt is None or in_roi is None or not res.instances:
            return len(res.instances)
        xyz = to_gt(np.array([i.xyz for i in res.instances], dtype=float))
        return int(np.asarray(in_roi(xyz)).sum())

    base = assoc.associate(list(lifted), recon)
    base_part = partition(base)
    counts, rois, changed, worst = [len(base.instances)], [roi_count(base)], 0, 1.0
    rng = np.random.default_rng(seed)
    for _ in range(n_perms):
        lf, rc = permute_photos(lifted, recon, rng.permutation(recon.n_views))
        res = assoc.associate(lf, rc)
        counts.append(len(res.instances))
        rois.append(roi_count(res))
        part = partition(res)
        if part != base_part:
            changed += 1
            worst = min(worst, pair_agreement(base_part, part))
    return {
        "assoc": name, "tree": tree, "side": side, "k": k, "n_perms": n_perms,
        "n_dets": len(lifted), "base_count": counts[0],
        "count_min": min(counts), "count_max": max(counts),
        "count_range": max(counts) - min(counts),
        "roi_count_min": min(rois), "roi_count_max": max(rois),
        "roi_count_range": max(rois) - min(rois),
        "n_partition_changed": changed, "min_rand_index": worst,
    }  # fmt: skip
