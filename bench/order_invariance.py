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
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from countonce.merge.base import Associator, ClusterResult
from countonce.types import Frame, LiftedDetection, Recon

_ARRAYS = ("intrinsics", "extrinsics", "depth", "depth_conf", "world_points", "world_points_conf")


def permute_frames(frames: Mapping[int, Frame], sigma: np.ndarray) -> dict[int, Frame]:
    """Frame ``p`` becomes frame ``sigma[p]`` (its image geometry travels with the photo)."""
    return {int(sigma[i]): dataclasses.replace(f, id=int(sigma[i])) for i, f in frames.items()}


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


# --------------------------------------------------------------------- reordering helpers


def reorder_by_frame(lifted: Sequence[LiftedDetection]) -> list[LiftedDetection]:
    """Detection list in frame-id order (stable), as the pipeline would emit it for that upload."""
    return sorted(lifted, key=lambda ld: ld.det.frame_id)


def distinct_permutations(n: int, count: int, seed: int = 0) -> list[np.ndarray]:
    """``count`` different non-identity permutations of ``n`` items (fewer if n! is smaller)."""
    want = min(count, math.factorial(n) - 1)
    rng = np.random.default_rng(seed)
    seen = {tuple(range(n))}
    out: list[np.ndarray] = []
    while len(out) < want:
        sigma = rng.permutation(n)
        if tuple(sigma.tolist()) not in seen:
            seen.add(tuple(sigma.tolist()))
            out.append(sigma)
    return out


def to_photo_ids(part: frozenset[frozenset[str]], inv: np.ndarray) -> frozenset[frozenset[str]]:
    """Rename ``"<position>:<k>"`` ids of a reordered upload back to ``"<photo>:<k>"``.

    ``inv[q]`` is the photo uploaded at position ``q``.
    """

    def back(det_id: str) -> str:
        pos, k = det_id.split(":", 1)
        return f"{int(inv[int(pos)])}:{k}"

    return frozenset(frozenset(back(m) for m in g) for g in part)


def common_agreement(a: frozenset[frozenset[str]], b: frozenset[frozenset[str]]) -> float:
    """Rand index over the detections both partitions cluster (lifting may drop different ones)."""
    common = {m for g in a for m in g} & {m for g in b for m in g}

    def cut(p: frozenset[frozenset[str]]) -> frozenset[frozenset[str]]:
        return frozenset(frozenset(g & common) for g in p if g & common)

    return pair_agreement(cut(a), cut(b))


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
    frames: Mapping[int, Frame] | None = None,
    sigmas: Sequence[np.ndarray] | None = None,
    reorder_list: bool = False,
) -> dict[str, object]:
    """Photo-order invariance of one Fuji window: count range and partition variation.

    ``to_gt`` maps points from the window's reconstruction frame to the GT frame and ``in_roi``
    tells which of those lie in the row's region of interest. Together they give the ROI count
    the Fuji benchmark reports; without them the ROI columns repeat the raw counts. The raw count
    and the partition are over all instances of the window. ``frames`` are the window's frames
    (pass them: without them v2/sync compare frame-pixel boxes with recon-pixel projections and
    cannot merge when the images were resized); they are permuted with the photos.
    """

    def roi_count(res: ClusterResult) -> int:
        if to_gt is None or in_roi is None or not res.instances:
            return len(res.instances)
        xyz = to_gt(np.array([i.xyz for i in res.instances], dtype=float))
        return int(np.asarray(in_roi(xyz)).sum())

    base = assoc.associate(list(lifted), recon, frames)
    base_part = partition(base)
    counts, rois, changed, worst = [len(base.instances)], [roi_count(base)], 0, 1.0
    rng = np.random.default_rng(seed)
    if sigmas is None:
        sigmas = [rng.permutation(recon.n_views) for _ in range(n_perms)]
    for sig in sigmas:
        lf, rc = permute_photos(lifted, recon, sig)
        if reorder_list:
            lf = reorder_by_frame(lf)
        res = assoc.associate(lf, rc, permute_frames(frames, sig) if frames else None)
        counts.append(len(res.instances))
        rois.append(roi_count(res))
        part = partition(res)
        if part != base_part:
            changed += 1
            worst = min(worst, pair_agreement(base_part, part))
    return {
        "assoc": name, "tree": tree, "side": side, "k": k, "n_perms": len(sigmas),
        "n_dets": len(lifted), "base_count": counts[0],
        "count_min": min(counts), "count_max": max(counts),
        "count_range": max(counts) - min(counts),
        "roi_count_min": min(rois), "roi_count_max": max(rois),
        "roi_count_range": max(rois) - min(rois),
        "n_partition_changed": changed, "min_rand_index": worst,
    }  # fmt: skip


# ------------------------------------------------------------------- merging-gate sweep (Fuji)


def n_merges(n_dets: int, count: int) -> int:
    """Merges a window made: detections that joined an existing instance (n_dets - instances)."""
    return max(n_dets - count, 0)


def gate_sweep(
    make: Callable[[float], Associator],
    windows: Sequence[tuple[Recon, Sequence[LiftedDetection]]],
    gates: Sequence[float],
    frames: Sequence[Mapping[int, Frame] | None] | None = None,
) -> list[dict[str, float]]:
    """Per gate, how many windows contain at least one merge (identity photo order).

    ``make(gate)`` builds the associator; a larger gate merges more. Rows are in ``gates`` order.
    """
    out = []
    for g in gates:
        merges = []
        for i, (recon, lifted) in enumerate(windows):
            fr = frames[i] if frames else None
            res = make(g).associate(list(lifted), recon, fr)
            merges.append(n_merges(len(lifted), len(res.instances)))
        hit = sum(m > 0 for m in merges)
        out.append(
            {
                "gate": float(g),
                "n_windows": len(merges),
                "n_with_merge": hit,
                "frac_with_merge": hit / len(merges) if merges else 0.0,
                "total_merges": sum(merges),
            }
        )
    return out


def smallest_merging_gate(
    sweeps: Sequence[Sequence[dict[str, float]]], min_frac: float = 0.5
) -> float | None:
    """Smallest gate at which every sweep has ``frac_with_merge >= min_frac`` (None if none)."""
    if not sweeps:
        return None
    common = set.intersection(*({r["gate"] for r in s} for s in sweeps))
    ok = [
        g
        for g in sorted(common)
        if all(r["frac_with_merge"] >= min_frac for s in sweeps for r in s if r["gate"] == g)
    ]
    return ok[0] if ok else None
