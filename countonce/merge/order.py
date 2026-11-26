"""Photo-order canonicalisation for order-dependent associators (PLAN section 0.2, M4).

v2's greedy track building and the spectral synchronization both depend on the order in which
frames are visited. Frame ids only say where a photo sits in the upload, not what it shows, so
visiting in id order makes the count depend on photo order. Sorting by camera centre instead makes
the visit order a function of the photos' geometry alone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from countonce.types import Recon


def camera_center(recon: Recon, view: int) -> tuple[float, float, float]:
    ext = recon.extrinsics[view]
    c = -ext[:, :3].T @ ext[:, 3]
    return (float(c[0]), float(c[1]), float(c[2]))


def canonical_frame_order(
    frame_ids: Iterable[int], recon: Recon, frame_to_view: Mapping[int, int] | None = None
) -> list[int]:
    """Frame ids sorted by camera centre (x, y, z); the id only breaks exact ties."""

    def key(f: int) -> tuple[tuple[float, float, float], int]:
        v = frame_to_view[f] if frame_to_view is not None else f
        return camera_center(recon, v), f

    return sorted(set(frame_ids), key=key)


def frame_rank(
    frame_ids: Iterable[int], recon: Recon, frame_to_view: Mapping[int, int] | None = None
) -> Callable[[int], int]:
    order = canonical_frame_order(frame_ids, recon, frame_to_view)
    rank = {f: i for i, f in enumerate(order)}
    return rank.__getitem__
