"""Chunked reconstruction (PLAN section 8.3): stitch overlapping chunks with Sim(3).

When a capture has more frames than the backbone can hold (config ``chunk_size`` or the
memory-guard calibration), reconstruct overlapping chunks, align each chunk to the previously
merged result with a Sim(3) fitted to the shared frames' camera centres (Umeyama,
``recon.align``) and merge into one dense ``types.Recon``. Shared frames keep the earlier
chunk's copy. Chaining is sequential, so scale/rotation drift accumulates with chunk count;
``stitch="pose_graph"`` instead solves a global Sim(3) synchronization over every pair of chunks
that share frames (``recon.posegraph``) and falls back to chaining when the graph is a tree
(see docs/DEVIATIONS.md).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from countonce.oom import RungUnavailable
from countonce.recon.align import Sim3, camera_centers, transform_extrinsics, umeyama
from countonce.recon.base import Reconstructor
from countonce.types import Frame, Recon

MIN_OVERLAP = 2


class ChunkAlignmentError(RuntimeError):
    """Chunks cannot be stitched (too little or degenerate overlap)."""


def chunk_ranges(n: int, cap: int, overlap: int) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` ranges of <= ``cap`` frames; neighbours share ``overlap``."""
    if overlap < MIN_OVERLAP:
        raise ValueError(f"overlap must be >= {MIN_OVERLAP} shared frames, got {overlap}")
    if cap <= overlap:
        raise ValueError(f"chunk size ({cap}) must exceed overlap ({overlap})")
    ranges: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + cap, n)
        ranges.append((start, end))
        if end >= n:
            return ranges
        start = end - overlap


def _alignment_points(ext: np.ndarray) -> np.ndarray:
    """Correspondence points for shared cameras: centres, plus (for exactly two cameras, where
    centres alone leave the rotation about their axis free) each camera's optical-axis and
    up-axis tips at one baseline length, which is scale-consistent within a chunk."""
    c = camera_centers(ext)
    if len(c) >= 3:
        return c
    baseline = float(np.linalg.norm(c[1] - c[0]))
    rot = ext[:, :3, :3]  # camera axes in world are the rows of R
    return np.concatenate([c, c + baseline * rot[:, 2], c + baseline * rot[:, 1]])


def align_chunk(merged_ext: np.ndarray, chunk_ext: np.ndarray) -> Sim3:
    """Sim(3) taking the chunk's world frame to the merged frame from shared cameras."""
    n = len(merged_ext)
    if n < MIN_OVERLAP or len(chunk_ext) != n:
        raise ChunkAlignmentError(
            f"need at least {MIN_OVERLAP} shared frames to align chunks, got {n}"
        )
    if np.linalg.norm(np.ptp(camera_centers(merged_ext), axis=0)) < 1e-9:
        raise ChunkAlignmentError("shared cameras coincide; cannot estimate chunk scale")
    try:
        return umeyama(_alignment_points(chunk_ext), _alignment_points(merged_ext))
    except ValueError as e:
        raise ChunkAlignmentError(f"chunk alignment failed: {e}") from e


def _scaled(inner: Any, scale: float) -> Any:
    fn = getattr(inner, "with_resolution_scale", None)
    if fn is None:
        raise RungUnavailable("inner reconstructor has no resolution control")
    return fn(scale)


class ChunkedReconstructor:
    """``Reconstructor`` wrapper; passes through unchanged when frames fit in one chunk."""

    def __init__(
        self, inner: Reconstructor, chunk_size: int, overlap: int = 4, stitch: str = "chain"
    ) -> None:
        if stitch not in ("chain", "pose_graph"):
            raise ValueError(f"stitch must be 'chain' or 'pose_graph', got {stitch!r}")
        self.stitch = stitch
        chunk_ranges(chunk_size + 1, chunk_size, overlap)  # validate arguments early
        self.inner = inner
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.name = f"chunked_{inner.name}"

    def config(self) -> dict[str, Any]:
        return {
            "chunked": True,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            **({"stitch": self.stitch} if self.stitch != "chain" else {}),
            "inner": self.inner.config(),
        }

    def with_resolution_scale(self, scale: float) -> ChunkedReconstructor:
        inner = _scaled(self.inner, scale)
        return ChunkedReconstructor(inner, self.chunk_size, self.overlap, self.stitch)

    def reconstruct(self, frames: list[Frame]) -> Recon:
        if len(frames) <= self.chunk_size:
            return self.inner.reconstruct(frames)
        t0 = time.perf_counter()
        ranges = chunk_ranges(len(frames), self.chunk_size, self.overlap)
        parts: list[dict[str, np.ndarray]] = []
        runtime, vram, backbone, scale = 0.0, 0.0, "", 1.0
        merged_ext = np.zeros((0, 3, 4))
        recs: list[Recon] | None = None
        graph_sims: list[Sim3] | None = None
        if self.stitch == "pose_graph":
            from countonce.recon.posegraph import solve_pose_graph

            recs = [self.inner.reconstruct(frames[a:b]) for a, b in ranges]
            for i, ((a, b), r) in enumerate(zip(ranges, recs, strict=True)):
                if r.n_views != b - a:
                    raise ChunkAlignmentError(
                        f"chunk {i} returned {r.n_views} views for {b - a} frames"
                    )
            res = solve_pose_graph(
                [r.extrinsics for r in recs], [np.arange(a, b) for a, b in ranges]
            )
            graph_sims = None if res.is_tree else res.sims  # a tree is exactly the chain
        for i, (a, b) in enumerate(ranges):
            rec = recs[i] if recs is not None else self.inner.reconstruct(frames[a:b])
            if rec.n_views != b - a:
                raise ChunkAlignmentError(
                    f"chunk {i} returned {rec.n_views} views for {b - a} frames"
                )
            runtime += rec.runtime_s
            vram = max(vram, rec.peak_vram_mb)
            ext, depth, wp = rec.extrinsics, rec.depth, rec.world_points
            skip = 0
            if i == 0:
                backbone, scale = rec.backbone, rec.scene_scale
            else:
                skip = self.overlap
                if graph_sims is not None:
                    sim = graph_sims[i]
                else:
                    try:
                        sim = align_chunk(merged_ext[a : a + skip], rec.extrinsics[:skip])
                    except ChunkAlignmentError as e:
                        raise ChunkAlignmentError(f"chunk {i} (frames {a}-{b - 1}): {e}") from e
                ext = transform_extrinsics(rec.extrinsics, sim)
                depth = rec.depth * sim.s
                wp = sim.apply(rec.world_points)
            parts.append(
                {
                    "intrinsics": rec.intrinsics[skip:],
                    "extrinsics": ext[skip:],
                    "depth": depth[skip:],
                    "depth_conf": rec.depth_conf[skip:],
                    "world_points": wp[skip:],
                    "world_points_conf": rec.world_points_conf[skip:],
                }
            )
            merged_ext = np.concatenate([merged_ext, ext[skip:]])
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        return Recon(
            backbone=backbone,
            scene_scale=scale,
            runtime_s=runtime + (time.perf_counter() - t0),
            peak_vram_mb=vram,
            **cat,
        )
