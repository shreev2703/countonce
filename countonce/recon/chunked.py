"""Chunked reconstruction (PLAN section 8.3): stitch overlapping chunks with Sim(3).

When a capture has more frames than the backbone can hold (config ``chunk_size`` or the
memory-guard calibration), reconstruct overlapping chunks, align each chunk to the previously
merged result with a Sim(3) fitted to the shared frames' camera centres (Umeyama,
``recon.align``) and merge into one dense ``types.Recon``. Shared frames keep the earlier
chunk's copy. Chaining is sequential, so scale/rotation drift accumulates with chunk count
(see docs/DEVIATIONS.md; pose-graph stitching is a separate task).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

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


class ChunkedReconstructor:
    """``Reconstructor`` wrapper; passes through unchanged when frames fit in one chunk."""

    def __init__(self, inner: Reconstructor, chunk_size: int, overlap: int = 4) -> None:
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
            "inner": self.inner.config(),
        }

    def reconstruct(self, frames: list[Frame]) -> Recon:
        if len(frames) <= self.chunk_size:
            return self.inner.reconstruct(frames)
        t0 = time.perf_counter()
        ranges = chunk_ranges(len(frames), self.chunk_size, self.overlap)
        parts: list[dict[str, np.ndarray]] = []
        runtime, vram, backbone, scale = 0.0, 0.0, "", 1.0
        merged_ext = np.zeros((0, 3, 4))
        for i, (a, b) in enumerate(ranges):
            rec = self.inner.reconstruct(frames[a:b])
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
