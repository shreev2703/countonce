"""First real Core benchmark on Fuji-SfM (PLAN §9.1, §9.3 baselines 1-4, §9.4, §9.8).

Fuji-SfM ships no camera poses and no image-to-tree map, so this module builds both from the
data it does have (all choices are logged in ``docs/DEVIATIONS.md``):

1. **Overview** per side: VGGT (chunked) + detector + v1 association on every ``stride``-th frame
   of the whole side. The lifted instances are registered to *all* GT apples (PCA-initialised
   trimmed Sim(3) ICP, ``register_to_gt``). This uses GT apple positions once per side, so it is an
   oracle alignment and only used to place cameras in the GT frame (pseudo-poses, interpolated
   to every frame) and to pick the frames that look at each tree. Counts are never aligned to GT.
2. **Windows**: for each tree and side, ``k`` contiguous frames centred on the frame whose camera
   is nearest the tree along the row. Each window is reconstructed on its own, and aligned to the
   GT frame with ``countonce.recon.chunked.align_chunk`` (camera centres + axis tips) against the
   overview pseudo-poses of the same frames, never against apples.
3. **GT for a window** is the set of GT apples (all 11 trees, since neighbours are in view)
   whose projection lands in a view, in front of the reconstructed surface (depth test), within
   ``max_depth``. Predictions and per-image baselines count only instances inside the row's
   region of interest, so background trees (no GT) are not false positives.

Images are pre-downscaled 3x into ``runs/`` (derived data, git-ignored; CC BY-NC-SA, never commit).
"""

from __future__ import annotations

import hashlib
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench import metrics
from bench.baselines import BASELINES
from bench.datasets.fuji_sfm import FujiScene
from bench.gt_matching import match_detections_to_gt
from bench.runner import Job, RunOutput, read_rows
from countonce.recon.align import Sim3, camera_centers, transform_extrinsics, umeyama
from countonce.types import Detection

DOWNSCALE = 3
GT_TOL_OCC_M = 0.15  # GT apple counts as visible if within this of the reconstructed surface
GT_MAX_DEPTH_M = 4.0
ROI_MARGIN_M = 0.7
MATCH_PX = 10.0  # grid pixels (recon grid is ~518 wide)


# ----------------------------------------------------------------------------- images


def prepare_images(scene: FujiScene, out_dir: Path, scale: int = DOWNSCALE) -> list[Path]:
    """Downscale every SfM frame ``scale``x (JPEG draft mode) into ``out_dir``."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)

    def one(p: Path) -> Path:
        dst = out_dir / (p.stem + ".jpg")
        if dst.exists():
            return dst
        with Image.open(p) as im:
            w, h = im.size
            im.draft("RGB", (w // scale, h // scale))
            small = im.convert("RGB").resize((w // scale, h // scale), Image.Resampling.BILINEAR)
            tmp = dst.with_suffix(".tmp.jpg")
            small.save(tmp, quality=92)
            tmp.replace(dst)
        return dst

    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, scene.image_paths))


# ----------------------------------------------------------------------------- registration


@dataclass
class Registration:
    sim: Sim3
    n_inliers: int
    n_src: int
    inlier_frac: float
    median_err_m: float
    candidate: int


def _axes(x: np.ndarray) -> np.ndarray:
    c = x - x.mean(0)
    w, v = np.linalg.eigh(c.T @ c / len(c))
    return np.asarray(v[:, ::-1])  # columns: major, mid, minor


def register_to_gt(src: np.ndarray, gt: np.ndarray, inlier_m: float = 0.08) -> Registration:
    """Sim(3) taking reconstruction-frame instance centres onto GT apple centroids.

    PCA gives up to 8 axis/sign initialisations (scale from the major-axis spread), each refined by
    trimmed ICP with a shrinking gate. The candidate with the most instances within ``inlier_m``
    of a GT apple wins.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src, float)
    gt = np.asarray(gt, float)
    tree = cKDTree(gt)
    a, b = _axes(src), _axes(gt)
    s0 = float(np.sqrt(np.var(gt @ b[:, 0]) / max(np.var(src @ a[:, 0]), 1e-18)))
    best: Registration | None = None
    cand = 0
    import itertools

    cands = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            r = b[:, list(perm)] @ np.diag(signs) @ a.T
            if np.linalg.det(r) > 0:
                cands.append(r)
    for r in cands:
        sim = Sim3(s0, r, gt.mean(0) - s0 * r @ src.mean(0))
        for gate in (1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.08, 0.08, 0.08, 0.08):
            d, j = tree.query(sim.apply(src))
            keep = d < gate
            if keep.sum() < 8:
                break
            try:
                sim = umeyama(src[keep], gt[j[keep]])
            except ValueError:
                break
        d, _ = tree.query(sim.apply(src))
        n_in = int((d < inlier_m).sum())
        med = float(np.median(d[d < inlier_m])) if n_in else float("inf")
        reg = Registration(sim, n_in, len(src), n_in / len(src), med, cand)
        cand += 1
        if best is None or reg.n_inliers > best.n_inliers:
            best = reg
    assert best is not None
    return best


# ----------------------------------------------------------------------------- pseudo poses


def interpolate_poses(idx: np.ndarray, ext_gt: np.ndarray, all_idx: np.ndarray) -> np.ndarray:
    """World-to-camera extrinsics (in the GT frame) at every ``all_idx`` frame from the overview
    frames ``idx``: linear camera centres, slerp rotations. Frames outside the overview range
    take the nearest end pose."""
    from scipy.spatial.transform import Rotation, Slerp

    r_c2w = Rotation.from_matrix(np.transpose(ext_gt[:, :3, :3], (0, 2, 1)))
    cen = camera_centers(ext_gt)
    q = np.clip(all_idx, idx.min(), idx.max())
    sl = Slerp(idx, r_c2w)(q)
    c = np.stack([np.interp(q, idx, cen[:, k]) for k in range(3)], axis=1)
    rw = sl.as_matrix()  # camera-to-world
    rc = np.transpose(rw, (0, 2, 1))
    t = -np.einsum("nij,nj->ni", rc, c)
    return np.concatenate([rc, t[:, :, None]], axis=2)


# ----------------------------------------------------------------------------- windows / GT


def roi_box(gt_xyz: np.ndarray, margin: float = ROI_MARGIN_M) -> tuple[np.ndarray, np.ndarray]:
    return gt_xyz.min(0) - margin, gt_xyz.max(0) + margin


def in_roi(xyz: np.ndarray, box: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    return np.asarray(np.all((xyz >= box[0]) & (xyz <= box[1]), axis=1))


def window_for_tree(cam_y: np.ndarray, side_len: int, tree_y: float, k: int) -> tuple[int, int]:
    """[start, stop) of ``k`` contiguous side-local frames centred on the camera nearest a tree."""
    c = int(np.argmin(np.abs(cam_y - tree_y)))
    start = min(max(c - k // 2, 0), side_len - k)
    return start, start + k


def visible_gt(
    gt_xyz: np.ndarray,
    intr: np.ndarray,
    ext_gt: np.ndarray,
    depth_gt: np.ndarray,
    tol: float = GT_TOL_OCC_M,
    max_depth: float = GT_MAX_DEPTH_M,
) -> list[np.ndarray]:
    """Per view: indices of GT apples in frame, in front of the reconstructed surface (metres)."""
    from bench.gt_matching import project_points

    out = []
    for v in range(len(intr)):
        h, w = depth_gt.shape[1:]
        uv, z = project_points(gt_xyz, intr[v], ext_gt[v])
        ok = (z > 0) & (z < max_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1)
        ok &= (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        idx = np.flatnonzero(ok)
        surf = depth_gt[v, uv[idx, 1].astype(int), uv[idx, 0].astype(int)]
        out.append(idx[z[idx] <= surf + tol])
    return out


def evaluate_window(
    result: Any,
    recon: Any,
    sim: Sim3,
    gt_ids: list[str],
    gt_xyz: np.ndarray,
    box: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Count, GT, association metrics and per-image baselines of one run, in the GT frame."""
    ext_gt = transform_extrinsics(recon.extrinsics, sim)
    depth_gt = recon.depth * sim.s
    vis = visible_gt(gt_xyz, recon.intrinsics, ext_gt, depth_gt)
    visible_set = sorted({int(i) for v in vis for i in v})
    h, w = recon.depth.shape[1:]

    rec_by_id = {d.id: d for d in result.detections}
    keep_inst = []
    for inst in result.instances:
        if bool(in_roi(sim.apply(np.asarray(inst.xyz)[None]), box)[0]):
            keep_inst.append(inst)
    pred = len(keep_inst)

    # identity labels: match each member detection to the nearest visible GT apple of its view
    dets: list[Detection] = []
    cluster_of: dict[str, int] = {}
    for inst in keep_inst:
        for m in inst.members:
            r = rec_by_id.get(m)
            if r is None or r.uv is None:
                continue
            dets.append(Detection(id=m, frame_id=r.view, score=1.0, point_xy=np.array(r.uv)))
            cluster_of[m] = inst.id
    labels: dict[str, str | None] = {}
    for v in range(len(vis)):
        dv = [d for d in dets if d.frame_id == v]
        if not dv:
            continue
        sel = vis[v]
        labels.update(
            match_detections_to_gt(
                dv,
                [gt_ids[i] for i in sel],
                gt_xyz[sel],
                {v: recon.intrinsics[v]},
                {v: ext_gt[v]},
                {v: (w, h)},
                threshold_px=MATCH_PX,
            )
        )
    clusters = [cluster_of[d.id] for d in dets]
    gts = [labels.get(d.id) for d in dets]
    universe = [gt_ids[i] for i in visible_set]
    _, _, f1 = metrics.pairwise_prf(clusters, gts)
    n_labeled = sum(g is not None for g in gts)

    per_image = []
    for v in range(recon.n_views):
        xyz = [
            sim.apply(np.asarray(r.xyz)[None])[0]
            for r in result.detections
            if r.view == v and r.xyz is not None
        ]
        per_image.append(int(in_roi(np.array(xyz), box).sum()) if xyz else 0)

    return {
        "pred": pred,
        "gt": len(visible_set),
        "raw_count": int(result.count),
        "assoc_f1": f1 if n_labeled >= 2 else float("nan"),
        "duplicate_rate": metrics.duplicate_rate(clusters, gts),
        "miss_rate": metrics.miss_rate(clusters, gts, universe) if universe else float("nan"),
        "n_labeled_dets": n_labeled,
        "per_image": per_image,
        "baselines": {n: fn(per_image) for n, fn in BASELINES.items()},
    }


# ----------------------------------------------------------------------------- jobs / rows


def host_info() -> dict[str, str]:
    return {"host": socket.gethostname()}


def config_hash(path: Path) -> str:
    """Short hash of the frozen config file's bytes; recorded in every row's config_json."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def done_slices(csv_path: Path, split: str) -> set[tuple[str, str, int, str]]:
    """(tree, side, k, method) of every finished (status ok) row of ``split`` in the CSV."""
    return {
        (r["scene_id"], r["sample"], int(r["n_views"]), r["method"])
        for r in read_rows(csv_path)
        if r["status"] == "ok" and r["split"] == split and r["n_views"]
    }


def pending_jobs(jobs: Any, csv_path: Path) -> Any:
    """Yield only jobs whose (tree, side, k, method) has no ok row yet.

    Keyed on the slice and not on ``Job.key()``, so a resume on another host or after a
    harmless config-metadata change does not redo finished work. The CSV is re-read lazily
    per job so rows appended meanwhile are respected.
    """
    for job in jobs:
        slice_id = (job.scene_id, job.sample, int(job.n_views or 0), job.method)
        if slice_id in done_slices(csv_path, job.split):
            continue
        yield job


def make_job(tree: str, split: str, side: str, k: int, method: str, config: dict[str, Any]) -> Job:
    """``scene_id`` is the tree and ``sample`` the side, so figures average a tree's two sides and
    bootstrap over trees (PLAN §9.8)."""
    return Job(
        scene_id=tree,
        method=method,
        n_views=k,
        sample=side,
        ordered=True,
        split=split,
        config=config,
    )


def run_output(ev: dict[str, Any], method: str, res: Any, runtime_s: float) -> RunOutput:
    is_pipe = method.startswith("countonce")
    pred = float(ev["pred"] if is_pipe else ev["baselines"][method])
    return RunOutput(
        pred=pred,
        gt=float(ev["gt"]),
        runtime_s=runtime_s,
        peak_vram_mb=float(res.peak_vram_mb),
        timings={k: float(v) for k, v in res.timings.items()},
        assoc_f1=ev["assoc_f1"] if is_pipe else None,
        duplicate_rate=ev["duplicate_rate"] if is_pipe else None,
        miss_rate=ev["miss_rate"] if is_pipe else None,
        count_error=abs(pred - ev["gt"]),
        domain="real",
    )


def now() -> float:
    return time.perf_counter()


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: np.asarray(o).tolist()))


# ----------------------------------------------------------------------------- dense registration


MAX_CLOUD_POINTS = 200_000  # cap on points fed to a KD-tree; bounds host RAM (float32, ~2.4 MB)
CLOUD_VOXEL_M = 0.02


def subsample_cloud(
    xyz: np.ndarray, voxel: float = CLOUD_VOXEL_M, max_points: int = MAX_CLOUD_POINTS
) -> np.ndarray:
    """Memory-bounded float32 copy of a point cloud: one point per ``voxel`` cell (the first in
    each cell), then an evenly strided cut to at most ``max_points``. Deterministic."""
    pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if voxel > 0 and len(pts):
        cell = np.floor((pts - pts.min(0)) / np.float32(voxel)).astype(np.int64)
        _, first = np.unique(cell, axis=0, return_index=True)
        pts = pts[np.sort(first)]
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(np.int64)]
    return np.ascontiguousarray(pts)


def load_gt_cloud(
    root: Path,
    every: int = 40,
    voxel: float = CLOUD_VOXEL_M,
    max_points: int = MAX_CLOUD_POINTS,
) -> np.ndarray:
    """Scene point cloud (GT frame, metres, shifted frame) as a bounded float32 array.

    The file is streamed, keeping every ``every``-th row, so the full 542k-point text file is
    never held in memory; the result is voxel-thinned and capped at ``max_points``.
    """
    import itertools

    path = Path(root) / "3-3D_data" / "Fuji_apple_trees_point_cloud.txt"
    rows = []
    with path.open() as f:
        for line in itertools.islice(f, 0, None, max(every, 1)):
            parts = line.split()
            if len(parts) >= 3:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return subsample_cloud(np.asarray(rows, dtype=np.float32), voxel, max_points)


def register_dense(
    src: np.ndarray,
    dst: np.ndarray,
    trim: float = 0.4,
    iters: int = 30,
    max_points: int = MAX_CLOUD_POINTS,
) -> Registration:
    """Sim(3) taking a reconstruction's confident points onto the GT scene cloud: PCA-initialised
    trimmed ICP (each step fits the closest ``trim`` share of points). Score = number of source
    points within 5 cm of the cloud. Both clouds are thinned to at most ``max_points`` (float32
    voxel/stride subsample) before the KD-tree is built, so memory is bounded."""
    import itertools

    from scipy.spatial import cKDTree

    src = subsample_cloud(src, voxel=0.0, max_points=max_points).astype(float)
    dst = subsample_cloud(dst, voxel=CLOUD_VOXEL_M, max_points=max_points).astype(float)
    tree = cKDTree(dst)
    a, b = _axes(src), _axes(dst)
    s0 = float(np.sqrt(np.var(dst @ b[:, 0]) / max(np.var(src @ a[:, 0]), 1e-18)))
    best: Registration | None = None
    cand = 0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            r = b[:, list(perm)] @ np.diag(signs) @ a.T
            if np.linalg.det(r) <= 0:
                continue
            sim = Sim3(s0, r, dst.mean(0) - s0 * r @ src.mean(0))
            for _ in range(iters):
                d, j = tree.query(sim.apply(src))
                keep = d <= np.quantile(d, trim)
                try:
                    sim = umeyama(src[keep], dst[j[keep]])
                except ValueError:
                    break
            d, _ = tree.query(sim.apply(src))
            n_in = int((d < 0.05).sum())
            med = float(np.median(d[d < 0.05])) if n_in else float("inf")
            reg = Registration(sim, n_in, len(src), n_in / len(src), med, cand)
            cand += 1
            if best is None or reg.n_inliers > best.n_inliers:
                best = reg
    assert best is not None
    return best
