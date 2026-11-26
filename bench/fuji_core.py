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
   GT frame with ``countonce.recon.chunked.align_chunk`` (camera orientations + centres) against the
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
from countonce.recon.align import (
    Sim3,
    camera_centers,
    rotation_angles_deg,
    transform_extrinsics,
    umeyama,
)
from countonce.recon.chunked import MIN_OVERLAP, ChunkAlignmentError, align_chunk
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
            d, j = map(np.asarray, tree.query(sim.apply(src)))
            keep = d < gate
            if keep.sum() < 8:
                break
            try:
                sim = umeyama(src[keep], gt[j[keep]])
            except ValueError:
                break
        d, _ = map(np.asarray, tree.query(sim.apply(src)))
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
    """[start, stop) of ``k`` contiguous side-local frames centred on the camera nearest a tree.

    ``cam_y`` has one entry per side frame. The window always has exactly ``k`` frames inside
    ``[0, side_len)``; a side shorter than ``k`` raises instead of returning a negative start.
    """
    cam_y = np.asarray(cam_y, float)
    if k < 1:
        raise ValueError(f"window size must be >= 1, got {k}")
    if side_len < k:
        raise ValueError(f"side has {side_len} frames, fewer than the window size {k}")
    if len(cam_y) != side_len:
        raise ValueError(f"cam_y has {len(cam_y)} entries for a side of {side_len} frames")
    if not np.all(np.isfinite(cam_y)):
        raise ValueError("cam_y contains non-finite values")
    c = int(np.argmin(np.abs(cam_y - tree_y)))
    start = min(max(c - k // 2, 0), side_len - k)
    return start, start + k


class AlignmentError(RuntimeError):
    """The window's reconstruction does not fit its overview pseudo-poses."""


MAX_CAM_RMS_FRAC = 0.5  # camera RMS misfit above this share of the window's camera span fails
MAX_ROT_ERR_DEG = 30.0  # median camera orientation error above this fails (set a priori, not tuned)


def registered_views(extrinsics: np.ndarray) -> np.ndarray:
    """Mask of views with a camera. Sparse backbones (COLMAP) leave unregistered views all zero."""
    ext = np.asarray(extrinsics, float)
    return np.asarray(np.abs(ext[:, :3, :3]).sum((1, 2)) > 0)


def align_window(gt_ext: np.ndarray, recon_ext: np.ndarray) -> Sim3:
    """``align_chunk`` over the registered views only (identical to it when all are)."""
    reg = registered_views(recon_ext)
    if int(reg.sum()) < MIN_OVERLAP:
        raise ChunkAlignmentError(f"only {int(reg.sum())} registered views")
    return align_chunk(gt_ext[reg], recon_ext[reg])


def align_to_reference(ref_ext: np.ndarray, recon_ext: np.ndarray) -> tuple[Sim3, float]:
    """Sim(3) placing another backbone's window into the reference window's *aligned* frame.

    Same fit as ``align_window``. Also returns the median orientation residual in degrees."""
    sim = align_window(ref_ext, recon_ext)
    reg = registered_views(recon_ext)
    resid = rotation_errors_deg(np.asarray(ref_ext)[reg], np.asarray(recon_ext)[reg], sim)
    return sim, float(np.median(resid))


def rotation_errors_deg(gt_ext: np.ndarray, recon_ext: np.ndarray, sim: Sim3) -> np.ndarray:
    """Per-view camera orientation error of the aligned reconstruction, in degrees."""
    aligned = transform_extrinsics(recon_ext, sim)
    return rotation_angles_deg(np.asarray(gt_ext, float)[:, :3, :3], aligned[:, :3, :3])


def scaled_depth(depth: np.ndarray, sim: Sim3) -> np.ndarray:
    """Depth maps in GT-frame metres. ``transform_extrinsics`` expresses camera coordinates in
    destination units (``s * x_c``), so reconstruction-frame depth must be multiplied by ``s``
    to be comparable with the ``z`` that ``project_points`` returns for GT apples."""
    return np.asarray(depth, float) * sim.s


def check_alignment(
    gt_ext: np.ndarray,
    recon_ext: np.ndarray,
    sim: Sim3,
    max_frac: float = MAX_CAM_RMS_FRAC,
    max_rot_deg: float = MAX_ROT_ERR_DEG,
) -> dict[str, float]:
    """Camera misfit of the aligned window against the overview pseudo-poses.

    Returns ``cam_rms_m``, ``cam_span_m`` and ``cam_rot_deg`` (median orientation error); raises
    ``AlignmentError`` when the centre misfit is a large share of the span (meaningless scale),
    the orientation error exceeds ``max_rot_deg`` (the depth test and GT would be built in a
    rotated frame) or the scale is not finite and positive."""
    if not np.isfinite(sim.s) or sim.s <= 0:
        raise AlignmentError(f"non-positive alignment scale {sim.s}")
    c_gt = camera_centers(gt_ext)
    c_al = camera_centers(transform_extrinsics(recon_ext, sim))
    rms = float(np.sqrt(((c_gt - c_al) ** 2).sum(1).mean()))
    span = float(np.linalg.norm(np.ptp(c_gt, axis=0)))
    if span > 0 and rms > max_frac * span:
        raise AlignmentError(f"camera misfit {rms:.3f} m is >{max_frac:.0%} of span {span:.3f} m")
    rot = float(np.median(rotation_errors_deg(gt_ext, recon_ext, sim)))
    if not rot <= max_rot_deg:
        raise AlignmentError(f"camera rotation error {rot:.1f} deg is >{max_rot_deg:g} deg")
    return {"cam_rms_m": rms, "cam_span_m": span, "cam_rot_deg": rot}


def _frame_gt(
    gt_xyz: np.ndarray, intr: np.ndarray, ext: np.ndarray, hw: tuple[int, int], max_depth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(indices, uv, z) of GT apples in front of the camera, within ``max_depth`` and the image."""
    from bench.gt_matching import project_points

    h, w = hw
    uv, z = project_points(gt_xyz, intr, ext)
    ok = np.isfinite(z) & (z > 0) & (z < max_depth)
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    idx = np.flatnonzero(ok)
    return idx, uv, z


def visible_gt(
    gt_xyz: np.ndarray,
    intr: np.ndarray,
    ext_gt: np.ndarray,
    depth_gt: np.ndarray,
    tol: float = GT_TOL_OCC_M,
    max_depth: float = GT_MAX_DEPTH_M,
) -> list[np.ndarray]:
    """Per view: indices of GT apples in frame, in front of the reconstructed surface (metres).

    ``depth_gt`` must already be in GT metres (``scaled_depth``). The depth map is read at the
    nearest pixel (not truncated), including the last row and column. A pixel with no valid depth
    (NaN, inf or <= 0) carries no occlusion evidence, so the apple stays visible there.
    """
    gt_xyz = np.asarray(gt_xyz, float).reshape(-1, 3)
    out = []
    h, w = depth_gt.shape[1:]
    for v in range(len(intr)):
        idx, uv, z = _frame_gt(gt_xyz, intr[v], ext_gt[v], (h, w), max_depth)
        px = np.clip(np.rint(uv[idx, 0]).astype(int), 0, w - 1)
        py = np.clip(np.rint(uv[idx, 1]).astype(int), 0, h - 1)
        surf = depth_gt[v, py, px]
        valid = np.isfinite(surf) & (surf > 0)
        out.append(idx[~valid | (z[idx] <= surf + tol)])
    return out


def evaluate_window(
    result: Any,
    recon: Any,
    sim: Sim3,
    gt_ids: list[str],
    gt_xyz: np.ndarray,
    box: tuple[np.ndarray, np.ndarray],
    gt_ext: np.ndarray | None = None,
    visible: list[np.ndarray] | None = None,
    match_px: float = MATCH_PX,
) -> dict[str, Any]:
    """Count, GT, association metrics and per-image baselines of one run, in the GT frame.

    ``gt_ext`` (the window's overview pseudo-poses), when given, is checked against the aligned
    cameras (``check_alignment``) and the misfit is recorded in the result.

    ``visible`` (per-view GT indices) replaces this reconstruction's own depth test. Another
    backbone is scored against the reference window's GT this way, so both share one GT.
    ``match_px`` is in this reconstruction's pixels.
    """
    reg = registered_views(recon.extrinsics)
    align = check_alignment(gt_ext[reg], recon.extrinsics[reg], sim) if gt_ext is not None else {}
    ext_gt = transform_extrinsics(recon.extrinsics, sim)
    if visible is None:
        vis = visible_gt(gt_xyz, recon.intrinsics, ext_gt, scaled_depth(recon.depth, sim))
    else:
        vis = visible
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
    # Identity labels: match each detection to the nearest GT apple *in frame* (visible or not),
    # then keep the label only if that apple is in the visible set. A detection on an occluded
    # apple is therefore unlabeled instead of being stolen by a different, visible neighbour.
    labels: dict[str, str | None] = {}
    for v in range(len(vis)):
        dv = [d for d in dets if d.frame_id == v]
        if not dv:
            continue
        inframe, _, _ = _frame_gt(gt_xyz, recon.intrinsics[v], ext_gt[v], (h, w), GT_MAX_DEPTH_M)
        visible_v = {int(i) for i in vis[v]}
        got = match_detections_to_gt(
            dv,
            [gt_ids[i] for i in inframe],
            gt_xyz[inframe],
            {v: recon.intrinsics[v]},
            {v: ext_gt[v]},
            {v: (w, h)},
            threshold_px=match_px,
        )
        pos = {gt_ids[i]: int(i) for i in inframe}
        labels.update(
            {k: (g if g is not None and pos[g] in visible_v else None) for k, g in got.items()}
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

    # Interval: the pipeline's covers every instance, ours counts only those in the ROI, so shift
    # both ends by the number of out-of-ROI instances and keep low <= pred <= high.
    low = high = None
    iv = getattr(result, "interval", None)
    if iv is not None:
        out_of_roi = int(result.count) - pred
        low = float(max(min(iv.low - out_of_roi, pred), 0))
        high = float(max(iv.high - out_of_roi, pred))
    return {
        **align,
        "low": low,
        "high": high,
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


def file_hash(path: Path) -> str:
    """Short sha256 of a file's bytes (checkpoint provenance in config_json)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


PROTOCOL = "fuji_core_v2"

# Pipeline method name per association. All of them read the same cached detections and
# reconstruction of a window, so their upstream inputs are identical.
ASSOC_METHODS: dict[str, str] = {
    "v1": "countonce_v1",
    "v2": "countonce_v2",
    "sync": "countonce_sync",
    "v3": "countonce_v3",
    "cdog": "countonce_cdog",  # C-DOG re-implementation, bench/baselines_cdog.py
}
BACKBONES = ("vggt", "colmap")
METHOD_ASSOC: dict[str, str] = {m: a for a, m in ASSOC_METHODS.items()}
METHOD_ASSOC.update({f"{m}_colmap": a for a, m in ASSOC_METHODS.items()})
BASELINE_METHODS = ("sum", "max", "median")


def method_for_assoc(assoc: str, backbone: str = "vggt") -> str:
    """Bench method name; VGGT (the default backbone) carries no suffix."""
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")
    try:
        return ASSOC_METHODS[assoc] + ("" if backbone == "vggt" else f"_{backbone}")
    except KeyError:
        known = sorted(ASSOC_METHODS)
        raise ValueError(f"unknown assoc {assoc!r}; expected one of {known}") from None


def methods_for(assocs: Any, baselines: bool = True, backbone: str = "vggt") -> tuple[str, ...]:
    """Method names to schedule for ``assocs`` (order kept, duplicates dropped), plus the
    per-image baselines when ``baselines`` (finished slices are skipped on resume anyway)."""
    out = tuple(dict.fromkeys(method_for_assoc(a, backbone) for a in assocs))
    # per-image baselines do not use a reconstruction, so only the VGGT run records them
    return out + BASELINE_METHODS if baselines and backbone == "vggt" else out


def _row_protocol(r: dict[str, Any]) -> str | None:
    try:
        protocol: str | None = json.loads(r.get("config_json") or "{}").get("protocol")
        return protocol
    except ValueError:
        return None


def done_slices(csv_path: Path, split: str) -> set[tuple[str, str, int, str]]:
    """(tree, side, k, method) of every finished row of ``split`` in the CSV: status ``ok``, or
    ``failed`` (a recorded reconstruction failure is a result, PLAN §9.8; ``error`` is retried).

    Only rows of the current protocol count, so rows from an older scoring protocol are
    never reused as finished work."""
    return {
        (r["scene_id"], r["sample"], int(r["n_views"]), r["method"])
        for r in read_rows(csv_path)
        if r["status"] in ("ok", "failed")
        and r["split"] == split
        and r["n_views"]
        and _row_protocol(r) == PROTOCOL
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
        low=ev.get("low") if is_pipe else None,
        high=ev.get("high") if is_pipe else None,
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
                d, j = map(np.asarray, tree.query(sim.apply(src)))
                keep = d <= np.quantile(d, trim)
                try:
                    sim = umeyama(src[keep], dst[j[keep]])
                except ValueError:
                    break
            d, _ = map(np.asarray, tree.query(sim.apply(src)))
            n_in = int((d < 0.05).sum())
            med = float(np.median(d[d < 0.05])) if n_in else float("inf")
            reg = Registration(sim, n_in, len(src), n_in / len(src), med, cand)
            cand += 1
            if best is None or reg.n_inliers > best.n_inliers:
                best = reg
    assert best is not None
    return best
