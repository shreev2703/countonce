"""Error slicing (PLAN §9.9): per-scene metadata, per-slice metrics, significantly-worse flags."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from countonce.recon.align import camera_centers
from countonce.types import CountResult, Detection, Frame, Recon

from .metrics import pairwise_prf


@dataclass
class SceneMeta:
    """Per-scene covariates. Anything that cannot be computed from the inputs is NaN."""

    gt_count: float
    n_views: int
    density: float = float("nan")  # predicted instances per megapixel of image area
    mean_depth_conf: float = float("nan")
    luminance: float = float("nan")  # mean of per-image mean luminance, 0..1
    blur_score: float = float("nan")  # mean Frame.blur_score
    score_mean: float = float("nan")
    score_std: float = float("nan")
    score_min: float = float("nan")
    baseline_spread: float = float("nan")  # RMS camera-centre spread / scene_scale
    mean_occlusion: float = float("nan")  # 0..1, blank (NaN) without GT or identity labels

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def luminance_of(image: np.ndarray) -> float:
    """Mean Rec.601 luma of an HxWx3 (or HxW) image, scaled to [0, 1] (uint8 assumed if > 1)."""
    a = np.asarray(image, dtype=float)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
    return float(a.mean() / (255.0 if a.max() > 1.0 else 1.0))


def baseline_spread(recon: Recon) -> float:
    """RMS distance of camera centres from their centroid, divided by ``scene_scale``.

    Larger for a wide-arc rig than a narrow-arc one; 0 for a single view.
    """
    return centers_spread(camera_centers(recon.extrinsics), recon.scene_scale)


def centers_spread(centers: np.ndarray, scale: float = 1.0) -> float:
    """RMS distance of (N,3) camera centres from their centroid, divided by ``scale``."""
    c = np.asarray(centers, dtype=np.float64)
    scale = scale if scale > 0 else 1.0
    return float(np.sqrt(((c - c.mean(axis=0)) ** 2).sum(axis=1).mean()) / scale)


def mean_occlusion(visibility: Mapping[Hashable, Sequence[float]]) -> float:
    """Mean occlusion from synthetic GT: ``visibility[obj]`` = visible fraction per view in
    which the object is in frame. Occlusion = 1 - visible; NaN when there is no GT."""
    vals = [1.0 - float(np.clip(v, 0.0, 1.0)) for vs in visibility.values() for v in vs]
    return float(np.mean(vals)) if vals else float("nan")


def occlusion_from_identities(frame_ids: Sequence[int], gt_ids: Sequence[Hashable | None]) -> float:
    """Proxy from identity labels: per identity, share of views inside its first..last observed
    span (by frame index) in which it was not detected; averaged over identities. NaN if none."""
    views: dict[Hashable, set[int]] = {}
    for f, g in zip(frame_ids, gt_ids, strict=True):
        if g is not None:
            views.setdefault(g, set()).add(int(f))
    miss = [1.0 - len(v) / (max(v) - min(v) + 1) for v in views.values()]
    return float(np.mean(miss)) if miss else float("nan")


def scene_metadata(
    result: CountResult,
    *,
    gt_count: float,
    recon: Recon | None = None,
    detections: Sequence[Detection] | None = None,
    frames: Sequence[Frame] | None = None,
    images: Sequence[np.ndarray] | None = None,
    occlusion: float | None = None,
) -> SceneMeta:
    """Collect slicing covariates from a ``CountResult`` and optional recon/detections/frames."""
    n_views = len(frames) if frames is not None else len(result.per_frame_counts)
    if recon is not None:
        n_views = recon.depth.shape[0]
    meta = SceneMeta(gt_count=float(gt_count), n_views=int(n_views))
    if occlusion is not None:
        meta.mean_occlusion = float(occlusion)
    if recon is not None:
        meta.baseline_spread = baseline_spread(recon)
        meta.mean_depth_conf = float(np.mean(recon.depth_conf))
        area_mpx = recon.depth.shape[0] * recon.depth.shape[1] * recon.depth.shape[2] / 1e6
        meta.density = len(result.instances) / area_mpx if area_mpx else float("nan")
    elif frames:
        area = sum(f.width * f.height for f in frames) / 1e6
        meta.density = len(result.instances) / area if area else float("nan")
    if frames:
        meta.blur_score = float(np.mean([f.blur_score for f in frames]))
    if images:
        meta.luminance = float(np.mean([luminance_of(im) for im in images]))
    if detections:
        s = np.array([d.score for d in detections], dtype=float)
        meta.score_mean, meta.score_std, meta.score_min = (
            float(s.mean()),
            float(s.std()),
            float(s.min()),
        )
    return meta


@dataclass
class SceneRecord:
    """One scene's metadata plus its outcomes. ``assoc_f1`` is NaN when no identity labels."""

    scene_id: str
    meta: SceneMeta
    pred_count: float
    abs_error: float = field(init=False)
    assoc_f1: float = float("nan")

    def __post_init__(self) -> None:
        self.abs_error = abs(self.pred_count - self.meta.gt_count)


def make_record(
    scene_id: str,
    meta: SceneMeta,
    pred_count: float,
    clusters: Sequence[Hashable] | None = None,
    gt_ids: Sequence[Hashable | None] | None = None,
) -> SceneRecord:
    f1 = float("nan")
    if clusters is not None and gt_ids is not None:
        f1 = pairwise_prf(clusters, gt_ids)[2]
    return SceneRecord(scene_id, meta, float(pred_count), assoc_f1=f1)


# A slice is a named predicate over SceneMeta.
SliceSpec = Mapping[str, Callable[[SceneMeta], bool]]


def default_slices() -> dict[str, Callable[[SceneMeta], bool]]:
    """§9.9 examples: count buckets, depth confidence, view count, brightness, blur."""
    return {
        "count<20": lambda m: m.gt_count < 20,
        "count20-50": lambda m: 20 <= m.gt_count <= 50,
        "count>50": lambda m: m.gt_count > 50,
        "depth_conf<0.6": lambda m: m.mean_depth_conf < 0.6,
        "depth_conf>=0.6": lambda m: m.mean_depth_conf >= 0.6,
        "views<10": lambda m: m.n_views < 10,
        "views>=10": lambda m: m.n_views >= 10,
        "dark(lum<0.25)": lambda m: m.luminance < 0.25,
        "blurry(blur<0.3)": lambda m: m.blur_score < 0.3,
    }


@dataclass
class SliceResult:
    name: str
    n: int
    mae: float
    assoc_f1: float  # mean over scenes with labels, NaN if none
    err_diff: float  # slice MAE - overall MAE
    err_diff_ci: tuple[float, float]
    f1_diff: float
    f1_diff_ci: tuple[float, float]
    worse_error: bool  # error CI of the difference lies entirely above 0
    worse_f1: bool  # F1 CI of the difference lies entirely below 0

    @property
    def flagged(self) -> bool:
        return self.worse_error or self.worse_f1


def _diff_ci(
    sl: np.ndarray, allv: np.ndarray, n_boot: int, level: float, rng: np.random.Generator
) -> tuple[float, tuple[float, float]]:
    """Point estimate and percentile CI of mean(slice) - mean(all); NaNs dropped.

    Slice and overall are resampled independently-by-scene: the slice from its own scenes, the
    overall set from all scenes (the slice is a subset of overall, so this is conservative).
    """
    sl, allv = sl[~np.isnan(sl)], allv[~np.isnan(allv)]
    if sl.size == 0 or allv.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    bs = sl[rng.integers(0, sl.size, (n_boot, sl.size))].mean(axis=1)
    ba = allv[rng.integers(0, allv.size, (n_boot, allv.size))].mean(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(bs - ba, [a, 1.0 - a])
    return float(sl.mean() - allv.mean()), (float(lo), float(hi))


def slice_report(
    records: Sequence[SceneRecord],
    slices: SliceSpec | None = None,
    *,
    min_scenes: int = 5,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> list[SliceResult]:
    """Error and association F1 per slice, flagging slices significantly worse than overall.

    Slices with fewer than ``min_scenes`` scenes are reported but never flagged. NaN-valued
    covariates fail every comparison, so such scenes simply fall outside the slice.
    """
    if not records:
        raise ValueError("no records")
    specs = default_slices() if slices is None else slices
    err = np.array([r.abs_error for r in records])
    f1 = np.array([r.assoc_f1 for r in records])
    out: list[SliceResult] = []
    for i, (name, pred) in enumerate(specs.items()):
        mask = np.array([bool(pred(r.meta)) for r in records])
        n = int(mask.sum())
        if n == 0:
            continue
        rng = np.random.default_rng(seed + i)
        ed, eci = _diff_ci(err[mask], err, n_boot, level, rng)
        fd, fci = _diff_ci(f1[mask], f1, n_boot, level, rng)
        big = n >= min_scenes
        f1_slice = f1[mask][~np.isnan(f1[mask])]
        out.append(
            SliceResult(
                name=name,
                n=n,
                mae=float(err[mask].mean()),
                assoc_f1=float(f1_slice.mean()) if f1_slice.size else float("nan"),
                err_diff=ed,
                err_diff_ci=eci,
                f1_diff=fd,
                f1_diff_ci=fci,
                worse_error=bool(big and eci[0] > 0),
                worse_f1=bool(big and fci[1] < 0),
            )
        )
    return out


def flagged_slices(results: Sequence[SliceResult]) -> list[str]:
    return [r.name for r in results if r.flagged]
