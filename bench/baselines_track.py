"""Tracker baselines for experiment E1 (docs/RELATED_WORK.md §4): ByteTrack and SAM 2.1 propagation.

Both consume the *same* detections CountOnce associates (boxes + scores, one list per frame, in
the order the frames are given) and return one track id per detection. A tracker's count is the
number of distinct track ids among the detections a window's region of interest keeps
(``count_tracks``), so it is scored exactly like a CountOnce instance count.

ByteTrack is a small re-implementation (IoU + constant-velocity Kalman, two-stage matching after
Zhang et al., ECCV 2022). No licensed pip package was used; see docs/DEVIATIONS.md.

SAM 2.1 runs through ``transformers`` ``Sam2VideoModel`` in streaming mode. A detection whose box
is not already covered by a propagated mask starts a new object (box prompt); covered ones join the
best-covering track. ``torch``/``transformers`` are imported lazily so the module stays importable
(and the ByteTrack tests runnable) without the GPU extras.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# one frame of detections: boxes (n,4) xyxy, scores (n,)
FrameDets = tuple[np.ndarray, np.ndarray]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, shape (len(a), len(b))."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def count_tracks(track_ids: Sequence[int], keep: Sequence[bool]) -> int:
    """Distinct track ids among the detections flagged ``keep`` (inside the region of interest)."""
    return len({int(t) for t, k in zip(track_ids, keep, strict=True) if k and t >= 0})


# ----------------------------------------------------------------------------- ByteTrack


def _xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def _xyah_to_xyxy(x: np.ndarray) -> np.ndarray:
    w = x[2] * x[3]
    return np.array([x[0] - w / 2, x[1] - x[3] / 2, x[0] + w / 2, x[1] + x[3] / 2])


class _Kalman:
    """Constant-velocity Kalman filter on (cx, cy, aspect, h), as in ByteTrack/SORT."""

    def __init__(self) -> None:
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def init(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * h / 20, 2 * h / 20, 1e-2, 2 * h / 20, 10 * h / 160, 10 * h / 160, 1e-5,
               10 * h / 160]  # fmt: skip
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        q = np.diag(np.square([h / 20, h / 20, 1e-2, h / 20, h / 160, h / 160, 1e-5, h / 160]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def update(
        self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        r = np.diag(np.square([h / 20, h / 20, 1e-1, h / 20]))
        s = self.H @ cov @ self.H.T + r
        k = cov @ self.H.T @ np.linalg.inv(s)
        return mean + k @ (z - self.H @ mean), (np.eye(8) - k @ self.H) @ cov


@dataclass
class _Track:
    tid: int
    mean: np.ndarray
    cov: np.ndarray
    last_frame: int
    box: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass(frozen=True)
class ByteTrackParams:
    """Thresholds tuned on the training trees only (see ``scripts/run_e1_tracker.py tune``)."""

    high_thr: float = 0.5  # detections above this start tracks and match first
    low_thr: float = 0.2  # detections in [low_thr, high_thr) only rescue unmatched tracks
    match_iou: float = 0.2  # first-stage IoU gate
    second_iou: float = 0.5  # second-stage (low-score) IoU gate
    max_lost: int = 30  # frames a track survives without a match


def _assign(iou: np.ndarray, gate: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    from scipy.optimize import linear_sum_assignment

    if iou.size == 0:
        return [], list(range(iou.shape[0])), list(range(iou.shape[1]))
    rows, cols = linear_sum_assignment(1.0 - iou)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= gate]
    mr = {r for r, _ in pairs}
    mc = {c for _, c in pairs}
    return (
        pairs,
        [r for r in range(iou.shape[0]) if r not in mr],
        [c for c in range(iou.shape[1]) if c not in mc],
    )


def _update(kf: _Kalman, tr: _Track, box: np.ndarray, t: int) -> None:
    tr.mean, tr.cov = kf.update(tr.mean, tr.cov, _xyxy_to_xyah(box))
    tr.box = box
    tr.last_frame = t


def bytetrack(
    frames: Sequence[FrameDets], params: ByteTrackParams | None = None
) -> list[np.ndarray]:
    """Track ids per detection, one int array per frame (-1 for a detection no track owns).

    Frames are processed in the given order; a frame gap is one step (sparse views are the point).
    """
    p = params or ByteTrackParams()
    kf = _Kalman()
    tracks: list[_Track] = []
    next_id = 0
    out: list[np.ndarray] = []
    for t, (boxes, scores) in enumerate(frames):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        ids = np.full(len(boxes), -1, dtype=int)
        tracks = [tr for tr in tracks if t - tr.last_frame <= p.max_lost + 1]
        for tr in tracks:
            tr.mean, tr.cov = kf.predict(tr.mean, tr.cov)
            tr.box = _xyah_to_xyxy(tr.mean[:4])
        high = np.flatnonzero(scores >= p.high_thr)
        low = np.flatnonzero((scores >= p.low_thr) & (scores < p.high_thr))

        pairs, un_tr, un_high = _assign(
            iou_matrix(np.array([tr.box for tr in tracks]), boxes[high]), p.match_iou
        )
        for ti, di in pairs:
            _update(kf, tracks[ti], boxes[int(high[di])], t)
            ids[int(high[di])] = tracks[ti].tid
        # second stage: tracks seen in the previous frame vs low-score detections
        rest = [i for i in un_tr if tracks[i].last_frame == t - 1]
        pairs2, _, _ = _assign(
            iou_matrix(np.array([tracks[i].box for i in rest]), boxes[low]), p.second_iou
        )
        for ri, di in pairs2:
            _update(kf, tracks[rest[ri]], boxes[int(low[di])], t)
            ids[int(low[di])] = tracks[rest[ri]].tid
        for di in un_high:
            j = int(high[di])
            mean, cov = kf.init(_xyxy_to_xyah(boxes[j]))
            tr = _Track(next_id, mean, cov, t, boxes[j])
            ids[j] = next_id
            next_id += 1
            tracks.append(tr)
        out.append(ids)
    return out


# ----------------------------------------------------------------------------- SAM 2.1


@dataclass(frozen=True)
class Sam2Params:
    """``cover_iou``: a box joins a track when mask/box IoU reaches it (tuned on training trees)."""

    cover_iou: float = 0.3
    model_id: str = "facebook/sam2.1-hiera-small"


def mask_box_iou(masks: Any, boxes: np.ndarray) -> np.ndarray:
    """IoU between each box and each binary mask (``masks``: (n, H, W) bool tensor), (n_box, n)."""
    import torch

    n = masks.shape[0]
    out = np.zeros((len(boxes), n))
    if n == 0 or len(boxes) == 0:
        return out
    area = masks.flatten(1).sum(1).to(torch.float32)
    h, w = masks.shape[1:]
    for i, b in enumerate(boxes):
        x0, y0 = max(int(np.floor(b[0])), 0), max(int(np.floor(b[1])), 0)
        x1, y1 = min(int(np.ceil(b[2])), w), min(int(np.ceil(b[3])), h)
        if x1 <= x0 or y1 <= y0:
            continue
        inter = masks[:, y0:y1, x0:x1].flatten(1).sum(1).to(torch.float32)
        barea = float((b[2] - b[0]) * (b[3] - b[1]))
        union = area + barea - inter
        out[i] = (inter / union.clamp(min=1.0)).cpu().numpy()
    return out


class Sam2Tracker:
    """Box-prompted SAM 2.1 propagation over frames in the given order, streaming."""

    def __init__(self, params: Sam2Params | None = None, device: str = "cuda") -> None:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.params = params or Sam2Params()
        self.device = device
        self.dtype = torch.bfloat16
        self.model = Sam2VideoModel.from_pretrained(self.params.model_id).to(device, self.dtype)
        self.model.eval()
        self.processor = Sam2VideoProcessor.from_pretrained(self.params.model_id)

    def track(self, images: Sequence[Any], frames: Sequence[FrameDets]) -> list[np.ndarray]:
        """Track ids per detection (``images``: PIL frames, boxes in those pixels)."""
        import torch

        sess = self.processor.init_video_session(
            inference_device=self.device,
            inference_state_device="cpu",
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        n_obj = 0
        out: list[np.ndarray] = []
        started = False
        idx = 0
        for img, (boxes, scores) in zip(images, frames, strict=True):
            boxes = np.asarray(boxes, float).reshape(-1, 4)
            ids = np.full(len(boxes), -1, dtype=int)
            if len(boxes) == 0 and not started:
                out.append(ids)
                continue
            inp = self.processor(images=img, device=self.device, return_tensors="pt")
            osz = inp.original_sizes[0].tolist()
            masks = None
            with torch.no_grad():
                if started:
                    res = self.model(
                        inference_session=sess, frame=inp.pixel_values[0].to(self.dtype)
                    )
                    masks = self.processor.post_process_masks(
                        [res.pred_masks], original_sizes=[osz], binarize=True
                    )[0]
                    masks = masks.reshape(-1, *masks.shape[-2:]).bool()
                    cur = idx
                    idx += 1
                else:
                    cur = None
            cover = (
                mask_box_iou(masks, boxes)
                if masks is not None and len(boxes)
                else np.zeros((len(boxes), 0))
            )
            new = []
            for j in range(len(boxes)):
                if cover.shape[1] and cover[j].max() >= self.params.cover_iou:
                    ids[j] = int(cover[j].argmax())  # obj index == track id
                else:
                    new.append(j)
            if new:
                first = cur is None
                fidx = 0 if first else cur
                new_ids = list(range(n_obj, n_obj + len(new)))
                with torch.no_grad():
                    if first:
                        # the very first frame is registered by the streaming forward below
                        pass
                    self.processor.add_inputs_to_inference_session(
                        inference_session=sess,
                        frame_idx=fidx if not first else idx,
                        obj_ids=list(new_ids),
                        input_boxes=[[boxes[j].tolist() for j in new]],
                        original_size=tuple(osz),
                    )
                    if first:
                        self.model(inference_session=sess, frame=inp.pixel_values[0].to(self.dtype))
                        cur = idx
                        idx += 1
                        started = True
                    else:
                        self.model(inference_session=sess, frame_idx=cur)
                for j, o in zip(new, new_ids, strict=True):
                    ids[j] = o
                n_obj += len(new)
            out.append(ids)
            del scores
        return out
