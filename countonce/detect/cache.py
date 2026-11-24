"""Per-frame detection cache keyed by (frame hash, prompt, detector, version, config).

Stores every raw detection with its score so thresholds can be swept without
rerunning the model (PLAN section 8.2). Masks and features are not cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from countonce.detect.base import Detector, Exemplar
from countonce.types import Detection, Frame


def detection_cache_key(
    frame: Frame,
    prompt: str,
    detector: str,
    version: str,
    config: dict[str, object] | None = None,
    exemplars: Sequence[Exemplar] | None = None,
) -> str:
    payload = json.dumps(
        {
            "frame": frame.sha256,
            "prompt": prompt,
            "detector": detector,
            "version": version,
            "config": config or {},
            "exemplars": [[e.frame_id, list(e.box_xyxy)] for e in exemplars or []],
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _to_json(d: Detection) -> dict[str, object]:
    return {
        "score": d.score,
        "box": None if d.box_xyxy is None else [float(v) for v in d.box_xyxy],
        "point": None if d.point_xy is None else [float(v) for v in d.point_xy],
    }


class DetectionCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str, frame_id: int) -> list[Detection] | None:
        try:
            raw = json.loads(self.path(key).read_text())
            return [
                Detection(
                    id=Detection.make_id(frame_id, k),
                    frame_id=frame_id,
                    score=float(r["score"]),
                    box_xyxy=None if r["box"] is None else np.array(r["box"], dtype=np.float64),
                    point_xy=None if r["point"] is None else np.array(r["point"], dtype=np.float64),
                )
                for k, r in enumerate(raw)
            ]
        except (OSError, ValueError, KeyError, TypeError):
            return None  # missing or corrupt: a miss

    def put(self, key: str, dets: Sequence[Detection]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump([_to_json(d) for d in dets], fh)
        os.replace(tmp, self.path(key))
        return self.path(key)


class CachedDetector:
    """Wraps a detector; only frames with a cache miss are sent to the model."""

    def __init__(self, inner: Detector, cache: DetectionCache) -> None:
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.version = inner.version
        self.hits = 0
        self.misses = 0

    def config(self) -> dict[str, object]:
        return self.inner.config()

    def detect(
        self,
        frames: Sequence[Frame],
        prompt: str,
        exemplars: Sequence[Exemplar] | None = None,
    ) -> list[Detection]:
        keys = {
            f.id: detection_cache_key(
                f, prompt, self.name, self.version, self.inner.config(), exemplars
            )
            for f in frames
        }
        found: dict[int, list[Detection]] = {}
        todo: list[Frame] = []
        for f in frames:
            hit = self.cache.get(keys[f.id], f.id)
            if hit is None:
                todo.append(f)
            else:
                found[f.id] = hit
        self.hits += len(found)
        self.misses += len(todo)
        if todo:
            fresh = self.inner.detect(todo, prompt, exemplars)
            for f in todo:
                mine = [d for d in fresh if d.frame_id == f.id]
                self.cache.put(keys[f.id], mine)
                found[f.id] = mine
        return [d for f in frames for d in found[f.id]]
