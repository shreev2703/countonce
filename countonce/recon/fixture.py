"""Fixture backbone (TEST BACKEND ONLY, see docs/DEVIATIONS.md).

Serves the known cameras and per-view depth stored in a ``scene.json`` (written by
``examples/tiny_scene/generate.py``) instead of running a reconstruction model, so the real CLI
runs end to end on CPU with no weights. Views are matched to frames by image file name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from countonce.recon.adapter import unproject_depth
from countonce.types import Frame
from countonce.types import Recon as DenseRecon


class FixtureReconstructor:
    name = "fixture"
    preprocess_mode: str | None = None

    def __init__(self, scene: str | Path) -> None:
        self.scene_path = Path(scene)
        raw = self.scene_path.read_bytes()
        self._sha = hashlib.sha256(raw).hexdigest()
        self._scene: dict[str, Any] = json.loads(raw)

    def config(self) -> dict[str, Any]:
        return {"scene_sha256": self._sha}

    @property
    def gt_count(self) -> int:
        return int(self._scene["gt_count"])

    def reconstruct(self, frames: list[Frame]) -> DenseRecon:
        by_name = {v["image"]: v for v in self._scene["views"]}
        views = []
        for f in frames:
            if f.path.name not in by_name:
                raise ValueError(f"{f.path.name} is not a view in {self.scene_path}")
            views.append(by_name[f.path.name])
        k = np.array([v["intrinsics"] for v in views], dtype=np.float64)
        e = np.array([v["extrinsics"] for v in views], dtype=np.float64)
        depth = np.array([v["depth"] for v in views], dtype=np.float32)
        h, w = depth.shape[1:]
        if any((f.height, f.width) != (h, w) for f in frames):
            raise ValueError("frame size does not match the fixture depth maps")
        conf = (depth > 0).astype(np.float32)  # 0 where the render hit nothing: lift skips it
        return DenseRecon(
            backbone="fixture",
            intrinsics=k,
            extrinsics=e,
            depth=depth,
            depth_conf=conf,
            world_points=unproject_depth(depth, k, e),
            world_points_conf=conf.copy(),
        )
