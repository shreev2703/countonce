"""Benchmark dataset loaders (download scripts live in ``scripts/``; no data is committed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bench.datasets.common import BenchScene

LOADERS = ("rgbd_videocount", "videocount", "onescene", "applegrowthvision")


def load_dataset(name: str, root: str | Path, **kw: Any) -> list[BenchScene]:
    """Load one of the opt-in video / multi-view datasets by name."""
    if name == "rgbd_videocount":
        from bench.datasets.rgbd_videocount import load_rgbd_videocount

        return load_rgbd_videocount(root, **kw)
    if name == "videocount":
        from bench.datasets.videocount import load_videocount

        return load_videocount(root, **kw)
    if name == "onescene":
        from bench.datasets.onescene import load_onescene

        return load_onescene(root, **kw)
    if name == "applegrowthvision":
        from bench.datasets.applegrowthvision import load_applegrowthvision

        return load_applegrowthvision(root, **kw)
    raise ValueError(f"unknown dataset {name!r}; expected one of {LOADERS}")
