"""Renderer-agnostic synthetic scene recipe (PLAN §10.2).

`make_recipe(seed)` draws a `SceneRecipe`: a pure description of objects, arrangement, occluders,
lighting, camera rig, distractors and domain-randomization knobs. It contains no renderer state, so
a Kubric or BlenderProc backend can both consume it. The same seed and `RECIPE_VERSION` always give
the same recipe (a private `numpy.random.Generator` is used, never global state).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

RECIPE_VERSION = "1.0"

Arrangement = Literal["scattered", "piled", "canopy", "shelf_rows"]
OccluderKind = Literal["leaf_plane", "branch", "box"]
RigKind = Literal["arc", "circle", "random_walk"]
ObjectShape = Literal["sphere", "ellipsoid", "asset", "clutter"]

ARRANGEMENTS: tuple[str, ...] = ("scattered", "piled", "canopy", "shelf_rows")
RIG_KINDS: tuple[str, ...] = ("arc", "circle", "random_walk")
COUNT_RANGE = (5, 150)
VIEW_RANGE = (3, 40)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectSpec(_Model):
    shape: ObjectShape
    asset: str  # asset-set key, e.g. "fruit/apple"; resolved by the renderer
    scale: float = Field(gt=0)
    texture_id: int = Field(ge=0)
    position: tuple[float, float, float]
    yaw: float


class OccluderSpec(_Model):
    kind: OccluderKind
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float


class DistractorSpec(_Model):
    asset: str
    scale: float = Field(gt=0)
    position: tuple[float, float, float]


class LightingSpec(_Model):
    hdri_id: int = Field(ge=0)
    sun_azimuth_deg: float
    sun_elevation_deg: float = Field(ge=0, le=90)
    sun_intensity: float = Field(gt=0)
    ambient_intensity: float = Field(ge=0)


class CameraSpec(_Model):
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fov_deg: float = Field(gt=0, lt=180)


class CameraRigSpec(_Model):
    kind: RigKind
    cameras: list[CameraSpec]
    resolution: tuple[int, int] = (518, 518)


class RandomizationSpec(_Model):
    """Domain-randomization knobs, logged per scene (PLAN §10.2)."""

    texture_variety: int = Field(ge=1)
    blur_sigma_px: float = Field(ge=0)
    noise_std: float = Field(ge=0)
    exposure_ev: float
    jpeg_quality: int = Field(ge=10, le=100)
    detection_drop_prob: float = Field(ge=0, le=1)
    false_positive_rate: float = Field(ge=0)


class SceneRecipe(_Model):
    recipe_version: str = RECIPE_VERSION
    seed: int = Field(ge=0)
    prompt: str
    arrangement: Arrangement
    objects: list[ObjectSpec]
    occluders: list[OccluderSpec]
    distractors: list[DistractorSpec]
    lighting: LightingSpec
    rig: CameraRigSpec
    randomization: RandomizationSpec

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @model_validator(mode="after")
    def _check_ranges(self) -> SceneRecipe:
        if not COUNT_RANGE[0] <= len(self.objects) <= COUNT_RANGE[1]:
            raise ValueError(f"object count {len(self.objects)} outside {COUNT_RANGE}")
        if not VIEW_RANGE[0] <= len(self.rig.cameras) <= VIEW_RANGE[1]:
            raise ValueError(f"view count {len(self.rig.cameras)} outside {VIEW_RANGE}")
        return self

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> SceneRecipe:
        return cls.model_validate_json(text)


_FRUIT = ("apple", "orange", "pear", "plum", "lemon")
_PROMPTS = {f: f + "s" for f in _FRUIT}


def _vec(a: np.ndarray) -> tuple[float, float, float]:
    return (float(a[0]), float(a[1]), float(a[2]))


def _positions(rng: np.random.Generator, arrangement: str, n: int, radius: float) -> np.ndarray:
    if arrangement == "scattered":
        xy = rng.uniform(-radius, radius, (n, 2))
        z = np.full(n, 0.05)
    elif arrangement == "piled":
        xy = rng.normal(0.0, radius * 0.3, (n, 2))
        z = 0.05 + 0.1 * rng.random(n) * np.sqrt(n) / 3.0
    elif arrangement == "canopy":
        xy = rng.uniform(-radius, radius, (n, 2))
        z = rng.uniform(0.8, 2.0, n)
    else:  # shelf_rows
        rows = max(1, int(np.sqrt(n) // 2))
        row = rng.integers(0, rows, n)
        xy = np.stack([rng.uniform(-radius, radius, n), (row - (rows - 1) / 2) * 0.4], axis=1)
        z = 0.3 + 0.5 * row
    return np.column_stack([xy, z])


def _rig(rng: np.random.Generator, kind: str, n_views: int, radius: float) -> list[CameraSpec]:
    dist = float(rng.uniform(2.5, 5.0)) * radius
    height = float(rng.uniform(0.5, 2.5))
    fov = float(rng.uniform(40.0, 75.0))
    target = (0.0, 0.0, 0.3)
    if kind == "circle":
        phase = rng.uniform(0, 2 * np.pi)
        ang = phase + 2 * np.pi * np.arange(n_views) / n_views
    elif kind == "arc":
        span = rng.uniform(np.pi / 4, np.pi)
        start = rng.uniform(0, 2 * np.pi)
        ang = start + np.linspace(0.0, span, n_views)
    else:  # random_walk
        ang = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0.0, 0.5, n_views))
    cams = []
    for i, a in enumerate(ang):
        d = dist * float(rng.uniform(0.85, 1.15)) if kind == "random_walk" else dist
        jitter = float(rng.normal(0, 3)) if kind == "random_walk" else 0.0
        h = height + (float(rng.normal(0, 0.2)) if kind == "random_walk" else 0.1 * np.sin(i))
        cams.append(
            CameraSpec(
                position=(float(d * np.cos(a)), float(d * np.sin(a)), float(max(h, 0.2))),
                look_at=target,
                fov_deg=float(np.clip(fov + jitter, 20, 100)),
            )
        )
    return cams


def make_recipe(
    seed: int,
    *,
    n_objects: int | None = None,
    n_views: int | None = None,
    arrangement: str | None = None,
    rig_kind: str | None = None,
) -> SceneRecipe:
    """Draw a recipe for `seed`; keyword arguments pin a field instead of drawing it."""
    rng = np.random.default_rng(seed)
    lo, hi = COUNT_RANGE
    n = int(n_objects if n_objects is not None else rng.integers(lo, hi + 1))
    v = int(n_views if n_views is not None else rng.integers(VIEW_RANGE[0], VIEW_RANGE[1] + 1))
    arr = arrangement or str(rng.choice(ARRANGEMENTS))
    kind = rig_kind or str(rng.choice(RIG_KINDS))
    radius = float(rng.uniform(0.6, 1.5))
    fruit = str(rng.choice(_FRUIT))
    tex_variety = int(rng.integers(1, 9))

    pos = _positions(rng, arr, n, radius)
    objects = [
        ObjectSpec(
            shape="ellipsoid" if rng.random() < 0.3 else "sphere",
            asset=f"fruit/{fruit}",  # many visually identical instances: one asset per scene
            scale=float(rng.uniform(0.04, 0.09)),
            texture_id=int(rng.integers(0, tex_variety)),
            position=_vec(pos[i]),
            yaw=float(rng.uniform(0, 2 * np.pi)),
        )
        for i in range(n)
    ]
    n_occ = int(rng.integers(0, max(2, n // 4) + 1))
    occluders = [
        OccluderSpec(
            kind=str(rng.choice(("leaf_plane", "branch", "box"))),
            position=_vec(np.array([*rng.uniform(-radius, radius, 2), rng.uniform(0.1, 1.5)])),
            size=_vec(rng.uniform(0.05, 0.4, 3)),
            yaw=float(rng.uniform(0, 2 * np.pi)),
        )
        for _ in range(n_occ)
    ]
    others = [f for f in _FRUIT if f != fruit]
    n_dis = int(rng.integers(0, max(2, n // 5) + 1))
    distractors = [
        DistractorSpec(
            asset=f"fruit/{rng.choice(others)}",
            scale=float(rng.uniform(0.04, 0.09)),
            position=_vec(np.array([*rng.uniform(-radius, radius, 2), 0.05])),
        )
        for _ in range(n_dis)
    ]
    lighting = LightingSpec(
        hdri_id=int(rng.integers(0, 100)),
        sun_azimuth_deg=float(rng.uniform(0, 360)),
        sun_elevation_deg=float(rng.uniform(10, 85)),
        sun_intensity=float(rng.uniform(0.5, 5.0)),
        ambient_intensity=float(rng.uniform(0.1, 1.0)),
    )
    rand = RandomizationSpec(
        texture_variety=tex_variety,
        blur_sigma_px=float(rng.uniform(0, 1.5)),
        noise_std=float(rng.uniform(0, 0.03)),
        exposure_ev=float(rng.uniform(-1.0, 1.0)),
        jpeg_quality=int(rng.integers(60, 96)),
        detection_drop_prob=float(rng.uniform(0.0, 0.3)),
        false_positive_rate=float(rng.uniform(0.0, 0.15)),
    )
    return SceneRecipe(
        seed=seed,
        prompt=_PROMPTS[fruit],
        arrangement=arr,
        objects=objects,
        occluders=occluders,
        distractors=distractors,
        lighting=lighting,
        rig=CameraRigSpec(kind=kind, cameras=_rig(rng, kind, v, radius)),
        randomization=rand,
    )
