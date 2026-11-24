"""Render the tiny bundled scene: coloured spheres seen from known cameras, with numpy only.

Writes ``view_*.png`` and ``scene.json`` (intrinsics, world->camera extrinsics in OpenCV
convention, per-view z-depth, ground-truth count of red spheres) next to this script. The
committed outputs are the product of ``python examples/tiny_scene/generate.py``; the render is
deterministic, so re-running reproduces them byte for byte.

Test backend data only: it exists to drive ``countonce run`` end to end on CPU without weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

W, H, FOCAL = 128, 96, 120.0
N_VIEWS = 5
RADIUS = 0.22
# Red spheres are the counted class; blue and green ones are distractors for the prompt.
RED = (220, 30, 30)
SPHERES: list[tuple[tuple[float, float, float], tuple[int, int, int], str]] = [
    ((1.00, 0.00, RADIUS), RED, "red"),
    ((0.31, 0.95, RADIUS), RED, "red"),
    ((-0.81, 0.59, RADIUS), RED, "red"),
    ((-0.81, -0.59, RADIUS), RED, "red"),
    ((0.31, -0.95, RADIUS), RED, "red"),
    ((0.00, 0.00, RADIUS), (30, 60, 220), "blue"),
    ((0.35, 0.20, RADIUS), (30, 200, 60), "green"),
]
LIGHT = np.array([0.3, -0.4, 0.86]) / np.linalg.norm([0.3, -0.4, 0.86])
BACKGROUND = (190, 190, 190)


def camera(azimuth_deg: float, radius: float = 4.0, height: float = 2.0) -> np.ndarray:
    """World->camera [R|t] (OpenCV: x right, y down, z forward) looking at the origin."""
    a = np.deg2rad(azimuth_deg)
    centre = np.array([radius * np.cos(a), radius * np.sin(a), height])
    forward = -centre / np.linalg.norm(centre)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward])
    return np.hstack([rot, (-rot @ centre)[:, None]])


def render(ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an (H,W,3) uint8 image and an (H,W) z-depth map (0 where nothing is hit)."""
    rot, trans = ext[:, :3], ext[:, 3]
    centre = -rot.T @ trans
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays_cam = np.stack([(u - (W - 1) / 2) / FOCAL, (v - (H - 1) / 2) / FOCAL, np.ones_like(u)], -1)
    rays = rays_cam @ rot  # camera -> world (row-vector form of R^T d)
    img = np.empty((H, W, 3), dtype=np.float64)
    img[:] = BACKGROUND
    best_t = np.full((H, W), np.inf)
    depth = np.zeros((H, W))
    for (cx, cy, cz), colour, _ in SPHERES:
        c = np.array([cx, cy, cz])
        oc = centre - c
        a = (rays * rays).sum(-1)
        b = 2 * (rays @ oc)
        disc = b * b - 4 * a * (oc @ oc - RADIUS**2)
        hit = disc >= 0
        t = np.where(hit, (-b - np.sqrt(np.where(hit, disc, 0))) / (2 * a), np.inf)
        vis = hit & (t > 0) & (t < best_t)
        best_t = np.where(vis, t, best_t)
        pt = centre + rays * np.where(vis, t, 0.0)[..., None]
        normal = (pt - c) / RADIUS
        shade = 0.35 + 0.65 * np.clip(normal @ LIGHT, 0, 1)
        img[vis] = (np.asarray(colour) * shade[..., None])[vis]
        depth[vis] = t[vis]  # |ray_cam.z| = 1, so t is the z-depth
    return np.clip(img, 0, 255).astype(np.uint8), depth


def main(out: Path | None = None) -> None:
    out = out or Path(__file__).resolve().parent
    k = [[FOCAL, 0.0, (W - 1) / 2], [0.0, FOCAL, (H - 1) / 2], [0.0, 0.0, 1.0]]
    views = []
    for i in range(N_VIEWS):
        ext = camera(360.0 * i / N_VIEWS + 20.0)
        img, depth = render(ext)
        name = f"view_{i}.png"
        Image.fromarray(img).save(out / name, optimize=True)
        views.append(
            {
                "image": name,
                "intrinsics": k,
                "extrinsics": np.round(ext, 6).tolist(),
                "depth": np.round(depth, 3).tolist(),
            }
        )
    scene = {
        "format": "countonce-tiny-scene-1",
        "image_size": [W, H],
        "gt_count": sum(1 for s in SPHERES if s[2] == "red"),
        "spheres": [{"xyz": list(c), "radius": RADIUS, "colour": name} for c, _, name in SPHERES],
        "views": views,
    }
    (out / "scene.json").write_text(json.dumps(scene, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
