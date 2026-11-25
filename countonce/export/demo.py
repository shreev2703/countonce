"""Static demo assets for the web app (PLAN §0.2, M13).

``build_scene`` runs the real pipeline on one scene and writes everything the demo viewer needs
into ``<out>/<scene_id>/``; ``check_demo`` re-validates a built tree (versions, hashes, schemas).
No server is needed at view time: every file is static.

Files per scene: ``result.json`` and ``markers.json`` (versioned wire schemas), ``points.bin``
(compact point cloud, see ``write_points_bin``), ``overlays/`` (annotated full frames),
``thumbs/`` (small copies), ``evidence.json`` (match evidence for "Explain this match"),
``provenance.json`` and ``manifest.json`` (versions + sha256 of every file).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from countonce.export.ply import sample_points
from countonce.schemas import CURRENT_VERSION, load_markers, load_result
from countonce.types import Frame
from countonce.types import Recon as DenseRecon

DEMO_FORMAT_VERSION = "1"
POINTS_MAGIC = b"CPC1"
POINTS_FORMAT_VERSION = 1
POINTS_HEADER = struct.Struct("<4sII")  # magic, format version, n points
THUMB_WIDTH = 160
MAX_POINTS = 60_000
ZOD_SCHEMA_FILE = Path("web/src/lib/schemas/result.ts")


class DemoVersionError(RuntimeError):
    """The demo assets, the backend schemas and the web schemas disagree on a version."""


def write_points_bin(xyz: np.ndarray, rgb: np.ndarray, path: Path) -> int:
    """``CPC1``: 12-byte header, then float32 xyz (n,3) and uint8 rgb (n,3), little endian."""
    xyz = np.ascontiguousarray(xyz, dtype="<f4")
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if xyz.shape != rgb.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz and rgb must both be (n,3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(POINTS_HEADER.pack(POINTS_MAGIC, POINTS_FORMAT_VERSION, len(xyz)))
        fh.write(xyz.tobytes())
        fh.write(rgb.tobytes())
    return len(xyz)


def read_points_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_bytes()
    magic, version, n = POINTS_HEADER.unpack_from(raw, 0)
    if magic != POINTS_MAGIC or version != POINTS_FORMAT_VERSION:
        raise DemoVersionError(f"{path}: bad points header {magic!r} v{version}")
    expect = POINTS_HEADER.size + n * 15
    if len(raw) != expect:
        raise ValueError(f"{path}: expected {expect} bytes for {n} points, got {len(raw)}")
    off = POINTS_HEADER.size
    xyz = np.frombuffer(raw, "<f4", n * 3, off).reshape(n, 3)
    rgb = np.frombuffer(raw, np.uint8, n * 3, off + n * 12).reshape(n, 3)
    return xyz, rgb


def frame_colors(recon: DenseRecon, frames: Sequence[Frame]) -> np.ndarray:
    """(N,H,W,3) uint8 colours sampled from the frame images at the recon's resolution."""
    from PIL import Image

    n, h, w = recon.world_points.shape[:3]
    by_id = {f.id: f for f in frames}
    ordered = [by_id[i] for i in sorted(by_id)][:n]
    out = np.full((n, h, w, 3), 128, dtype=np.uint8)
    for i, fr in enumerate(ordered):
        with Image.open(fr.path) as im:
            out[i] = np.asarray(im.convert("RGB").resize((w, h)))
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def web_schema_version(root: Path) -> str:
    """``CURRENT_VERSION`` declared by the Zod schemas in ``web/``."""
    text = (Path(root) / ZOD_SCHEMA_FILE).read_text(encoding="utf-8")
    m = re.search(r'export const CURRENT_VERSION = "([^"]+)"', text)
    if m is None:
        raise DemoVersionError(f"no CURRENT_VERSION in {ZOD_SCHEMA_FILE}")
    return m.group(1)


def assert_versions_match(root: Path) -> str:
    """Backend and web schema versions must be equal; returns the shared version."""
    web = web_schema_version(root)
    if web != CURRENT_VERSION:
        raise DemoVersionError(
            f"schema version mismatch: backend {CURRENT_VERSION!r} vs web {web!r} "
            f"({ZOD_SCHEMA_FILE}); update both together"
        )
    return web


def build_scene(
    scene_dir: Path,
    out_root: Path,
    *,
    scene_id: str,
    prompt: str,
    label: str,
    detector: str = "color_blob",
    backbone: str = "fixture",
    assoc: str = "v1",
    gt_count: int | None = None,
    extra_config: dict[str, Any] | None = None,
    repo_root: Path = Path("."),
) -> Path:
    """Run the pipeline on ``scene_dir`` and write the static assets; returns the scene folder."""
    from PIL import Image

    from countonce import pipeline
    from countonce.config import Config
    from countonce.export.writers import write_markers_json, write_result_json
    from countonce.lift import compute_scene_scale
    from countonce.provenance import run_metadata

    assert_versions_match(repo_root)
    scene_dir, out = Path(scene_dir), Path(out_root) / scene_id
    out.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {}
    if backbone == "fixture":
        options["scene"] = str(scene_dir / "scene.json")
    cfg = Config(
        detector=detector,
        backbone=backbone,
        assoc=assoc,
        out_dir=str(out / "_run"),
        backbone_options=options,
        **(extra_config or {}),
    )
    captured: dict[str, DenseRecon] = {}
    base = pipeline.build_reconstructor(cfg)

    class _Recording:
        name = base.name

        def reconstruct(self, frames: list[Frame]) -> DenseRecon:
            captured["recon"] = base.reconstruct(frames)
            return captured["recon"]

        def config(self) -> dict[str, Any]:
            return base.config()

    images = sorted(p for p in scene_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    result = pipeline.run(images, prompt, cfg, reconstructor=_Recording(), export=True)
    run_dir = out / "_run"

    # versioned wire files (re-written from the in-memory result so no run-dir paths leak)
    scale = compute_scene_scale(captured["recon"])
    write_result_json(result, out / "result.json")
    write_markers_json(result, out / "markers.json", scale)

    frames = pipeline.make_frames(images)
    xyz, rgb = sample_points(
        captured["recon"], MAX_POINTS, colors=frame_colors(captured["recon"], frames)
    )
    n_points = write_points_bin(xyz, rgb, out / "points.bin")

    (out / "overlays").mkdir(exist_ok=True)
    (out / "thumbs").mkdir(exist_ok=True)
    overlay_names = []
    for src in sorted((run_dir / "overlays").iterdir()):
        with Image.open(src) as raw:
            im = raw.convert("RGB")
            im.save(out / "overlays" / f"{src.stem}.png")
            h = max(1, round(im.height * THUMB_WIDTH / im.width))
            im.resize((THUMB_WIDTH, h)).save(out / "thumbs" / f"{src.stem}.jpg", quality=80)
        overlay_names.append(src.stem)
    shutil.rmtree(run_dir, ignore_errors=True)

    evidence = [
        {
            "det_a": e.det_a, "det_b": e.det_b, "dist_3d_norm": e.dist_3d_norm,
            "reproj_error_px": e.reproj_error_px, "appearance_sim": e.appearance_sim,
            "match_prob": e.match_prob, "merged": e.merged, "reason": e.reason,
        }
        for e in result.match_evidence
    ]  # fmt: skip
    _write_json(
        out / "evidence.json",
        {"schema_version": CURRENT_VERSION, "scene_id": scene_id, "evidence": evidence},
    )
    _write_json(
        out / "provenance.json",
        {
            "scene_id": scene_id,
            "synthetic": "synthetic" in label.lower(),
            "input_hash": result.input_hash,
            "inputs": {p.name: sha256_file(p) for p in images},
            "run": run_metadata(cfg.to_dict()),
            "model_versions": result.to_dict()["model_versions"],
        },
    )

    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    files = [f for f in files if f != "manifest.json"]
    _write_json(
        out / "manifest.json",
        {
            "demo_format_version": DEMO_FORMAT_VERSION,
            "schema_version": CURRENT_VERSION,
            "points_format_version": POINTS_FORMAT_VERSION,
            "scene_id": scene_id,
            "label": label,
            "prompt": prompt,
            "count": result.count,
            "gt_count": gt_count,
            "n_points": n_points,
            "frames": overlay_names,
            "files": {f: sha256_file(out / f) for f in files},
        },
    )
    check_scene(out)
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def check_scene(scene_dir: Path) -> None:
    """Validate one built scene folder; raises ``DemoVersionError`` / pydantic errors."""
    scene_dir = Path(scene_dir)
    man = json.loads((scene_dir / "manifest.json").read_text())
    if man["demo_format_version"] != DEMO_FORMAT_VERSION:
        raise DemoVersionError(f"{scene_dir}: demo_format_version {man['demo_format_version']}")
    if man["schema_version"] != CURRENT_VERSION:
        raise DemoVersionError(
            f"{scene_dir}: assets built for schema {man['schema_version']!r}, "
            f"backend is {CURRENT_VERSION!r}; rebuild with scripts/build_demo.py"
        )
    if man["points_format_version"] != POINTS_FORMAT_VERSION:
        raise DemoVersionError(f"{scene_dir}: points_format_version mismatch")
    for rel, digest in man["files"].items():
        if sha256_file(scene_dir / rel) != digest:
            raise DemoVersionError(f"{scene_dir}/{rel}: hash differs from manifest; rebuild")
    result = load_result(json.loads((scene_dir / "result.json").read_text()))
    markers = load_markers(json.loads((scene_dir / "markers.json").read_text()))
    if result.schema_version != CURRENT_VERSION or markers.schema_version != CURRENT_VERSION:
        raise DemoVersionError(f"{scene_dir}: result/markers schema_version is not current")
    if result.count != man["count"] or len(markers.markers) != result.count:
        raise DemoVersionError(f"{scene_dir}: count disagrees between manifest/result/markers")
    xyz, _ = read_points_bin(scene_dir / "points.bin")
    if len(xyz) != man["n_points"]:
        raise DemoVersionError(f"{scene_dir}: point count disagrees with manifest")


def check_demo(demo_root: Path, repo_root: Path = Path(".")) -> list[str]:
    """CI check: backend == web schema version, and every built scene is current and intact."""
    assert_versions_match(repo_root)
    scenes = sorted(p.parent.name for p in Path(demo_root).glob("*/manifest.json"))
    if not scenes:
        raise DemoVersionError(f"no demo scenes under {demo_root}; run scripts/build_demo.py")
    for s in scenes:
        check_scene(Path(demo_root) / s)
    return scenes
