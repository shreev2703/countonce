#!/usr/bin/env python
"""Turn a phone folder into a benchmark scene: ``data/scenes/<id>/{images/, gt.json}``.

Every photo is re-encoded from decoded pixels (orientation applied first), so EXIF, GPS,
maker notes and ICC blobs are dropped, and files are renamed ``img_000.jpg`` ... in sorted
order so original filenames (which can carry dates or names) are not kept.

The ground truth follows PLAN §9.2: the number of target objects visible in at least one
submitted photo, counted by hand from the photos. Two counts are required; a third settles a
disagreement. ``gt.json`` records the definition and the hand counts.

    python scripts/ingest_captures.py --src ~/phone/bikes --scene-id bikes_01 \
        --prompt "bicycle" --counts 12 12 --tier easy --physical-scene bike_rack_north

    python scripts/ingest_captures.py --src ~/phone/empty --scene-id empty_01 \
        --prompt "bicycle" --counts 0 0 --zero --physical-scene empty_hallway

``--scene-list configs/capture_scenes.template.yaml`` fills prompt/tier/zero/physical scene
from the matching entry, so only ``--src``, ``--scene-id`` and ``--counts`` are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SKIPPED_EXTS = {".heic", ".heif"}  # need a plugin; reported, never silently dropped
TIERS = ("easy", "medium", "hard")
DEFINITION = (
    "count = number of target objects visible in at least one submitted photo "
    "(fully hidden objects are out of scope), counted by hand from the photos"
)
MIN_SIDE = 64


class IngestError(ValueError):
    """Bad input; the message says what to fix."""


def resolve_count(counts: list[int]) -> int:
    """Two agreeing counts, or a third that settles it. Anything else is an error."""
    if not 2 <= len(counts) <= 3:
        raise IngestError("give 2 hand counts (3 if the first two disagree)")
    if any(c < 0 for c in counts):
        raise IngestError("counts must be >= 0")
    if counts[0] == counts[1]:
        if len(counts) == 3 and counts[2] != counts[0]:
            raise IngestError("third count differs from two agreeing counts; recount the photos")
        return counts[0]
    if len(counts) < 3:
        raise IngestError(f"counts {counts[0]} and {counts[1]} disagree; add a third count")
    if counts[2] in counts[:2]:
        return counts[2]
    raise IngestError(f"all three counts differ {counts}; recount together")


def list_images(src: Path) -> tuple[list[Path], list[Path]]:
    files = sorted(p for p in src.iterdir() if p.is_file() and not p.name.startswith("."))
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    skipped = [p for p in files if p.suffix.lower() in SKIPPED_EXTS]
    return images, skipped


def strip_and_encode(path: Path, max_side: int | None) -> tuple[bytes, tuple[int, int]]:
    """Decode, apply EXIF orientation, drop all metadata, return JPEG bytes and (w, h)."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = im.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise IngestError(f"{path.name}: not a readable image ({e})") from e
    if max_side and max(rgb.size) > max_side:
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if min(rgb.size) < MIN_SIDE:
        raise IngestError(f"{path.name}: image smaller than {MIN_SIDE}px")
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)  # fresh image object: carries no info/EXIF from the source
    buf = io.BytesIO()
    clean.save(buf, format="JPEG", quality=95, exif=b"")
    return buf.getvalue(), rgb.size


def load_scene_entry(scene_list: Path | None, scene_id: str) -> dict[str, Any]:
    if scene_list is None:
        return {}
    doc = yaml.safe_load(scene_list.read_text()) or {}
    for entry in doc.get("scenes", []):
        if entry.get("id") == scene_id:
            return dict(entry)
    raise IngestError(f"scene id {scene_id!r} not in {scene_list}")


def ingest(
    src: Path,
    out_root: Path,
    scene_id: str,
    *,
    prompt: str,
    counts: list[int],
    tier: str,
    zero: bool,
    physical_scene: str,
    notes: str = "",
    max_side: int | None = 2048,
    force: bool = False,
) -> Path:
    if not src.is_dir():
        raise IngestError(f"source folder not found: {src}")
    if not scene_id or any(c in scene_id for c in "/\\ ") or scene_id.startswith("."):
        raise IngestError("scene id must be a simple folder name without spaces or slashes")
    if tier not in TIERS:
        raise IngestError(f"tier must be one of {TIERS}")
    if not prompt.strip():
        raise IngestError("a non-empty prompt (what is being counted) is required")
    if not physical_scene:
        raise IngestError("physical scene id is required (splits are by physical scene)")
    count = resolve_count(counts)
    if zero and count != 0:
        raise IngestError(f"scene is marked zero-target but the hand count is {count}")
    if count == 0 and not zero:
        raise IngestError("hand count is 0; pass --zero to confirm this is a zero-target scene")

    images, skipped = list_images(src)
    if skipped:
        names = ", ".join(p.name for p in skipped[:5])
        raise IngestError(f"HEIC/HEIF not supported ({names}); export from the phone as JPEG")
    if not images:
        raise IngestError(f"no images in {src} (looked for {sorted(IMAGE_EXTS)})")

    encoded = [strip_and_encode(p, max_side) for p in images]  # validate all before writing

    scene_dir = out_root / scene_id
    if scene_dir.exists():
        if not force:
            raise IngestError(f"{scene_dir} exists; pass --force to replace it")
        for f in (scene_dir / "images").glob("*") if (scene_dir / "images").is_dir() else []:
            f.unlink()
    img_dir = scene_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, (data, size) in enumerate(encoded):
        name = f"img_{i:03d}.jpg"
        (img_dir / name).write_bytes(data)
        manifest.append(
            {
                "file": name,
                "width": size[0],
                "height": size[1],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    gt = {
        "count": count,
        "prompt": prompt,
        "tier": tier,
        "zero_target": count == 0,
        "physical_scene": physical_scene,
        "n_images": len(manifest),
        "hand_counts": counts,
        "count_definition": DEFINITION,
        "exif_stripped": True,
        "notes": notes,
        "images": manifest,
    }
    (scene_dir / "gt.json").write_text(json.dumps(gt, indent=2) + "\n")
    return scene_dir


def validate_scene(scene_dir: Path) -> list[str]:
    """Problems with an ingested scene (empty list = valid). Also usable on old scenes."""
    problems: list[str] = []
    gt_path = scene_dir / "gt.json"
    if not gt_path.is_file():
        return [f"{scene_dir}: missing gt.json"]
    gt = json.loads(gt_path.read_text())
    if not isinstance(gt.get("count"), int) or gt["count"] < 0:
        problems.append("count must be an int >= 0")
    if gt.get("tier") not in TIERS:
        problems.append(f"tier must be one of {TIERS}")
    imgs = sorted(p for p in (scene_dir / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        problems.append("images/ is empty")
    if gt.get("n_images") != len(imgs):
        problems.append(f"n_images {gt.get('n_images')} != {len(imgs)} files")
    for p in imgs:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if im.getexif():
                    problems.append(f"{p.name}: EXIF present")
        except (UnidentifiedImageError, OSError):
            problems.append(f"{p.name}: unreadable")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder of photos from the phone")
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/scenes"))
    ap.add_argument("--scene-list", type=Path, help="scene list YAML to take defaults from")
    ap.add_argument("--prompt")
    ap.add_argument("--counts", type=int, nargs="+", help="hand counts: two, or three")
    ap.add_argument("--tier", choices=TIERS)
    ap.add_argument("--zero", action="store_true", help="zero-target scene (count must be 0)")
    ap.add_argument("--physical-scene", help="id of the physical place (defaults to scene id)")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--max-side", type=int, default=2048, help="downscale long side; 0 = keep")
    ap.add_argument("--force", action="store_true", help="replace an existing scene folder")
    a = ap.parse_args(argv)

    try:
        e = load_scene_entry(a.scene_list, a.scene_id)
        counts = a.counts or e.get("hand_counts")
        if not counts:
            raise IngestError("--counts is required (hand counts are made from the photos)")
        scene = ingest(
            a.src,
            a.out,
            a.scene_id,
            prompt=a.prompt or e.get("prompt", ""),
            counts=list(counts),
            tier=a.tier or e.get("tier", ""),
            zero=a.zero or bool(e.get("zero_target", False)),
            physical_scene=a.physical_scene or e.get("physical_scene", a.scene_id),
            notes=a.notes if a.notes is not None else e.get("notes", ""),
            max_side=a.max_side or None,
            force=a.force,
        )
        problems = validate_scene(scene)
    except IngestError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if problems:
        print("invalid scene:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"wrote {scene} ({json.loads((scene / 'gt.json').read_text())['n_images']} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
