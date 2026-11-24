"""Annotated per-frame overlays: new = green, duplicate = blue, rejected = gray."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from countonce.types import CountResult, Detection, Frame

STATUS_COLORS = {
    "new": (46, 160, 67),
    "duplicate": (47, 111, 224),
    "rejected": (140, 140, 140),
}


def render_overlays(
    frames: Sequence[Frame],
    detections: Sequence[Detection],
    result: CountResult,
    out_dir: Path,
) -> list[Path]:
    """Draw every detection on its frame, labelled with its instance ID.

    Pillow is imported lazily so the rest of ``export`` has no image dependency.
    """
    from PIL import Image, ImageDraw

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_to_instance = {m: inst.id for inst in result.instances for m in inst.members}
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame_id, []).append(d)

    written: list[Path] = []
    for frame in frames:
        # Draw on the original image in original pixels when the frame carries a mapping
        # (PLAN 8.1); otherwise fall back to the frame's own image, scaled to its size.
        pm = frame.pixel_map
        if pm is not None and frame.original_path is not None:
            img = Image.open(frame.original_path).convert("RGB")
            sx, sy, ox, oy = pm.sx, pm.sy, pm.ox, pm.oy
        else:
            img = Image.open(frame.path).convert("RGB")
            sx, sy, ox, oy = img.width / frame.width, img.height / frame.height, 0.0, 0.0
        draw = ImageDraw.Draw(img)
        for d in by_frame.get(frame.id, []):
            status = result.detection_status.get(d.id, "rejected")
            color = STATUS_COLORS.get(status, STATUS_COLORS["rejected"])
            label = f"#{det_to_instance[d.id]}" if d.id in det_to_instance else status[:3]
            if d.box_xyxy is not None:
                x0, y0, x1, y1 = (float(v) for v in d.box_xyxy)
                x0, x1, y0, y1 = x0 * sx + ox, x1 * sx + ox, y0 * sy + oy, y1 * sy + oy
                draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
                anchor = (x0 + 2, max(0.0, y0 - 12))
            else:
                assert d.point_xy is not None or d.mask is not None
                if d.point_xy is not None:
                    px = float(d.point_xy[0]) * sx + ox
                    py = float(d.point_xy[1]) * sy + oy
                else:
                    ys, xs = d.mask.nonzero()  # type: ignore[union-attr]
                    px = float(xs.mean()) * sx + ox
                    py = float(ys.mean()) * sy + oy
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=color, width=3)
                anchor = (px + 7, py - 6)
            draw.text(anchor, label, fill=color)
        dest = out_dir / f"frame_{frame.id:04d}.png"
        img.save(dest)
        written.append(dest)
    return written
