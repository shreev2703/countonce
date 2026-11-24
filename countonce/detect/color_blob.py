"""Deterministic HSV-threshold detector (TEST BACKEND ONLY, see docs/DEVIATIONS.md).

Finds connected blobs of one colour named in the prompt ("red sphere" -> red). It exists so the
real ``countonce run`` command can be driven on CPU, with no weights, against the bundled
``examples/tiny_scene``. It is not a general detector. Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

from countonce.detect.base import Exemplar
from countonce.types import Detection, Frame

# colour word -> hue interval(s) in degrees; a tuple of (lo, hi) pairs, wrap-around as two pairs
HUES: dict[str, tuple[tuple[float, float], ...]] = {
    "red": ((0.0, 15.0), (345.0, 360.0)),
    "orange": ((15.0, 45.0),),
    "yellow": ((45.0, 70.0),),
    "green": ((70.0, 165.0),),
    "blue": ((200.0, 260.0),),
    "purple": ((260.0, 310.0),),
}


def _hue_sat_val(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB (uint8 HxWx3) -> hue degrees, saturation, value."""
    x = rgb.astype(np.float64) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(
            mx == r,
            ((g - b) / d) % 6,
            np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4),
        )
        s = np.where(mx > 0, d / mx, 0.0)
    return np.where(d > 0, h * 60.0, 0.0), s, mx


class ColorBlobDetector:
    name = "color_blob"
    version = "1"

    def __init__(self, min_saturation: float = 0.5, min_value: float = 0.15, min_area: int = 12):
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.min_area = min_area

    def config(self) -> dict[str, object]:
        return {
            "min_saturation": self.min_saturation,
            "min_value": self.min_value,
            "min_area": self.min_area,
        }

    @staticmethod
    def colour_of(prompt: str) -> str:
        for word in prompt.lower().replace(",", " ").split():
            if word in HUES:
                return word
        raise ValueError(f"color_blob needs a colour word ({', '.join(HUES)}) in the prompt")

    def detect(
        self,
        frames: Sequence[Frame],
        prompt: str,
        exemplars: Sequence[Exemplar] | None = None,
    ) -> list[Detection]:
        colour = self.colour_of(prompt)
        out: list[Detection] = []
        for f in frames:
            with Image.open(f.path) as im:
                rgb = np.asarray(im.convert("RGB"))
            hue, sat, val = _hue_sat_val(rgb)
            in_hue = np.zeros(hue.shape, dtype=bool)
            for lo, hi in HUES[colour]:
                in_hue |= (hue >= lo) & (hue < hi)
            mask = in_hue & (sat >= self.min_saturation) & (val >= self.min_value)
            labels, n = ndimage.label(mask)  # 4-connectivity, raster-ordered labels
            k = 0
            for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
                if sl is None:
                    continue
                blob = labels[sl] == lab
                area = int(blob.sum())
                if area < self.min_area:
                    continue
                y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
                full = np.zeros(mask.shape, dtype=bool)
                full[sl] = blob
                # score: how much of the box the blob fills, capped; a sphere fills ~pi/4
                fill = area / float((y1 - y0) * (x1 - x0))
                out.append(
                    Detection(
                        id=Detection.make_id(f.id, k),
                        frame_id=f.id,
                        score=float(min(1.0, 0.5 + fill / 2)),
                        box_xyxy=np.array([x0, y0, x1, y1], dtype=np.float64),
                        mask=full,
                        label=colour,
                    )
                )
                k += 1
        return out


__all__ = ["ColorBlobDetector", "HUES"]
