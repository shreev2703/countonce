"""Privacy-mask producer: blur faces/plates and record where (PLAN 14.8, 8.4 step 2).

``lift.py`` already drops ``Frame.privacy_mask`` pixels and fails detections that are mostly
inside it with ``privacy_mask``; this module is what produces that mask. Region detection sits
behind a small protocol so tests use a fake and a real model is plugged in lazily. Nothing is
downloaded: the OpenCV adapter loads only from an explicit path given in ``Config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from countonce.types import GridMap


@runtime_checkable
class PrivacyRegionDetector(Protocol):
    """Finds regions to blur. Boxes are ``(x0, y0, x1, y1)`` pixels in the given image."""

    def regions(self, image: np.ndarray) -> list[tuple[float, float, float, float]]: ...


def regions_to_mask(
    boxes: list[tuple[float, float, float, float]], height: int, width: int
) -> np.ndarray:
    """HxW bool mask, True inside any box (clipped to the image)."""
    mask = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in boxes:
        xa, xb = max(0, int(np.floor(x0))), min(width, int(np.ceil(x1)))
        ya, yb = max(0, int(np.floor(y0))), min(height, int(np.ceil(y1)))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True
    return mask


def blur_regions(
    image: np.ndarray,
    detector: PrivacyRegionDetector,
    sigma: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-blur the detected regions of an HxWx3 uint8 image.

    Returns ``(blurred_image, mask)``; the mask is HxW bool and True where pixels were replaced.
    The input is not modified. The blur is applied inside the mask only.
    """
    from scipy.ndimage import gaussian_filter

    h, w = image.shape[:2]
    mask = regions_to_mask(detector.regions(image), h, w)
    out = image.copy()
    if mask.any():
        blurred = gaussian_filter(image.astype(np.float32), sigma=(sigma, sigma, 0))
        out[mask] = np.clip(blurred[mask], 0, 255).astype(image.dtype)
    return out, mask


def blur_image_file(
    path: Path, out_dir: Path, detector: PrivacyRegionDetector, sigma: float = 12.0
) -> tuple[Path, np.ndarray]:
    """Blur one image file into ``out_dir`` (PNG, same stem); returns the new path and mask."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    blurred, mask = blur_regions(arr, detector, sigma)
    dest = out_dir / f"{Path(path).stem}.png"
    Image.fromarray(blurred).save(dest)
    return dest, mask


def mask_to_grid(mask: np.ndarray, grid_map: GridMap) -> np.ndarray:
    """Resample a frame-pixel privacy mask onto the backbone grid (nearest, same convention as
    ``lift._source_index``), so it lines up with recon pixels. Cropped/padded cells are False."""
    h, w = mask.shape
    gh, gw = grid_map.grid_height, grid_map.grid_width
    src_y = np.floor((np.arange(gh) + 0.5 - grid_map.oy) / grid_map.sy).astype(int)
    src_x = np.floor((np.arange(gw) + 0.5 - grid_map.ox) / grid_map.sx).astype(int)
    vy, vx = (src_y >= 0) & (src_y < h), (src_x >= 0) & (src_x < w)
    out = mask[np.ix_(np.clip(src_y, 0, h - 1), np.clip(src_x, 0, w - 1))]
    return np.asarray(out & vy[:, None] & vx[None, :])


class OpenCVRegionDetector:
    """Face/plate regions from an explicit OpenCV model file, loaded on first use.

    ``.xml`` is a Haar/LBP cascade (``cv2.CascadeClassifier``); ``.onnx`` is a YuNet-style face
    model (``cv2.FaceDetectorYN``). The installed ``cv2`` ships no model files and this class
    never downloads one: a missing path raises ``FileNotFoundError``.
    """

    def __init__(self, model_path: str | Path, score_threshold: float = 0.6) -> None:
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            import cv2

            if not self.model_path.is_file():
                raise FileNotFoundError(f"privacy model not found: {self.model_path}")
            if self.model_path.suffix.lower() == ".onnx":
                self._model = cv2.FaceDetectorYN.create(
                    str(self.model_path), "", (320, 320), self.score_threshold
                )
            else:
                # cv2 5.x stubs omit CascadeClassifier, though it still exists at runtime.
                clf = cv2.CascadeClassifier(str(self.model_path))  # type: ignore[attr-defined]
                if clf.empty():
                    raise ValueError(f"could not load cascade: {self.model_path}")
                self._model = clf
        return self._model

    def regions(self, image: np.ndarray) -> list[tuple[float, float, float, float]]:
        import cv2

        model = self._load()
        h, w = image.shape[:2]
        bgr = np.ascontiguousarray(image[..., ::-1])
        if self.model_path.suffix.lower() == ".onnx":
            model.setInputSize((w, h))  # type: ignore[attr-defined]
            _, faces = model.detect(bgr)  # type: ignore[attr-defined]
            rows = [] if faces is None else faces[:, :4]
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rows = model.detectMultiScale(gray)  # type: ignore[attr-defined]
        return [(float(x), float(y), float(x + bw), float(y + bh)) for x, y, bw, bh in rows]


def build_privacy_detector(model_path: str | Path | None) -> PrivacyRegionDetector:
    """Detector for ``Config.privacy_model_path``; refuses to run blur without a model."""
    if model_path is None:
        raise ValueError("privacy_blur is on but privacy_model_path is not set")
    return OpenCVRegionDetector(model_path)
