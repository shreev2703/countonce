"""Real SAM 2.1 (HF) on one Fuji-SfM mask-set image: mean IoU vs the GT polygon masks."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
if importlib.util.find_spec("transformers") is None:
    pytest.skip("transformers is not installed", allow_module_level=True)

DATA = Path(__file__).resolve().parents[1] / "data" / "fuji_sfm"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_sam2_hf_mean_iou_on_fuji_mask_image() -> None:
    from PIL import Image, ImageDraw

    from bench.datasets.fuji_sfm import _read_layout, load_fuji_mask_set
    from countonce.detect.masks import Sam2HfMaskProducer

    try:
        (im,) = load_fuji_mask_set(DATA, "val", limit=1)
        ann = _read_layout(DATA).mask_annotations["val"]
    except (FileNotFoundError, ValueError):
        pytest.skip("Fuji-SfM data not downloaded")
    rec = next(r for r in json.loads(ann.read_text()).values() if r["filename"] == im.path.name)
    with Image.open(im.path) as pil:
        rgb = np.asarray(pil.convert("RGB"))
    h, w = rgb.shape[:2]
    gt = []
    for r in rec["regions"].values():
        s = r["shape_attributes"]
        canvas = Image.new("L", (w, h), 0)
        ImageDraw.Draw(canvas).polygon(
            list(zip(s["all_points_x"], s["all_points_y"], strict=True)), fill=1
        )
        gt.append(np.asarray(canvas, bool))

    prod = Sam2HfMaskProducer()
    prod.masks(rgb, im.boxes_xyxy[:1].tolist())  # warm-up: model load
    torch.cuda.synchronize()
    t = time.perf_counter()
    ms = prod.masks(rgb, im.boxes_xyxy.tolist())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    assert len(ms) == len(gt)
    ious = [(m & g).sum() / max((m | g).sum(), 1) for m, g in zip(ms, gt, strict=True)]
    mean_iou = float(np.mean(ious))
    print(
        f"SAM2.1-small {im.path.name}: n={len(ms)} mean IoU={mean_iou:.3f} "
        f"runtime={dt:.2f}s version={prod.version}"
    )
    assert np.isfinite(mean_iou) and mean_iou > 0.5
