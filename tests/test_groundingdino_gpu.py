"""Real GroundingDINO-tiny (HF) on two Fuji-SfM images. Needs CUDA, transformers, cached weights."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
if importlib.util.find_spec("transformers") is None:
    pytest.skip("transformers is not installed", allow_module_level=True)

DATA = Path(__file__).resolve().parents[1] / "data" / "fuji_sfm"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_hf_groundingdino_box_format_on_fuji_images() -> None:
    from PIL import Image

    from bench.datasets.fuji_sfm import load_fuji_mask_set
    from countonce.detect.groundingdino import GroundingDinoDetector
    from countonce.types import Frame

    try:
        imgs = load_fuji_mask_set(DATA, "val", limit=2)
    except FileNotFoundError:
        pytest.skip("Fuji-SfM data not downloaded")
    if len(imgs) < 2:
        pytest.skip("Fuji-SfM mask set incomplete")
    det = GroundingDinoDetector()
    frames = []
    for i, im in enumerate(imgs):
        with Image.open(im.path) as pil:
            w, h = pil.size
        frames.append(
            Frame(
                id=i,
                path=im.path,
                width=w,
                height=h,
                sha256=hashlib.sha256(im.path.read_bytes()).hexdigest(),
            )
        )
    dets = det.detect(frames, "apple")
    assert dets, "no boxes above the raw floor on apple images"
    assert det.version.startswith("hf:IDEA-Research/grounding-dino-tiny@")
    by_id = {f.id: f for f in frames}
    for d in dets:
        assert d.box_xyxy is not None
        x0, y0, x1, y1 = (float(v) for v in d.box_xyxy)
        f = by_id[d.frame_id]
        assert 0 <= x0 < x1 <= f.width + 1e-3 and 0 <= y0 < y1 <= f.height + 1e-3
        assert det.floor <= d.score <= 1.0
    assert max(d.score for d in dets) > 0.3
    assert np.isfinite([d.score for d in dets]).all()
