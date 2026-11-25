"""M2 regression: real GroundingDINO-tiny + VGGT-1B counts 5 on the bundled tiny scene.

Root cause of the old 7 (docs/DEVIATIONS.md, "Tiny-scene over-count"): the default near-duplicate
filter collapsed the five symmetric views to one, and at the default 0.3 score floor the blue and
green distractor spheres scored 0.44-0.50 for "red sphere" and were counted. Reds score 0.58-0.64.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
if importlib.util.find_spec("vggt") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("vggt/transformers not installed", allow_module_level=True)

SCENE = Path(__file__).resolve().parents[1] / "examples" / "tiny_scene"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_real_path_counts_five_on_tiny_scene(tmp_path: Path) -> None:
    from countonce import pipeline
    from countonce.config import Config

    cfg = Config(
        detector="gdino",
        backbone="vggt",
        assoc="v1",
        score_threshold=0.5,  # above the distractor scores, below the red ones
        nms_iou=0.7,
        out_dir=str(tmp_path / "out"),
        backbone_options={"weights_dtype": "fp32"},
    )
    if os.environ.get("COUNTONCE_OFFLINE"):
        pytest.skip("offline")
    res = pipeline.run(SCENE, "red sphere", cfg, export=False)
    assert res.count == 5
