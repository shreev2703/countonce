"""Real VGGT-1B on the bundled tiny scene. Needs CUDA, the vggt package and cached weights."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
if importlib.util.find_spec("vggt") is None:
    pytest.skip("gpu test: vggt is not installed", allow_module_level=True)

SCENE = Path(__file__).resolve().parents[1] / "examples" / "tiny_scene"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_vggt_color_blob_matches_fixture_count(tmp_path: Path) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from countonce.config import Config
    from countonce.pipeline import run

    # out_dir defaults to results/ and run() writes sanitized copies there even without export.
    out = str(tmp_path / "out")
    real_cfg = Config(backbone="vggt", detector="color_blob", out_dir=out)
    real = run(SCENE, "red ball", real_cfg, export=False)
    ref = run(
        SCENE,
        "red ball",
        Config(
            backbone="fixture",
            detector="color_blob",
            backbone_options={"scene": str(SCENE / "scene.json")},
            out_dir=out,
        ),
        export=False,
    )
    assert real.count == ref.count == 5
    assert real.peak_vram_mb and real.peak_vram_mb > 0
