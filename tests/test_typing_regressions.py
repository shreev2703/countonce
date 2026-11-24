"""Behaviour pinned while paying down mypy debt (docs/DEVIATIONS.md, "CI / mypy").

The three suspected bugs (colmap config path, v2 extractor guard, DINOv2 model handle)
turned out to be typing-only; these tests pin the runtime behaviour the fixes rely on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from countonce.merge.match import V2Associator
from countonce.types import Detection, LiftedDetection


def _ld(det_id: str, frame: int, feature: np.ndarray | None = None) -> LiftedDetection:
    det = Detection(
        id=det_id, frame_id=frame, score=0.9, point_xy=np.array([5.0, 5.0]), feature=feature
    )
    return LiftedDetection(det=det, xyz=np.zeros(3), size=1.0, lift_conf=0.9)


def test_colmap_config_path_handling(tmp_path: Path) -> None:
    pytest.importorskip("pycolmap")
    from countonce.recon.colmap import ColmapConfig

    assert ColmapConfig.from_yaml() is not None  # shipped default (or dataclass defaults)
    with pytest.raises(FileNotFoundError):
        ColmapConfig.from_yaml(tmp_path / "missing.yaml")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}\n")
    assert ColmapConfig.from_yaml(str(cfg)) == ColmapConfig()  # str accepted at runtime


def test_v2_features_without_extractor_uses_only_stored_features() -> None:
    feat = np.array([1.0, 0.0])
    ok = [_ld("0:0", 0, feat), _ld("1:0", 1)]
    feats = V2Associator()._features(ok, None)
    assert set(feats) == {"0:0"}
    np.testing.assert_array_equal(feats["0:0"], feat)


def test_v2_features_fills_missing_from_extractor() -> None:
    class Fixed:
        def extract(self, frame: object, dets: list[Detection]) -> np.ndarray:
            return np.tile(np.array([0.0, 1.0]), (len(dets), 1))

    ok = [_ld("0:0", 0, np.array([1.0, 0.0])), _ld("1:0", 1)]
    feats = V2Associator(extractor=Fixed())._features(ok, None)
    assert set(feats) == {"0:0", "1:0"}
    np.testing.assert_array_equal(feats["1:0"], [0.0, 1.0])


def test_dino_extractor_needs_frame_and_defers_model_load() -> None:
    pytest.importorskip("torch")
    from countonce.merge.appearance import DinoV2Extractor

    ex = DinoV2Extractor()
    assert ex._model is None  # nothing is downloaded at construction
    with pytest.raises(ValueError, match="needs the Frame"):
        ex.extract(None, [_ld("0:0", 0).det])
