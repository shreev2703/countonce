"""Capped GPU matcher training smoke: 20 synthetic episodes, 1 epoch (PLAN §10.4). Needs CUDA."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")

from countonce.match.model import load_checkpoint  # noqa: E402
from countonce.match.train import TrainConfig, train  # noqa: E402
from synth.geometry import GeometryConfig, write_dataset  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_one_epoch_on_20_episodes_runs_on_cuda(tmp_path: Path) -> None:
    geo = GeometryConfig(n_views=(3, 5), n_instances=(5, 30))
    write_dataset(tmp_path / "train", 16, 0, geo)
    write_dataset(tmp_path / "val", 4, 1000, geo)
    cfg = TrainConfig(
        train_dir=str(tmp_path / "train"),
        val_dir=str(tmp_path / "val"),
        out_dir=str(tmp_path / "run"),
        results_csv=str(tmp_path / "curves.csv"),
        epochs=1,
        batch_size=4,
        device="cuda",
        model={"width": 32, "layers": 2, "heads": 4},
        max_wall_minutes=5,
    )
    history = train(cfg)
    assert len(history) == 1
    assert 0.0 <= history[0]["val_f1"] <= 1.0
    assert history[0]["wall_s"] > 0
    model = load_checkpoint(tmp_path / "run" / "best.pt", "cuda")
    assert next(model.parameters()).is_cuda
