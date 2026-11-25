"""Matcher training, cycle loss, calibration (PLAN §10.3-10.6). CPU, tiny episodes."""

import csv
import io
import json

import numpy as np
import pytest

from countonce.match.calibration import (
    apply_temperature,
    ece,
    fit_temperature,
    reliability_data,
)
from synth.geometry import GeometryConfig, write_dataset

torch = pytest.importorskip("torch")

from countonce.match.model import load_checkpoint  # noqa: E402
from countonce.match.train import TrainConfig, cycle_consistency_loss, train  # noqa: E402

TINY = GeometryConfig(n_views=(3, 3), n_instances=(4, 6))


def _perm(n, seed):
    return torch.eye(n)[torch.as_tensor(np.random.default_rng(seed).permutation(n))]


def test_cycle_loss_zero_for_consistent_positive_for_inconsistent():
    a, b = _perm(5, 0), _perm(5, 1)
    assert float(cycle_consistency_loss(a, b, a @ b)) == 0.0
    wrong = _perm(5, 2)
    assert not torch.equal(wrong, a @ b)
    assert float(cycle_consistency_loss(a, b, wrong)) > 0.0


def test_cycle_loss_ignores_dustbin_mass():
    # a detection that is dustbinned everywhere has an all-zero core row/col: still consistent
    a = torch.zeros(3, 3)
    a[0, 0] = a[1, 1] = 1.0
    assert float(cycle_consistency_loss(a, a, a @ a)) == 0.0


def test_temperature_scaling_lowers_ece_on_overconfident_toy():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.2, 0.8, 4000)
    y = rng.random(4000) < true_p
    over = np.clip(np.where(true_p > 0.5, true_p + 0.4, true_p - 0.4), 0.01, 0.99)
    val, test = slice(0, 2000), slice(2000, None)
    t = fit_temperature(over[val], y[val])
    assert t > 1.0
    before = ece(over[test], y[test])
    after = ece(apply_temperature(over[test], t), y[test])
    assert after < before
    rel = reliability_data(over[test], y[test])
    assert rel.count.sum() == 2000 and len(rel.to_dict()["edges"]) == 11


def _cfg(tmp_path, **kw):
    write_dataset(tmp_path / "train", 6, seed0=0, cfg=TINY)
    write_dataset(tmp_path / "val", 3, seed0=100, cfg=TINY)
    base = dict(
        train_dir=str(tmp_path / "train"),
        val_dir=str(tmp_path / "val"),
        out_dir=str(tmp_path / "run"),
        results_csv=str(tmp_path / "results" / "r.csv"),
        epochs=2,
        batch_size=2,
        lr=3e-3,
        cycle_weight=0.5,
        model={"width": 32, "layers": 2, "heads": 2},
    )
    base.update(kw)
    return TrainConfig(**base)


def test_two_epoch_cpu_run_lowers_loss_and_logs(tmp_path):
    hist = train(_cfg(tmp_path, device="cpu"))
    assert [h["epoch"] for h in hist] == [1, 2]
    assert hist[1]["train_nll"] < hist[0]["train_nll"]
    assert np.isfinite(hist[1]["train_cycle"])
    lines = [json.loads(x) for x in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert lines[0]["event"] == "run" and "data_version" in lines[0] and "git_commit" in lines[0]
    assert [x["event"] for x in lines[1:]] == ["epoch", "epoch"]
    rows = list(csv.DictReader(io.StringIO((tmp_path / "results" / "r.csv").read_text())))
    assert len(rows) == 2
    assert (tmp_path / "run" / "calibration.json").exists()
    assert load_checkpoint(tmp_path / "run" / "best.pt") is not None


def test_resume_reproduces_uninterrupted_metrics(tmp_path):
    full = train(_cfg(tmp_path / "a", device="cpu"))
    part = train(_cfg(tmp_path / "b", device="cpu", stop_after_epoch=1))
    assert len(part) == 1
    resumed = train(_cfg(tmp_path / "b", device="cpu"))
    strip = lambda h: [{k: v for k, v in r.items() if k != "wall_s"} for r in h]  # noqa: E731
    assert strip(resumed) == strip(full)  # wall_s is real time, never reproducible


def _stop_cfg(tmp_path, **kw):
    write_dataset(tmp_path / "tr", 6, 0, TINY)
    write_dataset(tmp_path / "va", 3, 100, TINY)
    return TrainConfig(
        train_dir=str(tmp_path / "tr"),
        val_dir=str(tmp_path / "va"),
        out_dir=str(tmp_path / "run"),
        results_csv=str(tmp_path / "curves.csv"),
        epochs=6,
        batch_size=3,
        device="cpu",
        model={"width": 16, "layers": 1, "heads": 2},
        **kw,
    )


def test_wall_clock_cap_stops_after_one_epoch_and_still_writes_curves(tmp_path):
    cfg = _stop_cfg(tmp_path, max_wall_minutes=1e-9)
    history = train(cfg)
    assert len(history) == 1 and history[0]["wall_s"] > 0
    assert (tmp_path / "curves.csv").exists() and (tmp_path / "run" / "best.pt").exists()
    events = [json.loads(x) for x in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert events[-1] == {"event": "stop", "reason": "wall_clock_cap", "epoch": 1}


def test_early_stop_patience_stops_when_val_f1_stalls(tmp_path):
    cfg = _stop_cfg(tmp_path, early_stop_patience=1, lr=0.0)  # lr 0: val F1 never improves
    history = train(cfg)
    assert len(history) == 2  # epoch 1 sets the best, epoch 2 does not beat it
