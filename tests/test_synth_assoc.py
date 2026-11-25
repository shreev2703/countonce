"""Synthetic association bench and ablation runner: smoke tests on tiny episodes in tmp."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from bench.runner import read_rows
from bench.synth_ablation import AblationConfig, run_ablation, variants
from bench.synth_assoc import SynthAssocConfig, episode_to_lifted, run_synth_assoc
from countonce.cli import main
from countonce.match.model import episode_inputs, make_model, mask_inputs, save_checkpoint
from countonce.merge.v3 import build_inputs
from synth.geometry import GeometryConfig, generate_episode, write_dataset

TINY = GeometryConfig(n_views=(3, 4), n_instances=(4, 8))


@pytest.fixture
def splits(tmp_path: Path) -> Path:
    for name, n, seed0 in (("train", 4, 0), ("val", 2, 100), ("test", 3, 200)):
        write_dataset(tmp_path / name, n, seed0=seed0, cfg=TINY)
    return tmp_path


def test_lifted_inputs_match_trainer_inputs() -> None:
    ep = generate_episode(5, TINY)
    lifted, recon = episode_to_lifted(ep)
    got = build_inputs(lifted, recon, None)
    want = episode_inputs(ep)
    for name in ("xyz", "size", "box", "ray", "cam"):
        np.testing.assert_allclose(getattr(got, name), getattr(want, name), atol=1e-6, err_msg=name)
    np.testing.assert_array_equal(got.view, want.view)


def test_mask_inputs_groups() -> None:
    d = episode_inputs(generate_episode(1, TINY))
    only3d = mask_inputs(d, "3d")
    assert not only3d.ray.any() and not only3d.cam.any() and not only3d.box.any()
    rays = mask_inputs(d, "3d_rays")
    np.testing.assert_array_equal(rays.ray, d.ray)
    assert not rays.score.any()
    full = mask_inputs(d, "full", appearance_dim=8)
    assert full.appearance is not None and not full.appearance.any()
    with pytest.raises(ValueError):
        mask_inputs(d, "bogus")


def test_synth_assoc_rows_and_v3_skip(splits: Path) -> None:
    csv_path = splits / "bench.csv"
    cfg = SynthAssocConfig(test_dir=str(splits / "test"), csv=str(csv_path))
    ran, skipped, errored = run_synth_assoc(cfg)
    assert (ran, skipped, errored) == (9, 3, 0)  # v3 skipped without a checkpoint
    rows = read_rows(csv_path)
    ok = [r for r in rows if r["status"] == "ok"]
    assert {r["method"] for r in ok} == {"v1", "v2", "sync"}
    for r in ok:
        assert r["domain"] == "synthetic"
        assert 0.0 <= float(r["assoc_f1"]) <= 1.0
        assert r["duplicate_rate"] != "" and r["miss_rate"] != "" and r["count_error"] != ""
    assert {r["status"] for r in rows if r["method"] == "v3"} == {"skipped_no_checkpoint"}
    assert run_synth_assoc(cfg)[0] == 0  # resumable


def test_synth_assoc_v3_cli(splits: Path) -> None:
    pytest.importorskip("torch")
    import torch

    torch.manual_seed(0)
    ckpt = splits / "m.pt"
    save_checkpoint(make_model(width=16, layers=2, heads=2), ckpt)
    cfgp = splits / "sa.yaml"
    cfgp.write_text(
        f"test_dir: {splits / 'test'}\ncsv: {splits / 'out.csv'}\nmethods: [v3]\n"
        f"v3_checkpoint: {ckpt}\nv3_synchronize: true\n"
    )
    assert main(["bench", "--synth-assoc", str(cfgp)]) == 0
    rows = read_rows(splits / "out.csv")
    assert len(rows) == 3 and {r["status"] for r in rows} == {"ok"}
    assert all(r["domain"] == "synthetic" for r in rows)


def test_ablation_needs_three_seeds() -> None:
    with pytest.raises(ValueError, match="3 distinct seeds"):
        AblationConfig(seeds=[0, 1]).validate()


def test_ablation_smoke(splits: Path) -> None:
    pytest.importorskip("torch")
    cfg = AblationConfig(
        train_dir=str(splits / "train"),
        val_dir=str(splits / "val"),
        test_dir=str(splits / "test"),
        work_dir=str(splits / "work"),
        out_csv=str(splits / "summary.csv"),
        per_seed_csv=str(splits / "per_seed.csv"),
        seeds=[0, 1, 2],
        epochs=1,
        batch_size=2,
        device="cpu",
        model={"width": 16, "layers": 2, "heads": 2},
        n_boot=50,
    )
    assert [v.name for v in variants(cfg)] == [
        "arm:pairwise", "arm:cycle", "arm:sync", "arm:both",
        "group:3d", "group:3d_rays", "group:full",
    ]  # fmt: skip
    summary, per_seed = run_ablation(cfg)
    rows = list(csv.DictReader(io.StringIO(summary.read_text())))
    assert len(rows) == 7
    for r in rows:
        assert int(r["n_seeds"]) == 3
        for m in ("assoc_f1", "duplicate_rate", "miss_rate", "count_error"):
            assert f"{m}_mean" in r and f"{m}_std" in r and f"{m}_ci_low" in r
        lo, mean, hi = (float(r[f"assoc_f1_{k}"]) for k in ("ci_low", "mean", "ci_high"))
        assert lo <= hi
        assert 0.0 <= mean <= 1.0
    assert len(list(csv.DictReader(io.StringIO(per_seed.read_text())))) == 7 * 3
    # arm:pairwise and group:full share one trained model and inference mode
    by = {r["variant"]: r for r in rows}
    assert by["arm:pairwise"]["assoc_f1_mean"] == by["group:full"]["assoc_f1_mean"]
