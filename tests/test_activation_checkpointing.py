"""Activation checkpointing in the matcher (PLAN §13 item 11). CPU torch, skipped without it."""

import json

import pytest

torch = pytest.importorskip("torch")

from countonce.match.model import (  # noqa: E402
    checkpoint_meta,
    episode_inputs,
    episode_loss,
    make_model,
)
from countonce.match.train import TrainConfig, profile_memory, train  # noqa: E402
from synth.geometry import GeometryConfig, generate_episode, write_dataset  # noqa: E402

SMALL = GeometryConfig(n_views=(3, 3), n_instances=(4, 6))


def _grads(mode):
    torch.manual_seed(0)
    model = make_model(width=32, layers=4, activation_checkpointing=mode).train()
    ep = generate_episode(3, SMALL)
    loss = episode_loss(model, ep, episode_inputs(ep))
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()}


@pytest.mark.parametrize("mode", ["attention", "all"])
def test_loss_and_grads_match_baseline(mode):
    base_loss, base = _grads("none")
    loss, got = _grads(mode)
    assert abs(loss - base_loss) < 1e-5
    assert base.keys() == got.keys()
    for name in base:
        assert torch.allclose(base[name], got[name], atol=1e-5, rtol=1e-5), name


def _count_calls(monkeypatch):
    calls = []
    real = torch.utils.checkpoint.checkpoint

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", spy)
    return calls


def test_checkpoint_call_counts_and_flag(monkeypatch):
    ep = generate_episode(1, SMALL)
    d = episode_inputs(ep)
    calls = _count_calls(monkeypatch)
    for mode, expected in (("none", 0), ("attention", 4), ("all", 8)):
        calls.clear()
        episode_loss(make_model(width=32, layers=4, activation_checkpointing=mode).train(), ep, d)
        assert len(calls) == expected, mode
        assert all(c["use_reentrant"] is False for c in calls)


def test_eval_and_no_grad_never_checkpoint(monkeypatch):
    ep = generate_episode(1, SMALL)
    d = episode_inputs(ep)
    model = make_model(width=32, layers=4, activation_checkpointing="all")
    calls = _count_calls(monkeypatch)
    model.eval()
    model.match_probs(d)
    episode_loss(model, ep, d)  # eval mode, grad enabled
    model.train()
    with torch.no_grad():
        model.match_probs(d)
    assert calls == []


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="activation_checkpointing"):
        make_model(width=32, layers=2, activation_checkpointing="bogus")
    with pytest.raises(ValueError, match="activation_checkpointing"):
        TrainConfig(activation_checkpointing="bogus").validated()


def test_hparams_unchanged_so_checkpoints_stay_compatible():
    a = make_model(width=32, layers=2).hparams
    b = make_model(width=32, layers=2, activation_checkpointing="all").hparams
    assert a == b


def test_profile_memory_saved_activations_shrink(tmp_path):
    write_dataset(tmp_path / "train", 2, 0, SMALL)
    cfg = TrainConfig(train_dir=str(tmp_path / "train"), device="cpu", model={"width": 32})
    rows = {r["mode"]: r for r in profile_memory(cfg)}
    assert rows["none"]["peak_vram_mb"] is None
    assert rows["all"]["saved_mb"] < rows["attention"]["saved_mb"] < rows["none"]["saved_mb"]


def test_train_records_setting_in_meta_and_log(tmp_path):
    write_dataset(tmp_path / "train", 3, 0, SMALL)
    write_dataset(tmp_path / "val", 2, 100, SMALL)
    cfg = TrainConfig(
        train_dir=str(tmp_path / "train"),
        val_dir=str(tmp_path / "val"),
        out_dir=str(tmp_path / "run"),
        results_csv=str(tmp_path / "r.csv"),
        epochs=1,
        batch_size=2,
        device="cpu",
        model={"width": 32, "layers": 2},
        activation_checkpointing="all",
    )
    train(cfg)
    assert checkpoint_meta(tmp_path / "run" / "best.pt")["activation_checkpointing"] == "all"
    first = json.loads((tmp_path / "run" / "log.jsonl").read_text().splitlines()[0])
    assert first["activation_checkpointing"] == "all"
    assert first["config"]["activation_checkpointing"] == "all"
