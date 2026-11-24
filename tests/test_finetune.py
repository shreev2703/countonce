"""Detector adaptation harness: LoRA, head-only, training, registry and report (no real weights)."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from countonce.detect import finetune as ft  # noqa: E402

S = 16


class TinyDetector(nn.Module):
    """Per-pixel MLP: (B,3,H,W) -> density (B,1,H,W)."""

    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.ModuleDict({"q_proj": nn.Linear(3, 12), "v_proj": nn.Linear(12, 12)})
        self.head = nn.Linear(12, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 3, 1)
        h = torch.relu(self.fusion["q_proj"](h))
        h = torch.relu(self.fusion["v_proj"](h))
        return torch.nn.functional.softplus(self.head(h)).permute(0, 3, 1, 2)


def blobs(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    rng = np.random.default_rng(seed)
    xs, ds, cs = [], [], []
    for _ in range(n):
        k = int(rng.integers(1, 4))
        pts = rng.uniform(3, S - 3, size=(k, 2))
        img = np.zeros((3, S, S), dtype=np.float32)
        yy, xx = np.mgrid[0:S, 0:S]
        for x, y in pts:
            img[0] += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4).astype(np.float32)
        xs.append(torch.from_numpy(img))
        ds.append(torch.from_numpy(ft.points_to_density(pts, S, S, 1.5))[None])
        cs.append(k)
    return torch.stack(xs), torch.stack(ds), cs


def cfg(**kw: object) -> ft.FinetuneConfig:
    base: dict[str, object] = {
        "target_layers": ["fusion.*"],
        "rank": 4,
        "lr": 0.02,
        "epochs": 40,
        "batch_size": 4,
    }
    base.update(kw)
    return ft.FinetuneConfig.from_dict(base)


def test_only_adapters_train_and_base_unchanged() -> None:
    torch.manual_seed(0)
    m = TinyDetector()
    before = copy.deepcopy(m.state_dict())
    x, d, _ = blobs(8)
    c = cfg(epochs=5)
    names = ft.prepare(m, c)
    assert names == ["fusion.q_proj", "fusion.v_proj"]
    res = ft.train(m, x, d, c)
    assert all(".lora_" in n for n in res.trainable)
    assert res.n_trainable_params < res.n_total_params
    for k, v in before.items():  # base weights (incl. the head) are untouched
        key = k.replace("fusion.q_proj.", "fusion.q_proj.base.").replace(
            "fusion.v_proj.", "fusion.v_proj.base."
        )
        assert torch.equal(m.state_dict()[key], v), k
    assert any(float(p.abs().sum()) > 0 for p in (mod.lora_B for mod in ft.lora_modules(m)))


def test_merge_matches_unmerged_and_unmerge_restores() -> None:
    torch.manual_seed(1)
    m = TinyDetector()
    x, d, _ = blobs(8)
    c = cfg(epochs=5)
    ft.prepare(m, c)
    ft.train(m, x, d, c)
    w0 = m.fusion["q_proj"].base.weight.clone()
    with torch.no_grad():
        ref = m(x)
        ft.merge_lora(m)
        assert not torch.equal(m.fusion["q_proj"].base.weight, w0)
        assert torch.allclose(m(x), ref, atol=1e-5)
        ft.unmerge_lora(m)
        assert torch.allclose(m.fusion["q_proj"].base.weight, w0, atol=1e-6)
        assert torch.allclose(m(x), ref, atol=1e-5)


def test_head_only_mode_trains_only_head() -> None:
    torch.manual_seed(2)
    m = TinyDetector()
    before = copy.deepcopy(m.state_dict())
    x, d, _ = blobs(8)
    c = cfg(mode="head", head_layers=["head"], epochs=5)
    ft.prepare(m, c)
    res = ft.train(m, x, d, c)
    assert set(res.trainable) == {"head.weight", "head.bias"}
    assert torch.equal(m.fusion["q_proj"].weight, before["fusion.q_proj.weight"])
    assert not torch.equal(m.head.weight, before["head.weight"])


def test_loss_falls_on_overfit_set() -> None:
    torch.manual_seed(3)
    m = TinyDetector()
    x, d, _ = blobs(4)
    c = cfg(epochs=80, lr=0.03, batch_size=4)
    ft.prepare(m, c)
    res = ft.train(m, x, d, c)
    assert res.losses[-1] < 0.5 * res.losses[0]


def test_bad_targets_raise() -> None:
    with pytest.raises(ValueError, match="no nn.Linear"):
        ft.inject_lora(TinyDetector(), ["nope.*"])
    with pytest.raises(ValueError, match="mode"):
        ft.FinetuneConfig.from_dict({"mode": "full"})
    with pytest.raises(ValueError, match="unknown"):
        ft.FinetuneConfig.from_dict({"bogus": 1})


def test_run_adaptation_registers_adapter_and_writes_rows(tmp_path: Path) -> None:
    torch.manual_seed(4)
    m = TinyDetector()
    x, d, cs = blobs(8)
    tx, _, tc = blobs(4, seed=9)
    c = cfg(
        epochs=10,
        registry_root=str(tmp_path / "reg"),
        out_dir=str(tmp_path / "out"),
        report_csv=str(tmp_path / "rep.csv"),
    )
    base = ft.base_checkpoint_hash(m)
    out = ft.run_adaptation(
        m,
        c,
        (x, d),
        {"seen": ("minneapple", x, cs), "unseen": ("fuji_sfm", tx, tc)},
        dataset_version="v-test",
        e2e={"unseen": ("fuji_sfm", lambda _m: [(3.0, 2.0), (5.0, 6.0), (4.0, 4.0)])},
    )
    e = out["registry_entry"]
    assert e["dataset_version"] == "v-test"
    assert e["config"]["base_checkpoint_sha256"] == base
    assert e["version"] == "v001"
    rows = Path(c.report_csv).read_text().splitlines()
    assert len(rows) == 1 + 2 * (2 + 1)  # zero-shot + fine-tuned x (2 per-image + 1 e2e)

    spec = importlib.util.spec_from_file_location(
        "rep", Path(__file__).parent.parent / "scripts/make_detector_adaptation_report.py"
    )
    assert spec and spec.loader
    rep = importlib.util.module_from_spec(spec)
    sys.modules["rep"] = rep
    spec.loader.exec_module(rep)
    text = rep.render(rep.latest_run(Path(c.report_csv)))
    assert "v001" in text
    assert "(n=3)" in text


def test_report_without_results(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "rep2", Path(__file__).parent.parent / "scripts/make_detector_adaptation_report.py"
    )
    assert spec and spec.loader
    rep = importlib.util.module_from_spec(spec)
    sys.modules["rep2"] = rep
    spec.loader.exec_module(rep)
    out = tmp_path / "r.md"
    assert rep.main(["--csv", str(tmp_path / "none.csv"), "--out", str(out)]) == 0
    text = out.read_text()
    assert text.count("TODO: fill from results") >= 8
    assert "seen" in text and "unseen" in text


def test_module_detector_is_detector(tmp_path: Path) -> None:
    from PIL import Image

    from countonce.detect.base import Detector
    from countonce.types import Frame

    p = tmp_path / "a.png"
    Image.new("RGB", (32, 32)).save(p)
    det = ft.ModuleDetector(TinyDetector(), size=S)
    assert isinstance(det, Detector)
    f = Frame(id=0, path=p, width=32, height=32, sha256="0" * 64)
    assert isinstance(det.detect([f], "apple"), list)


def test_cli_requires_module(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from countonce.cli import main

    assert main(["finetune-detector", "--config", "configs/finetune.yaml"]) == 2
    assert "gated" in capsys.readouterr().err
