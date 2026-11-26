"""One-switch Fuji ablations (PLAN §9.6): overrides and the stale-protocol guard."""

import pytest

from bench import fuji_core as fc
from countonce.config import Config


@pytest.mark.parametrize("name", sorted(fc.ABLATIONS))
def test_each_ablation_changes_exactly_one_config_field(name: str) -> None:
    base = Config(assoc="v2").to_dict()
    flipped = Config(assoc="v2", **fc.ABLATIONS[name]).to_dict()
    assert [k for k in base if base[k] != flipped[k]] == list(fc.ABLATIONS[name])
    assert fc.ablation_method(name) == f"countonce_v2__{name}"


def test_stale_protocol_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fc, "PROTOCOL", "fuji_core_v2")
    with pytest.raises(ValueError, match="predate the alignment fix"):
        fc.check_ablation_request("sam_masks", "test", [10], "vggt")


@pytest.mark.parametrize(
    ("split", "ks", "backbone"),
    [("val", [10], "vggt"), ("test", [5, 10], "vggt"), ("test", [10], "colmap")],
)
def test_only_test_k10_vggt(
    monkeypatch: pytest.MonkeyPatch, split: str, ks: list[int], backbone: str
) -> None:
    monkeypatch.setattr(fc, "PROTOCOL", "next")
    fc.check_ablation_request("sam_masks", "test", [10], "vggt")
    with pytest.raises(ValueError, match="--ablation needs"):
        fc.check_ablation_request("sam_masks", split, ks, backbone)


def test_unknown_ablation() -> None:
    with pytest.raises(ValueError, match="unknown ablation"):
        fc.ablation_method("nope")
