"""Retrain + promotion loop on CPU: consent filtering, gate outcomes, log row."""

import csv
import json
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.corrections import Correction
from countonce.match.registry import Registry
from countonce.match.retrain import (
    RetrainConfig,
    RetrainError,
    append_log_row,
    eligible_corrections,
    retrain,
)
from countonce.schemas import load_result
from synth.geometry import GeometryConfig, write_dataset

FIX = Path(__file__).parent / "fixtures" / "schemas" / "v1.1"


@pytest.fixture
def res():
    return load_result(json.loads((FIX / "result.json").read_text()))


def _corrections():
    return [
        Correction("job1", "split", ("2:0", "3:0"), user="alice", consent=True),
        Correction("job1", "reject", ("3:4",), user="alice", consent=True),
        Correction("job1", "split", ("2:1", "3:1"), user="bob", consent=True),  # opted out
        Correction("job1", "split", ("2:2", "3:2"), user="alice", consent=False),  # no consent
        Correction("job1", "split", ("2:3", "3:3"), user=None, consent=True),  # anonymous
    ]


def _frozen(table):
    def run(ckpt):
        mae, f1 = table.get(ckpt.stem, table["default"])
        return {
            "mae": {"estimate": mae, "low": mae - 0.1, "high": mae + 0.1},
            "assoc_f1": {"estimate": f1, "low": f1 - 0.01, "high": f1 + 0.01},
            "frozen_problems": [],
        }

    return run


def _golden(table):
    def run(ckpt):
        m, f1 = table.get(ckpt.stem, table["default"])
        return {"scenes": {}, "overall": {"mape": m, "assoc_f1": f1}}

    return run


def _fake_trainer(payload):
    def run(tcfg):
        out = Path(tcfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "best.pt").write_bytes(payload)
        run.cfg = tcfg
        return []

    return run


@pytest.fixture
def setup(tmp_path):
    reg = Registry(tmp_path / "models")
    (tmp_path / "inc.pt").write_bytes(b"incumbent")
    reg.register(tmp_path / "inc.pt", git_sha="x")
    reg.promote("v001")
    cfg = RetrainConfig(
        registry_root=str(tmp_path / "models"),
        out_dir=str(tmp_path / "rt"),
        log_csv=str(tmp_path / "results" / "retrain_log.csv"),
    )
    return reg, cfg


def _read(cfg):
    return list(csv.DictReader(Path(cfg.log_csv).open()))


def test_eligibility_excludes_opted_out_unconsented_and_anonymous():
    used = eligible_corrections(_corrections(), {"alice"}, {"job1"})
    assert [(c.user, c.detection_ids) for c in used] == [
        ("alice", ("2:0", "3:0")),
        ("alice", ("3:4",)),
    ]
    assert eligible_corrections(_corrections(), set(), {"job1"}) == []
    assert eligible_corrections(_corrections(), {"alice"}, set()) == []


def test_opted_out_corrections_never_reach_the_training_file(setup, res):
    reg, cfg = setup
    trainer = _fake_trainer(b"cand")
    retrain(
        cfg,
        frozen_evals={"val": _frozen({"default": (2.0, 0.9)})},
        trainer=trainer,
        corrections=_corrections(),
        results={"job1": res},
        opted_in_users={"alice"},
    )
    path = Path(trainer.cfg.corrections)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows and {r["user"] for r in rows} == {"alice"}
    assert {r["kind"] for r in rows} <= {"hard_negative", "hard_positive", "false_positive"}
    assert "false_positive" in {r["kind"] for r in rows}  # the reject
    assert trainer.cfg.init_checkpoint.endswith("v001.pt")
    assert _read(cfg)[0]["n_corrections"] == "2"


def test_nobody_opted_in_means_zero_corrections(setup, res):
    _, cfg = setup
    trainer = _fake_trainer(b"cand")
    retrain(
        cfg,
        frozen_evals={"val": _frozen({"default": (2.0, 0.9)})},
        trainer=trainer,
        corrections=_corrections(),
        results={"job1": res},
    )
    assert trainer.cfg.corrections == ""
    assert _read(cfg)[0]["n_corrections"] == "0"


def test_worse_candidate_is_registered_not_promoted_and_logged(setup):
    reg, cfg = setup
    out = retrain(
        cfg,
        frozen_evals={"val": _frozen({"v001": (2.0, 0.9), "default": (3.0, 0.9)})},
        golden_eval=_golden({"default": (0.05, 0.9)}),
        trainer=_fake_trainer(b"worse"),
        corrections=[],
    )
    assert not out.decision.passed and out.candidate == "v002"
    assert reg.get("v002")["version"] == "v002"  # registered
    assert reg.current()["version"] == "v001"  # but not promoted
    assert reg.log()[-1]["passed"] is False
    row = _read(cfg)[0]
    assert row["promoted"] == "false" and "frozen/mae" in row["reasons"]
    assert row["val_before_mae"] == "2.0" and row["val_after_mae"] == "3.0"


def test_golden_regression_blocks_promotion(setup):
    reg, cfg = setup
    out = retrain(
        cfg,
        frozen_evals={"val": _frozen({"default": (2.0, 0.9)})},
        golden_eval=_golden({"v001": (0.05, 0.9), "default": (0.30, 0.9)}),
        trainer=_fake_trainer(b"golden-worse"),
        corrections=[],
    )
    assert not out.decision.passed and reg.current()["version"] == "v001"


def test_non_regressing_candidate_is_promoted_and_row_has_cis(setup, res):
    reg, cfg = setup
    out = retrain(
        cfg,
        frozen_evals={
            "val": _frozen({"v001": (2.0, 0.90), "default": (1.9, 0.91)}),
            "test": _frozen({"v001": (2.2, 0.88), "default": (2.1, 0.89)}),
        },
        golden_eval=_golden({"default": (0.05, 0.9)}),
        trainer=_fake_trainer(b"better"),
        corrections=_corrections(),
        results={"job1": res},
        opted_in_users={"alice"},
    )
    assert out.decision.passed and reg.current()["version"] == "v002"
    row = _read(cfg)[0]
    assert row["promoted"] == "true" and row["n_corrections"] == "2"
    assert row["incumbent"] == "v001" and row["candidate"] == "v002"
    assert (
        float(row["val_after_mae_lo"])
        < float(row["val_after_mae"])
        < float(row["val_after_mae_hi"])
    )
    assert row["test_before_mae"] == "2.2" and row["golden_after_mape"] == "0.05"
    assert row["git_sha"] and "n_false_positive" in row


def test_log_appends_rows_and_extends_columns(tmp_path):
    p = tmp_path / "l.csv"
    append_log_row(p, {"incumbent": "v1", "val_before_mae": 1.0})
    append_log_row(p, {"incumbent": "v2", "golden_after_mape": 0.1})
    rows = list(csv.DictReader(p.open()))
    assert [r["incumbent"] for r in rows] == ["v1", "v2"]
    assert rows[0]["golden_after_mape"] == "" and rows[1]["val_before_mae"] == ""


def test_errors_without_incumbent_or_frozen_eval(tmp_path, setup):
    _, cfg = setup
    with pytest.raises(RetrainError, match="frozen"):
        retrain(cfg, frozen_evals={}, trainer=_fake_trainer(b"x"))
    empty = RetrainConfig(registry_root=str(tmp_path / "none"))
    with pytest.raises(RetrainError, match="no promoted"):
        retrain(empty, frozen_evals={"val": _frozen({"default": (1, 1)})})


def test_real_fine_tune_from_incumbent_with_corrections(tmp_path, res):
    pytest.importorskip("torch")
    from countonce.match.model import make_model, save_checkpoint

    tiny = GeometryConfig(n_views=(3, 3), n_instances=(4, 6))
    write_dataset(tmp_path / "train", 4, seed0=0, cfg=tiny)
    write_dataset(tmp_path / "val", 2, seed0=100, cfg=tiny)
    model = make_model(width=32, layers=2, heads=2)
    save_checkpoint(model, tmp_path / "inc.pt")
    reg = Registry(tmp_path / "models")
    reg.register(tmp_path / "inc.pt", git_sha="x")
    reg.promote("v001")
    cfg = RetrainConfig(
        registry_root=str(tmp_path / "models"),
        out_dir=str(tmp_path / "rt"),
        log_csv=str(tmp_path / "log.csv"),
        train={
            "train_dir": str(tmp_path / "train"),
            "val_dir": str(tmp_path / "val"),
            "results_csv": str(tmp_path / "train.csv"),
            "epochs": 1,
            "batch_size": 2,
            "device": "cpu",
        },
    )
    out = retrain(
        cfg,
        frozen_evals={"val": _frozen({"default": (2.0, 0.9)})},
        corrections=_corrections(),
        results={"job1": res},
        opted_in_users={"alice"},
    )
    assert out.decision.passed and reg.current()["version"] == "v002"
    assert "+corr-" in reg.get("v002")["dataset_version"]
    assert _read(cfg)[0]["n_corrections"] == "2"


def test_cli_missing_frozen_config_exits_2(tmp_path, capsys):
    cfgp = tmp_path / "r.yaml"
    cfgp.write_text("registry_root: nowhere\n")
    assert main(["retrain", "--config", str(cfgp)]) == 2
    assert "frozen" in capsys.readouterr().err
