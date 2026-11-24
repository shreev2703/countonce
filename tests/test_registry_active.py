"""Model registry + promotion gate, active-learning queue, correction-mixed training."""

import json

import pytest

from countonce.active import export_review_queue, find_jobs, rank_jobs
from countonce.cli import main
from countonce.match.registry import GateConfig, Registry, RegistryError


def _ckpt(tmp_path, name, payload):
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def _frozen(table):
    """Fake frozen evaluator keyed by checkpoint stem -> (mae, f1)."""

    def run(ckpt):
        mae, f1 = table[ckpt.stem]
        return {"mae": {"estimate": mae}, "assoc_f1": {"estimate": f1}, "frozen_problems": []}

    return run


def _golden(table):
    def run(ckpt):
        m, f1 = table[ckpt.stem]
        return {
            "git_sha": ckpt.stem,
            "scenes": {"s1": {"mape": m, "assoc_f1": f1, "peak_vram_mb": 0, "latency_s": 1}},
            "overall": {"mape": m, "assoc_f1": f1, "peak_vram_mb": 0, "latency_s": 1},
        }

    return run


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "models")
    r.register(_ckpt(tmp_path, "a.pt", b"aaa"), dataset_version="d1", config={"lr": 1}, git_sha="x")
    r.register(_ckpt(tmp_path, "b.pt", b"bbb"), dataset_version="d2", metrics={"val_f1": 0.9})
    r.register(_ckpt(tmp_path, "c.pt", b"ccc"))
    return r


def test_register_records_hash_and_rejects_duplicates(reg, tmp_path):
    first = reg.get("v001")
    assert len(first["sha256"]) == 64 and first["dataset_version"] == "d1"
    assert first["config"] == {"lr": 1} and first["git_sha"] == "x"
    assert reg.current() is None
    with pytest.raises(RegistryError, match="already registered"):
        reg.register(_ckpt(tmp_path, "dup.pt", b"aaa"))
    with pytest.raises(RegistryError, match="unknown"):
        reg.get("v999")


def test_first_promotion_has_no_comparison_and_is_logged(reg):
    d = reg.promote("v001")
    assert d.passed and reg.current()["version"] == "v001"
    assert reg.log()[0]["incumbent"] is None


def test_regression_is_refused_and_logged_and_current_unchanged(reg):
    reg.promote("v001")
    frozen = _frozen({"v001": (2.0, 0.90), "v002": (2.5, 0.90), "v003": (1.9, 0.905)})
    golden = _golden({"v001": (0.05, 0.9), "v002": (0.05, 0.9), "v003": (0.05, 0.9)})
    bad = reg.promote("v002", frozen_eval=frozen, golden_eval=golden)
    assert not bad.passed and any("frozen/mae" in r for r in bad.reasons)
    assert reg.current()["version"] == "v001"
    ok = reg.promote("v003", frozen_eval=frozen, golden_eval=golden)
    assert ok.passed and reg.current()["version"] == "v003"
    log = reg.log()
    assert [x["passed"] for x in log] == [True, False, True]
    assert log[1]["candidate"] == "v002" and log[1]["reasons"]


def test_golden_regression_refuses_even_when_frozen_is_fine(reg):
    reg.promote("v001")
    frozen = _frozen({"v001": (2.0, 0.9), "v002": (2.0, 0.9)})
    golden = _golden({"v001": (0.05, 0.9), "v002": (0.20, 0.9)})
    d = reg.promote("v002", frozen_eval=frozen, golden_eval=golden)
    assert not d.passed and any(r.startswith("golden:") for r in d.reasons)
    loose = GateConfig(golden={"mape": 0.5, "assoc_f1": 0.5})
    assert reg.promote("v002", gate=loose, frozen_eval=frozen, golden_eval=golden).passed


def test_f1_drop_and_missing_frozen_evaluator_refuse(reg):
    reg.promote("v001")
    frozen = _frozen({"v001": (2.0, 0.90), "v002": (2.0, 0.85)})
    assert not reg.promote("v002", frozen_eval=frozen).passed
    d = reg.promote("v002")  # no evaluator: refuses rather than skipping silently
    assert not d.passed and "frozen" in d.reasons[0]
    assert reg.promote("v002", gate=GateConfig(require_frozen=False)).passed


def test_tampered_checkpoint_is_refused(reg):
    reg.checkpoint_path("v001").write_bytes(b"changed")
    d = reg.promote("v001")
    assert not d.passed and "integrity" in d.reasons[0]
    assert reg.current() is None


def test_cli_list_and_refused_exit_code(reg, tmp_path, capsys):
    root = str(reg.root)
    assert main(["models", "--root", root, "list"]) == 0
    assert "v001" in capsys.readouterr().out
    assert main(["models", "--root", root, "promote", "v001", "--no-golden"]) == 0
    # incumbent exists but no frozen evaluator configured -> refused (exit 1) and logged
    assert main(["models", "--root", root, "promote", "v002", "--no-golden"]) == 1
    assert reg.log()[-1]["passed"] is False
    assert main(["models", "--root", root, "promote", "v404", "--no-golden"]) == 2


# ------------------------------------------------------------------ active learning
def _result(count, low, high, probs):
    return {
        "schema_version": "1.1",
        "prompt": "x",
        "count": count,
        "interval": {"low": low, "high": high, "level": 0.9, "method": "heuristic"},
        "confidence_method": "heuristic",
        "model_versions": {"detector": "d", "backbone": "b", "associator": "v3"},
        "input_hash": "h",
        "match_evidence": [
            {
                "det_a": f"0:{i}",
                "det_b": f"1:{i}",
                "dist_3d_norm": 1.0,
                "reproj_error_px": 1.0,
                "appearance_sim": 0.5,
                "match_prob": p,
                "merged": p > 0.5,
                "reason": "r",
            }
            for i, p in enumerate(probs)
        ],
    }


def _job(root, name, **kw):
    d = root / name
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(_result(**kw)))
    return d


def test_ranking_prefers_uncertain_probs_and_wide_intervals(tmp_path):
    _job(tmp_path, "sure", count=10, low=10, high=10, probs=[0.99, 0.01, 0.98])
    _job(tmp_path, "unsure", count=10, low=9, high=11, probs=[0.5, 0.45, 0.55])
    _job(tmp_path, "wide", count=10, low=4, high=16, probs=[0.99, 0.02])
    done = _job(tmp_path, "reviewed", count=10, low=0, high=40, probs=[0.5])
    (done / "corrected").mkdir()
    items = rank_jobs(find_jobs(tmp_path))
    assert [i.job_id for i in items] == ["unsure", "wide", "sure"]
    assert items[-1].score < items[0].score
    assert "reviewed" in [i.job_id for i in rank_jobs(find_jobs(tmp_path), skip_reviewed=False)]


def test_queue_export_and_cli(tmp_path):
    _job(tmp_path / "jobs", "a", count=5, low=5, high=5, probs=[0.5])
    _job(tmp_path / "jobs", "b", count=5, low=1, high=9, probs=[0.9])
    out = tmp_path / "q.jsonl"
    assert main(["review-queue", str(tmp_path / "jobs"), "--out", str(out), "--top", "1"]) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["rank"] == 1
    export_review_queue(rank_jobs(find_jobs(tmp_path / "jobs")), out)
    assert len(out.read_text().splitlines()) == 2


# ------------------------------------------------------------------ correction mixing
def test_corrections_mixed_into_training(tmp_path):
    pytest.importorskip("torch")
    from countonce.match.correction_data import build_groups, load_pairs
    from countonce.match.train import TrainConfig, train
    from synth.geometry import GeometryConfig, write_dataset

    def rec(i, frame, x):
        return {"id": i, "frame_id": frame, "view": frame, "xyz": [x, 0.0, 0.0], "size": 0.1}

    rows = [
        {"kind": "hard_positive", "job_id": "j", "det_a": "0:0", "det_b": "1:0",
         "geometry": {"0:0": rec("0:0", 0, 0.0), "1:0": rec("1:0", 1, 0.01)}},
        {"kind": "hard_negative", "job_id": "j", "det_a": "0:1", "det_b": "1:1",
         "geometry": {"0:1": rec("0:1", 0, 1.0), "1:1": rec("1:1", 1, 1.4)}},
        {"kind": "false_positive", "job_id": "j", "det_a": "0:2", "det_b": None, "geometry": {}},
    ]  # fmt: skip
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("\n".join(json.dumps(r) for r in rows))
    (g,) = build_groups(load_pairs(pairs))
    assert len(g.positives) == 1 and len(g.negatives) == 1 and g.inputs.n == 4

    tiny = GeometryConfig(n_views=(3, 3), n_instances=(4, 6))
    write_dataset(tmp_path / "train", 4, seed0=0, cfg=tiny)
    write_dataset(tmp_path / "val", 2, seed0=100, cfg=tiny)
    cfg = TrainConfig(
        train_dir=str(tmp_path / "train"),
        val_dir=str(tmp_path / "val"),
        out_dir=str(tmp_path / "run"),
        results_csv=str(tmp_path / "r.csv"),
        epochs=2,
        batch_size=2,
        device="cpu",
        model={"width": 16, "layers": 2, "heads": 2},
        corrections=str(pairs),
    )
    hist = train(cfg)
    assert "train_corr" in hist[0] and hist[0]["train_corr"] > 0
    head = json.loads((tmp_path / "run" / "log.jsonl").read_text().splitlines()[0])
    assert "+corr-" in head["data_version"]


def test_unknown_pair_kind_rejected(tmp_path):
    from countonce.match.correction_data import load_pairs

    p = tmp_path / "p.jsonl"
    p.write_text('{"kind": "bogus"}\n')
    with pytest.raises(ValueError, match="unknown pair kind"):
        load_pairs(p)
