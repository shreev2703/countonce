"""Backbone switch of the Fuji harness with a fake reconstructor (no pycolmap, no torch)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bench import fuji_core as fc
from bench import fuji_report as fr
from bench.runner import COLUMNS, read_rows
from countonce.recon.adapter import ReconAdaptError
from countonce.recon.align import transform_extrinsics
from countonce.recon.chunked import ChunkAlignmentError, align_chunk
from countonce.recon.staging import single_root

REPO = Path(__file__).resolve().parents[1]


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("rfc", REPO / "scripts/run_fuji_core.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _line_cameras(n: int) -> np.ndarray:
    ext = np.tile(np.eye(3, 4), (n, 1, 1))
    ext[:, 0, 3] = -np.arange(n) * 0.1
    return ext


def test_method_names_carry_backbone() -> None:
    assert fc.methods_for(["v2"], backbone="colmap") == ("countonce_v2_colmap",)
    assert fc.method_for_assoc("v2") == "countonce_v2"
    assert fc.METHOD_ASSOC["countonce_v2_colmap"] == "v2"
    with pytest.raises(ValueError):
        fc.method_for_assoc("v2", "nerf")


def test_align_window_ignores_unregistered_views() -> None:
    gt = _line_cameras(4)
    rec = gt.copy()
    rec[1] = 0.0
    assert fc.registered_views(rec).tolist() == [True, False, True, True]
    assert fc.align_window(gt, rec).s == pytest.approx(1.0)
    rec[2:] = 0.0
    with pytest.raises(ChunkAlignmentError):
        fc.align_window(gt, rec)


def test_window_alignment_recovers_rotation_for_collinear_cameras() -> None:
    gt = _line_cameras(3)
    roll = np.eye(4)
    roll[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()  # about the path
    rec = np.stack([(np.vstack([e, [0, 0, 0, 1]]) @ roll)[:3] for e in gt])
    np.testing.assert_allclose(align_chunk(gt, rec).R, roll[:3, :3], atol=1e-6)


def test_align_to_reference_fixes_roll_for_collinear_cameras() -> None:
    ref = _line_cameras(3)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()  # about the path
    world[:3, 3] = (1.0, 2.0, 3.0)
    rec = np.stack([(np.vstack([e, [0, 0, 0, 1]]) @ world)[:3] for e in ref])
    rec[1] = 0.0  # unregistered
    sim, resid = fc.align_to_reference(ref, rec)
    got = transform_extrinsics(rec[[0, 2]], sim)
    np.testing.assert_allclose(got, ref[[0, 2]], atol=1e-6)
    assert resid == pytest.approx(0.0, abs=1e-6)
    rec[2] = 0.0
    with pytest.raises(ChunkAlignmentError):
        fc.align_to_reference(ref, rec)


def test_single_root(tmp_path: Path) -> None:
    paths = []
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        paths.append(tmp_path / d / "x.jpg")
        paths[-1].write_bytes(d.encode())
    assert single_root(paths[:1], tmp_path / "s") == paths[:1]
    out = single_root(paths, tmp_path / "s")
    assert [p.name for p in out] == ["00000_x.jpg", "00001_x.jpg"]
    assert [p.read_bytes() for p in out] == [b"a", b"b"]


@pytest.mark.parametrize("fails", [False, True])
def test_cmd_bench_colmap_switch(tmp_path: Path, monkeypatch: Any, fails: bool) -> None:
    m = _script()
    monkeypatch.setattr(m, "WORK", tmp_path)
    for side in ("east", "west"):
        (tmp_path / f"overview_{side}.json").write_text("{}")
    scene = SimpleNamespace(trees={"test": ["Tree01"]}, gt_xyz=np.zeros((2, 3)), gt_ids=["a", "b"])
    monkeypatch.setattr(m, "load_fuji_sfm", lambda *a, **k: scene)
    monkeypatch.setattr(fc, "prepare_images", lambda *a: [])
    monkeypatch.setattr(m, "build_detector", lambda c: SimpleNamespace(name="fake", version="0"))
    monkeypatch.setattr(
        m, "build_reconstructor", lambda c: SimpleNamespace(name=c.backbone, config=dict)
    )
    ext = _line_cameras(3)
    monkeypatch.setattr(m, "_window", lambda *a: (0, 3, [], ext))
    used: list[str] = []

    def fake_run(paths: Any, prompt: str, cfg: Any, *, reconstructor: Any, **kw: Any) -> Any:
        used.append(reconstructor.name)
        assert (cfg.assoc, cfg.score_threshold, cfg.merge_k) == ("v2", 0.2, 0.3)
        if fails and reconstructor.name == "colmap":
            raise ReconAdaptError("reconstruction failed: no_model")
        return SimpleNamespace(
            recon_cache_key=reconstructor.name, timings={"total": 1.0}, peak_vram_mb=0.0
        )

    monkeypatch.setattr(m, "run", fake_run)
    monkeypatch.setattr(
        m,
        "load_recon",
        lambda cfg, key: SimpleNamespace(
            extrinsics=ext, intrinsics=ext[:, :3, :3], depth=np.ones((3, 4, 4))
        ),
    )
    monkeypatch.setattr(fc, "visible_gt", lambda *a: [np.array([0, 1])] * 3)

    def fake_eval(*a: Any, visible: Any, match_px: float) -> dict[str, Any]:
        assert len(visible) == 3 and match_px == fc.MATCH_PX
        return dict(pred=1, gt=2, assoc_f1=1.0, duplicate_rate=0.0, miss_rate=0.5)

    monkeypatch.setattr(fc, "evaluate_window", fake_eval)
    csv = tmp_path / "bench.csv"
    argv = ["bench", "--split", "test", "--backbone", "colmap", "--ks", "3", "--csv", str(csv),
            "--config", str(REPO / "configs/fuji_core.yaml")]  # fmt: skip
    m.main(argv)
    rows = read_rows(csv)
    assert [r["method"] for r in rows] == ["countonce_v2_colmap"] * 2
    assert {r["status"] for r in rows} == {"failed" if fails else "ok"}
    assert used == ["vggt", "colmap"] * 2
    if fails:
        assert "no_model" in rows[0]["error"]
    m.main(argv)  # resume: ok and failed slices are both finished
    assert len(read_rows(csv)) == 2


def test_report_backbone_comparison() -> None:
    def row(t: str, s: str, method: str, pred: float, status: str = "ok") -> dict[str, str]:
        r = {c: "" for c in COLUMNS}
        r.update(status=status, scene_id=t, sample=s, method=method, n_views="3",
                 pred=str(pred) if status == "ok" else "", gt="10" if status == "ok" else "",
                 config_json=json.dumps({"protocol": fr.PROTOCOL}))  # fmt: skip
        return r

    rows = [row(t, s, fr.VGGT_REF, 9) for t in fr.TEST_TREES for s in fr.SIDES]
    rows += [
        row(t, s, fr.COLMAP, 7, "failed" if s == "west" else "ok")
        for t in fr.TEST_TREES
        for s in fr.SIDES
    ]
    got = fr.backbone_compare(rows)
    assert list(got) == ["3"]
    assert got["3"]["n_failed"] == 8 and got["3"]["failure_rate"] == 0.5
    assert got["3"]["diff"] == pytest.approx(2.0)
    assert fr.backbone_compare(rows[:-1]) == {}  # a missing slice keeps the cell TODO
    assert "8/16 (50%)" in fr.backbone_table({"backbone": got})
    assert fr.TODO in fr.backbone_table({"backbone": {}})
