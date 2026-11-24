"""Split conformal, Jackknife+, calibration builder, run() wiring and figure 11 (synthetic only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench import figures
from bench.calibration import (
    CalibrationScene,
    SplitViolation,
    build_calibration_set,
)
from bench.splits import write_splits
from countonce.conformal import (
    CalibrationArtifact,
    CalibrationError,
    coverage_by_level,
    interval_from_artifact,
    jackknife_plus_interval,
    leave_one_out,
    min_calibration_size,
    resolve_method,
    sigma_hat,
    split_conformal_interval,
)

LEVELS = (0.8, 0.9, 0.95)


def _data(rng, n):
    """Heteroscedastic scenes: noise sd proportional to the range width (known noise)."""
    true = rng.integers(10, 200, n).astype(float)
    width = rng.uniform(2, 20, n)
    pred = true + rng.normal(0, 1, n) * width / 2
    return true, pred, pred - width / 2, pred + width / 2


@pytest.mark.parametrize("level", LEVELS)
def test_split_conformal_coverage(level):
    rng = np.random.default_rng(1)
    yc, pc, lc, hc = _data(rng, 3000)
    from countonce.conformal import build_artifact

    art = build_artifact(yc, pc, lc, hc)
    yt, pt, lt, ht = _data(rng, 20000)
    cov = coverage_by_level(art, yt, pt, lt, ht, [level])[level]
    assert abs(cov - level) < 0.03


@pytest.mark.parametrize("level", LEVELS)
def test_jackknife_plus_coverage(level):
    rng = np.random.default_rng(2)

    def make(n):
        x = rng.uniform(10, 100, n)
        return x, 1.2 * x + rng.normal(0, 4, n)  # tuned multiplier 1.2, noise sd 4

    xc, yc = make(150)
    scenes = [
        CalibrationScene(str(i), y, x, 0, 0, "synthetic")
        for i, (x, y) in enumerate(zip(xc, yc, strict=True))
    ]
    fit = lambda tr: sum(s.y for s in tr) / sum(s.count for s in tr)  # noqa: E731
    thetas, res = leave_one_out(scenes, lambda s: s.y, fit, lambda th, s: th * s.count)
    xt, yt = make(4000)
    hit = 0
    for x, y in zip(xt, yt, strict=True):
        lo, hi = jackknife_plus_interval([t * x for t in thetas], res, level)
        hit += lo <= y <= hi
    assert abs(hit / len(xt) - level) < 0.03


def test_small_n_errors():
    assert min_calibration_size(0.9) == 9 and min_calibration_size(0.95) == 19
    from countonce.conformal import conformal_quantile

    conformal_quantile(np.arange(9.0), 0.9)  # n = 9 is fine
    with pytest.raises(CalibrationError):
        conformal_quantile(np.arange(8.0), 0.9)
    with pytest.raises(CalibrationError):
        conformal_quantile(np.arange(18.0), 0.95)
    with pytest.raises(CalibrationError):
        split_conformal_interval(10, 2, [1.0] * 5, 0.9)
    with pytest.raises(CalibrationError):
        jackknife_plus_interval([1.0] * 5, [1.0] * 5, 0.9)


def test_sigma_floor():
    assert sigma_hat(5, 5) == 1.0 and sigma_hat(0, 10) == 10.0


def _splits(tmp_path, val, test):
    p = tmp_path / "splits.json"
    hashes = {s: "h" for s in [*val, *test]}
    write_splits(hashes, p)
    doc = json.loads(p.read_text())
    doc["splits"] = {"val": {s: "h" for s in val}, "test": {s: "h" for s in test}}
    p.write_text(json.dumps(doc))
    return p


def _scenes(ids, domain="real"):
    return [
        CalibrationScene(i, 20 + n, 20 + n + (n % 3) - 1, 18 + n, 24 + n, domain,
                         {0.5: 19 + n, 1.0: 20 + n + (n % 2), 2.0: 22 + n})
        for n, i in enumerate(ids)
    ]  # fmt: skip


def test_builder_refuses_test_split(tmp_path):
    p = _splits(tmp_path, [f"v{i}" for i in range(10)], ["t0", "t1"])
    with pytest.raises(SplitViolation):
        build_calibration_set(_scenes(["v0", "v1", "t0"]), p)
    with pytest.raises(SplitViolation):
        build_calibration_set(_scenes(["v0", "v1"]), p, split="test")
    with pytest.raises(SplitViolation):  # unassigned real scene
        build_calibration_set(_scenes(["v0", "zz"]), p)
    with pytest.raises(SplitViolation):  # a synthetic scene cannot reuse a test id
        build_calibration_set(_scenes(["t1"], "synthetic"), p)
    art = build_calibration_set(_scenes([f"v{i}" for i in range(10)]), p, tuned="k")
    assert art.n_calibration == art.n_real == 10 and art.has_jackknife
    assert set(art.thetas) <= {0.5, 1.0, 2.0}
    art2 = build_calibration_set(_scenes(["s1", "s2"], "synthetic"), p)
    assert art2.n_real == 0


def test_artifact_roundtrip_and_resolve(tmp_path):
    p = _splits(tmp_path, [f"v{i}" for i in range(20)], ["t0"])
    big = build_calibration_set(_scenes([f"v{i}" for i in range(20)]), p, tuned="scale")
    path = big.save(tmp_path / "cal.json")
    assert CalibrationArtifact.load(path) == big
    assert resolve_method("conformal", big).method == "split_conformal"
    small = build_calibration_set(_scenes([f"v{i}" for i in range(10)]), p, tuned="scale")
    r = resolve_method("conformal", small)
    assert r.method == "jackknife_plus" and r.warning["code"] == "conformal_fallback"
    nojk = build_calibration_set(_scenes([f"v{i}" for i in range(10)]), p)
    assert resolve_method("conformal", nojk).method == "sensitivity"
    with pytest.raises(CalibrationError):
        resolve_method("jackknife_plus", nojk)
    with pytest.raises(CalibrationError):
        resolve_method("conformal", None)
    lo, hi = interval_from_artifact("split_conformal", big, 25, 22, 28, 0.9)
    assert lo <= 25 <= hi


def test_run_records_interval_method(tmp_path):
    from countonce.config import Config
    from countonce.detect.base import FakeDetector
    from countonce.pipeline import run
    from tests.test_pipeline import PlaneRecon, make_scene

    img_dir, script = make_scene(tmp_path)

    def _run_fixture(out, **kw):
        cfg = Config(out_dir=out, max_frames=0, **kw)
        det = FakeDetector(script)
        return run(img_dir, "widgets", cfg, detector=det, reconstructor=PlaneRecon())

    p = _splits(tmp_path, [f"v{i}" for i in range(20)], ["t0"])
    art = build_calibration_set(_scenes([f"v{i}" for i in range(20)]), p, tuned="scale")
    cal = art.save(tmp_path / "cal.json")
    for method, expect in (("conformal", "split_conformal"), ("jackknife_plus", "jackknife_plus")):
        res = _run_fixture(tmp_path / method, interval_method=method, calibration_path=cal)
        assert res.interval.method == expect and res.interval.n_calibration == 20
        assert res.interval.level == 0.9 and res.interval.low <= res.count <= res.interval.high
        doc = json.loads((tmp_path / method / "result.json").read_text())
        assert doc["interval"]["method"] == expect and doc["interval"]["n_calibration"] == 20
    with pytest.raises(CalibrationError):
        _run_fixture(tmp_path / "x", interval_method="conformal", calibration_path=None)


def test_figure_conformal_coverage(tmp_path):
    rng = np.random.default_rng(3)
    from countonce.conformal import build_artifact

    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    curves = {}
    for dom, n in (("synthetic", 500), ("real", 40)):
        yc, pc, lc, hc = _data(rng, n)
        art = build_artifact(yc, pc, lc, hc)
        curves[dom] = coverage_by_level(art, *_data(rng, 2000), levels)
    assert 0.95 not in curves["real"] or curves["real"][0.95] > 0.8
    out = figures.fig_conformal_coverage(curves, tmp_path / "fig11.png", {"real": 40})
    assert out is not None and out.stat().st_size > 0
    assert figures.fig_conformal_coverage({}, tmp_path / "none.png") is None
