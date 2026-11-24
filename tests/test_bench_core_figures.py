"""Core figure set, schema evolution and gallery contact sheets (synthetic data under tmp only)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from bench import figures, gallery, runner
from bench.runner import COLUMNS, Job, RunOutput, read_rows, run_jobs

pytest.importorskip("PIL")


def _synthetic_csv(tmp_path, with_runs=True):
    from PIL import Image

    path = tmp_path / "bench.csv"
    rng = np.random.default_rng(0)
    methods = ["co_v1", "co_v2", "co_sync", "co_v2+merge_k=0.25", "co_v2+detector=gdino", "sum"]
    jobs = []
    for s in range(8):
        for m in methods:
            for n in (3, 10, 40):
                jobs.append(Job(f"s{s}", m, n, "even", True, "test", {"tier": "easy"}))

    def fn(job):
        gt = 20.0 + int(job.scene_id[1:])
        bias = {"co_v2+merge_k=0.25": 6.0, "co_v2+detector=gdino": -1.0, "sum": 40.0}.get(
            job.method, 2.0
        )
        pred = gt + bias + rng.normal(0, 1)
        run_dir = None
        if with_runs and job.method == "co_v2" and job.n_views == 10:
            run_dir = tmp_path / "runs" / job.key()
            (run_dir / "overlays").mkdir(parents=True)
            for i in range(3):
                Image.new("RGB", (60, 40 + 10 * i), (50 * i, 90, 120)).save(
                    run_dir / "overlays" / f"frame_{i:04d}.png"
                )
        vram = None if job.method == "sum" else 1000.0 + 100 * job.n_views
        f1 = {"co_v1": 0.6, "co_v2": 0.8, "co_sync": 0.85}.get(job.method)
        return RunOutput(
            pred, gt, gt - 5, gt + 5, 0.5 * job.n_views, vram,
            None if vram is None else {"recon": 0.4 * job.n_views, "total": 0.5 * job.n_views},
            None if f1 is None else f1 + rng.normal(0, 0.02),
            str(run_dir) if run_dir else None,
        )  # fmt: skip

    run_jobs(jobs, fn, path)
    return path


def test_new_columns_roundtrip(tmp_path):
    p = _synthetic_csv(tmp_path)
    r = next(x for x in read_rows(p) if x["method"] == "co_v2" and x["n_views"] == "10")
    assert float(r["peak_vram_mb"]) == 2000.0
    assert '"recon": 4.0' in r["timings_json"]
    assert r["assoc_f1"] != ""
    base = next(x for x in read_rows(p) if x["method"] == "sum")
    assert base["peak_vram_mb"] == "" and base["timings_json"] == ""


def test_old_csv_still_loads_and_resumes(tmp_path):
    old = COLUMNS[: runner._V1_COLUMN_COUNT]
    p = tmp_path / "old.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow({
            **dict.fromkeys(old, ""), "run_key": "k1", "status": "ok", "scene_id": "a",
            "method": "m", "n_views": "3", "ordered": "1", "pred": "5", "gt": "5",
            "config_json": "{}",
        })  # fmt: skip
    rows = read_rows(p)
    assert rows[0]["peak_vram_mb"] == "" and rows[0]["timings_json"] == ""
    assert figures.load_rows(p)[0].peak_vram_mb is None
    assert runner.completed_keys(p) == {"k1"}
    # Resuming appends under the upgraded header without losing the old row.
    assert run_jobs([Job("b", "m", 3)], lambda j: RunOutput(1.0, 1.0, peak_vram_mb=9.0), p) == (
        1, 0, 0,
    )  # fmt: skip
    rows = read_rows(p)
    assert [r["run_key"] for r in rows][0] == "k1" and len(rows) == 2
    assert rows[1]["peak_vram_mb"] == "9.0"
    with p.open(newline="") as f:
        assert csv.DictReader(f).fieldnames == COLUMNS


def test_foreign_prefix_header_rejected(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text(",".join(reversed(COLUMNS)) + "\n")
    with pytest.raises(ValueError):
        read_rows(p)


def test_core_figures_and_ablation_deltas(tmp_path):
    p = _synthetic_csv(tmp_path)
    rows = figures.load_rows(p)
    deltas = {d.method: d for d in figures.ablation_deltas(rows, "co_v2")}
    assert set(deltas) == {"co_v2+merge_k=0.25", "co_v2+detector=gdino"}
    assert deltas["co_v2+merge_k=0.25"].lo > 0  # planted +4 bias is clearly worse
    assert deltas["co_v2+merge_k=0.25"].n_scenes == 8
    assert figures.ablation_deltas(rows) == figures.ablation_deltas(rows, "co_v2")  # base inferred

    outs = figures.make_all(p, tmp_path / "figs", "co_v2")
    names = {o.name for o in outs}
    assert {"assoc_f1.png", "runtime_vs_views.png", "ablation.png"} <= names
    assert all(o.stat().st_size > 0 for o in outs)


def test_optional_figures_skip_without_data(tmp_path):
    p = tmp_path / "b.csv"
    run_jobs([Job("a", "m", 3), Job("b", "m", 3)], lambda j: RunOutput(5.0, 5.0), p)
    names = {o.name for o in figures.make_all(p, tmp_path / "figs")}
    assert "assoc_f1.png" not in names and "ablation.png" not in names


def test_gallery_contact_sheet_next_to_row(tmp_path):
    p = _synthetic_csv(tmp_path)
    out = tmp_path / "figs" / "failure_gallery.md"
    worst = gallery.write_gallery(p, out, 0.10)
    text = out.read_text()
    assert "| overlays |" in text
    sheets = list((tmp_path / "figs" / "gallery").glob("*.png"))
    assert sheets and "![" in text and "gallery/" in text
    assert len(worst) >= 1
    # Rows with no overlays say so; no results are written outside tmp.
    assert "no overlay" in text or all(f"gallery/{s.name}" in text for s in sheets)


def test_gallery_resolves_runs_dir_by_key(tmp_path):
    p = _synthetic_csv(tmp_path, with_runs=False)
    assert "no overlay" in (gallery.write_gallery(p, tmp_path / "g1" / "g.md") and
                            (tmp_path / "g1" / "g.md").read_text())  # fmt: skip
    row = next(r for r in figures.load_rows(p) if r.method == "co_v2")
    from PIL import Image

    d = tmp_path / "runs2" / row.run_key / "overlays"
    d.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(d / "frame_0000.png")
    assert gallery.overlay_paths(row, tmp_path / "runs2")
    assert gallery.overlay_paths(row) == []


def test_contact_sheet_subsamples_and_rejects_empty(tmp_path):
    from PIL import Image

    imgs = []
    for i in range(20):
        f = tmp_path / f"{i}.png"
        Image.new("RGB", (30, 20)).save(f)
        imgs.append(f)
    out = gallery.contact_sheet(imgs, tmp_path / "s.png", thumb=32, cols=4)
    with Image.open(out) as im:
        assert im.size == (4 * 32, 3 * 32)  # 12 frames -> 3 rows of 4
    with pytest.raises(ValueError):
        gallery.contact_sheet([], tmp_path / "e.png")
