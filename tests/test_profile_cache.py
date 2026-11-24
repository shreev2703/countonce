"""Repeat-job cache profiling (PLAN 13 item 6b). CPU only: tiny scene, color_blob + fixture."""

import csv
import importlib.util
import sys
from pathlib import Path

from countonce.cli import main
from countonce.config import Config
from countonce.profile import (
    COLUMNS,
    CacheProfile,
    ProfileConfig,
    load_profile_csv,
    run_profile_cache,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "examples" / "tiny_scene"


def _script():
    spec = importlib.util.spec_from_file_location(
        "make_profile_report", ROOT / "scripts" / "make_profile_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_profile_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cfg(tmp_path: Path, repeats: int = 2) -> ProfileConfig:
    return ProfileConfig(
        inputs=[SCENE],
        prompt="red sphere",
        repeats=repeats,
        warmup=0,
        out_csv=tmp_path / "profile.csv",
        run=Config(
            detector="color_blob",
            backbone="fixture",
            backbone_options={"scene": str(SCENE / "scene.json")},
            assoc="v1",
            out_dir=tmp_path / "out",
        ),
        cache=CacheProfile("blue sphere", tmp_path / "cache"),
    )


def test_cache_states_skip_the_right_stages(tmp_path):
    _, rows = run_profile_cache(_cfg(tmp_path))
    totals = {(r.repeat, r.cache_state): r for r in rows if r.stage == "total"}
    assert len(totals) == 6
    for rep in (0, 1):
        assert (totals[rep, "cold"].recon_cache_hit, totals[rep, "cold"].det_cache_hit) == (
            False,
            False,
        )
        # new prompt: reconstruction reused, detection recomputed
        assert totals[rep, "new_prompt"].recon_cache_hit is True
        assert totals[rep, "new_prompt"].det_cache_hit is False
        # exact repeat: both reused
        assert totals[rep, "repeat"].recon_cache_hit is True
        assert totals[rep, "repeat"].det_cache_hit is True
    assert not (tmp_path / "cache" / "warmup").exists()


def test_cache_variant_does_not_call_reconstruct_or_detect_on_hits(tmp_path):
    from countonce.detect.base import FakeDetector
    from countonce.pipeline import run
    from tests.test_pipeline import PlaneRecon, make_scene  # fake backends, 4 views

    img_dir, script = make_scene(tmp_path)
    calls = {"recon": 0, "detect": 0}

    class CountingRecon(PlaneRecon):
        def reconstruct(self, frames):
            calls["recon"] += 1
            return super().reconstruct(frames)

    class CountingDet(FakeDetector):
        def detect(self, frames, prompt, exemplars=None):
            calls["detect"] += 1
            return super().detect(frames, prompt, exemplars)

    def run_fn(c, prompt):
        return run(
            img_dir,
            prompt,
            c,
            export=False,
            detector=CountingDet(script),
            reconstructor=CountingRecon(),
        )

    cfg = ProfileConfig(
        inputs=[img_dir],
        prompt="widgets",
        repeats=1,
        warmup=0,
        out_csv=tmp_path / "p.csv",
        run=Config(out_dir=tmp_path / "out", max_frames=0),
        cache=CacheProfile("gadgets", tmp_path / "cache"),
    )
    seen = []

    def spy(c, prompt):
        before = dict(calls)
        res = run_fn(c, prompt)
        seen.append({k: calls[k] - before[k] for k in calls})
        return res

    run_profile_cache(cfg, run_fn=spy)
    assert seen == [
        {"recon": 1, "detect": 1},  # cold
        {"recon": 0, "detect": 1},  # new prompt: reconstruct skipped
        {"recon": 0, "detect": 0},  # exact repeat: reconstruct and detect skipped
    ]


def test_columns_append_only_and_old_csv_loads(tmp_path):
    assert COLUMNS[:8] == [
        "repeat", "stage", "wall_s", "peak_vram_mb", "n_frames", "backbone", "detector", "assoc",
    ]  # fmt: skip
    assert COLUMNS[8:] == ["cache_state", "recon_cache_hit", "det_cache_hit"]
    old = tmp_path / "old.csv"
    old.write_text(
        "repeat,stage,wall_s,peak_vram_mb,n_frames,backbone,detector,assoc\n"
        "0,total,1.5,,4,vggt,countgd,v1\n"
    )
    (row,) = load_profile_csv(old)
    assert (row.cache_state, row.recon_cache_hit, row.det_cache_hit) == ("", None, None)
    assert row.wall_s == 1.5 and row.peak_vram_mb is None
    # the report script reads the old file without error and finds no cache rows
    mod = _script()
    assert mod.summarize_cache(mod.load_rows(old)) == []
    assert "TODO: fill from results" in mod.render_cache_block(old)


def test_cli_variant_cache_writes_columns(tmp_path, capsys):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        f"inputs: {SCENE}\nprompt: red sphere\nwarmup: 0\nrepeats: 1\nout_csv: {tmp_path}/p.csv\n"
        "run: {detector: color_blob, backbone: fixture, "
        f"backbone_options: {{scene: {SCENE}/scene.json}}, out_dir: {tmp_path}/out}}\n"
        f"cache: {{new_prompt: blue sphere, dir: {tmp_path}/cache}}\n"
    )
    assert main(["profile", "--config", str(cfg), "--variant", "cache"]) == 0
    with (tmp_path / "p.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["cache_state"] for r in rows} == {"cold", "new_prompt", "repeat"}
    assert main(["profile", "--config", str(ROOT / "configs" / "profile.yaml")]) in (0, 2)  # parses


def test_report_renders_todo_on_empty_and_numbers_from_rows(tmp_path):
    mod = _script()
    empty = tmp_path / "empty.csv"
    empty.write_text(",".join(COLUMNS) + "\n")
    block = mod.render_cache_block(empty)
    assert "Repeat-job caching" in block and "TODO: fill from results" in block
    assert "| Cold job" not in block
    assert "TODO: fill from results" in mod.render_cache_block(tmp_path / "missing.csv")

    _, rows = run_profile_cache(_cfg(tmp_path, repeats=3))
    block = mod.render_cache_block(tmp_path / "profile.csv")
    for label in ("Cold job", "New prompt", "Exact repeat"):
        assert label in block
    assert "| 1.00x |" in block  # cold vs itself
    summ = mod.summarize_cache(mod.load_rows(tmp_path / "profile.csv"))
    assert [s["state"] for s in summ] == ["cold", "new_prompt", "repeat"]
    assert all(s["n"] == 3 and s["p90"] >= s["p50"] for s in summ)
    # idempotent splice into an existing doc
    doc = mod.update_doc(None, mod.render_block(tmp_path / "none.csv"), block)
    assert mod.update_doc(doc, mod.render_block(tmp_path / "none.csv"), block) == doc
