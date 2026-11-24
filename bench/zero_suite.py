"""Zero-count and distractor suite (PLAN §9.8b): scenes whose true count is 0.

Three sources, all appended to the bench CSV with ``gt=0``:

* seeded synthetic CPU scenes (plane reconstructor + scripted fake detector) that are empty or
  hold only look-alike distractors (low-score fake detections);
* wrong-prompt jobs on existing bench scenes, declared in ``gt.json`` as
  ``wrong_prompts: ["a prompt whose true count is 0", ...]``;
* a summary table (false-positive rate, mean false count, interval-includes-0 rate), each with
  a scene-level bootstrap CI from :mod:`bench.stats`.

The plane reconstructor mirrors the one in ``tests/test_pipeline.py``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bench.metrics import zero_count_metrics
from bench.runner import Job, RunOutput, read_rows, run_jobs
from bench.stats import bootstrap_ci
from countonce.config import Config
from countonce.detect.base import Detector, FakeDetector
from countonce.ingest.frames import list_images
from countonce.pipeline import run
from countonce.recon.base import Reconstructor
from countonce.types import Frame, Recon

W, H, F, DEPTH = 64, 48, 50.0, 5.0
CAM_X = (0.0, 0.3, 0.6, 0.9)
ZERO_DOMAIN = "zero_suite"
SYNTH_PROMPT = "widgets"


class PlaneRecon:
    """Fake reconstructor: every view looks at the plane z=DEPTH, cameras slide along x."""

    name = "fake"

    def config(self) -> dict[str, Any]:
        return {}

    def reconstruct(self, frames: list[Frame]) -> Recon:
        n = len(frames)
        k = np.tile(np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]]), (n, 1, 1))
        ext = np.tile(np.eye(3, 4), (n, 1, 1))
        wp = np.zeros((n, H, W, 3))
        ys, xs = np.mgrid[0:H, 0:W]
        for i in range(n):
            cx = CAM_X[i % len(CAM_X)]
            wp[i, ..., 0] = (xs + 0.5 - W / 2) * DEPTH / F + cx
            wp[i, ..., 1] = (ys + 0.5 - H / 2) * DEPTH / F
            wp[i, ..., 2] = DEPTH
            ext[i, 0, 3] = -cx
        return Recon(
            backbone="fake",
            intrinsics=k,
            extrinsics=ext,
            depth=np.full((n, H, W), DEPTH),
            depth_conf=np.ones((n, H, W)),
            world_points=wp,
            world_points_conf=np.full((n, H, W), 2.0),
        )


def distractor_script(
    rng: np.random.Generator, n_objects: int, max_score: float
) -> dict[int, list[tuple[float, float, float, float, float]]]:
    """Look-alike detections of ``n_objects`` fixed plane positions, seen from every camera,
    each with a low score in [0.05, max_score]."""
    objs = rng.uniform([-1.4, -0.8], [1.0, 0.8], size=(n_objects, 2))
    scores = rng.uniform(0.05, max_score, size=(n_objects, len(CAM_X)))
    script: dict[int, list[tuple[float, float, float, float, float]]] = {}
    for i, cx in enumerate(CAM_X):
        boxes = []
        for j, (ox, oy) in enumerate(objs):
            u = F * (ox - cx) / DEPTH + W / 2
            v = F * oy / DEPTH + H / 2
            if 4 <= u < W - 4 and 4 <= v < H - 4:
                boxes.append((u - 3, v - 3, u + 3, v + 3, round(float(scores[j, i]), 4)))
        script[i] = boxes
    return script


def generate_scenes(
    out_dir: Path | str, n_scenes: int = 12, seed: int = 0, max_distractor_score: float = 0.45
) -> list[dict[str, Any]]:
    """Write ``n_scenes`` (clamped to 10-20 by convention, not enforced) zero-count scenes.

    Even-indexed scenes are empty; odd ones hold 2-5 distractors. Each scene directory gets
    ``images/``, ``gt.json`` (``count`` 0, ``synthetic_zero`` kind and the detector script) and is
    returned in ``discover_scenes`` format. Deterministic in ``seed``.
    """
    out = Path(out_dir)
    scenes = []
    for s in range(n_scenes):
        rng = np.random.default_rng([seed, s])
        d = out / f"zero{s:02d}"
        (d / "images").mkdir(parents=True, exist_ok=True)
        for i in range(len(CAM_X)):
            arr = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / "images" / f"f{i}.png")
        kind = "empty" if s % 2 == 0 else "distractor"
        script = (
            {}
            if kind == "empty"
            else distractor_script(rng, int(rng.integers(2, 6)), max_distractor_score)
        )
        gt = {
            "count": 0,
            "tier": "zero",
            "prompt": SYNTH_PROMPT,
            "synthetic_zero": kind,
            "script": {str(k): v for k, v in script.items()},
        }
        (d / "gt.json").write_text(json.dumps(gt, sort_keys=True))
        scenes.append({"id": d.name, "dir": d, "tier": "zero", "gt": gt})
    return scenes


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "prompt"


def wrong_prompt_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One zero-count scene per ``wrong_prompts`` entry of an existing scene's ``gt.json``."""
    out = []
    for sc in scenes:
        for p in sc["gt"].get("wrong_prompts", []) or []:
            out.append(
                {
                    "id": f"{sc['id']}@wrong-{_slug(p)}",
                    "dir": sc["dir"],
                    "tier": "zero",
                    "gt": {**sc["gt"], "count": 0, "prompt": p, "wrong_prompt_of": sc["id"]},
                }
            )
    return out


def build_zero_jobs(
    method: str, options: dict[str, Any], scenes: list[dict[str, Any]], split: str, seed: int
) -> list[Job]:
    return [
        Job(
            scene_id=sc["id"],
            method=method,
            n_views=None,
            sample="zero",
            ordered=True,
            split=split,
            config={"kind": "zero_suite", "options": options, "tier": "zero", "seed": seed,
                    "prompt": sc["gt"].get("prompt", "")},
        )
        for sc in scenes
    ]  # fmt: skip


def make_zero_executor(
    scenes: list[dict[str, Any]],
    out_dir: Path,
    cache_dir: Path | None = None,
    n_views: int = 5,
    detector_factory: Callable[[Any], Detector] | None = None,
    reconstructor_factory: Callable[[Any], Reconstructor] | None = None,
) -> Callable[[Job], RunOutput]:
    """Synthetic scenes use the scripted fake detector and plane reconstructor; wrong-prompt
    scenes use the factories (default: the configured real backends) on evenly sampled views."""
    by_id = {s["id"]: s for s in scenes}

    def fn(job: Job) -> RunOutput:
        sc = by_id[job.scene_id]
        gt = sc["gt"]
        conf = Config(cache_dir=cache_dir, **job.config["options"]).with_overrides(
            out_dir=out_dir / job.key()
        )
        t0 = time.perf_counter()
        paths = list_images(sc["dir"])
        if "synthetic_zero" in gt:
            script = {int(k): [tuple(b) for b in v] for k, v in gt["script"].items()}
            conf = conf.with_overrides(max_frames=0)
            det: Detector | None = FakeDetector(script)
            recon: Reconstructor | None = PlaneRecon()
        else:
            from countonce.ingest.frames import sample_frames

            paths = sample_frames(paths, n_views, None)
            det = detector_factory(conf) if detector_factory else None
            recon = reconstructor_factory(conf) if reconstructor_factory else None
        res = run(paths, gt["prompt"], conf, detector=det, reconstructor=recon)
        return RunOutput(
            pred=float(res.count),
            gt=0.0,
            low=float(res.interval.low),
            high=float(res.interval.high),
            runtime_s=time.perf_counter() - t0,
            peak_vram_mb=float(res.peak_vram_mb),
            timings={k: float(v) for k, v in res.timings.items()},
            run_dir=str(conf.out_dir),
            domain=ZERO_DOMAIN,
        )

    return fn


def run_zero_suite(
    csv_path: Path,
    method: str,
    options: dict[str, Any],
    scenes_dir: Path,
    real_scenes: list[dict[str, Any]],
    *,
    n_synthetic: int = 12,
    seed: int = 0,
    split: str = "test",
    runs_dir: Path = Path("results/runs"),
    cache_dir: Path | None = None,
    n_views: int = 5,
    detector_factory: Callable[[Any], Detector] | None = None,
    reconstructor_factory: Callable[[Any], Reconstructor] | None = None,
) -> tuple[int, int, int]:
    """Generate synthetic scenes under ``scenes_dir``, add the wrong-prompt scenes of
    ``real_scenes`` and append one ``gt=0`` row per scene for ``method``. Resumable."""
    scenes = generate_scenes(scenes_dir, n_synthetic, seed) + wrong_prompt_scenes(real_scenes)
    jobs = build_zero_jobs(method, options, scenes, split, seed)
    fn = make_zero_executor(
        scenes, runs_dir, cache_dir, n_views, detector_factory, reconstructor_factory
    )
    return run_jobs(jobs, fn, csv_path, {"zero_suite": True, "method": method, "options": options})


# ---------------------------------------------------------------- summary


def _f(x: str) -> float:
    return float(x) if x != "" else float("nan")


def summarize_zero(csv_path: Path | str, n_boot: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    """Per-method zero-count summary from the bench CSV. Rows with ``gt == 0`` and ``status ==
    ok`` are first averaged within each scene, then the scene is the bootstrap unit. Each metric
    is ``(estimate, low, high)`` or None when undefined (e.g. no intervals)."""
    per: dict[str, dict[str, list[dict[str, str]]]] = {}
    for r in read_rows(csv_path):
        if r["status"] != "ok" or r["gt"] == "" or float(r["gt"]) != 0:
            continue
        per.setdefault(r["method"], {}).setdefault(r["scene_id"], []).append(r)
    out = []
    for method, by_scene in sorted(per.items()):
        fpr, mfc, inc = [], [], []
        for rows in by_scene.values():
            pred = [_f(r["pred"]) for r in rows]
            low = [_f(r["low"]) for r in rows]
            high = [_f(r["high"]) for r in rows]
            has_iv = not (np.isnan(low).any() or np.isnan(high).any())
            m = zero_count_metrics(pred, [0.0] * len(rows), low if has_iv else None,
                                   high if has_iv else None)  # fmt: skip
            fpr.append(m["false_positive_rate"])
            mfc.append(m["mean_false_count"])
            inc.append(m.get("interval_includes_zero_rate", float("nan")))

        def ci(v: list[float]) -> tuple[float, float, float] | None:
            if np.isnan(v).all():
                return None
            c = bootstrap_ci(v, n_boot=n_boot, seed=seed)
            return (c.estimate, c.low, c.high)

        out.append(
            {
                "method": method,
                "n_scenes": len(by_scene),
                "false_positive_rate": ci(fpr),
                "mean_false_count": ci(mfc),
                "interval_includes_zero_rate": ci(inc),
            }
        )
    return out


def _cell(v: tuple[float, float, float] | None) -> str:
    return "n/a" if v is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def zero_summary_markdown(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Zero-count and distractor suite (scene bootstrap 95% CI)",
        "",
        "| method | scenes | FPR | mean false count | interval includes 0 |",
        "|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['method']} | {s['n_scenes']} | {_cell(s['false_positive_rate'])} | "
            f"{_cell(s['mean_false_count'])} | {_cell(s['interval_includes_zero_rate'])} |"
        )
    if not summary:
        lines.append("| TODO: fill from results | | | | |")
    return "\n".join(lines) + "\n"


def write_zero_summary(csv_path: Path | str, out_path: Path | str) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(zero_summary_markdown(summarize_zero(csv_path)))
    return p
