"""Golden benchmark regression gate (PLAN §16B.2).

``python -m bench.golden run [--cpu-synthetic]`` runs the frozen 5-scene golden set from
``configs/golden.yaml`` and writes ``results/golden/<git-sha>.json`` (MAPE, association F1,
peak VRAM, latency per scene and overall). ``python -m bench.golden compare --baseline X.json``
prints a Markdown delta table against a baseline and exits 1 on a regression beyond the
configured tolerances.

Two modes share one result format:

* real (GPU): scenes are directories under ``--data-root``; each is pinned by the
  ``bench.splits.scene_hash`` digest and refused if the hash is missing or differs.
* ``--cpu-synthetic``: seeded synthetic plane scenes with a scripted fake detector and the
  plane reconstructor from :mod:`bench.zero_suite`; each scene spec is pinned by a SHA-256 of
  its canonical JSON. Safe to run on untrusted code in hosted CI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from bench.metrics import mape, pairwise_prf
from bench.splits import scene_hash
from bench.zero_suite import CAM_X, DEPTH, F, H, PlaneRecon, W
from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.pipeline import run

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "golden.yaml"
DEFAULT_OUT_DIR = Path("results/golden")
RESULT_VERSION = 1

# metric -> (direction, kind). "up" = higher is worse, "down" = lower is worse.
# kind "abs" compares the raw difference, "rel" the relative increase.
METRICS: dict[str, tuple[str, str]] = {
    "mape": ("up", "abs"),
    "assoc_f1": ("down", "abs"),
    "peak_vram_mb": ("up", "rel"),
    "latency_s": ("up", "rel"),
}
DEFAULT_TOLERANCES: dict[str, float | None] = {
    "mape": 0.02,
    "assoc_f1": 0.02,
    "peak_vram_mb": 0.10,
    "latency_s": 0.25,
}


# ---------------------------------------------------------------- config / pinning


def spec_hash(spec: dict[str, Any]) -> str:
    """SHA-256 of a synthetic scene spec (everything except its own ``sha256`` key)."""
    import hashlib

    body = {k: v for k, v in spec.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping")
    return doc


def tolerances(
    cfg: dict[str, Any], overrides: dict[str, float | None] | None = None
) -> dict[str, float | None]:
    tol = {**DEFAULT_TOLERANCES, **(cfg.get("tolerances") or {})}
    tol.update(overrides or {})
    unknown = set(tol) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown tolerance keys: {sorted(unknown)}")
    return tol


def verify_synthetic_pins(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = cfg.get("synthetic_scenes") or []
    for s in scenes:
        if s.get("sha256") != spec_hash(s):
            raise ValueError(f"synthetic scene {s.get('id')}: spec hash differs from the pin")
    return list(scenes)


def verify_real_pins(cfg: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    scenes = cfg.get("scenes") or []
    for s in scenes:
        if not s.get("sha256"):
            raise ValueError(f"scene {s['id']}: no pinned sha256 (run `bench.golden freeze`)")
        got = scene_hash(data_root / s["dir"])
        if got != s["sha256"]:
            raise ValueError(f"scene {s['id']}: hash {got[:12]} != pinned {s['sha256'][:12]}")
    return list(scenes)


# ---------------------------------------------------------------- running


def git_sha() -> str:
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def synthetic_scene(
    spec: dict[str, Any], root: Path
) -> tuple[Path, dict[int, list[tuple[float, float, float, float, float]]], dict[str, int]]:
    """Write images and return (image dir, detector script, detection id -> GT object index)."""
    rng = np.random.default_rng(int(spec["seed"]))
    n = int(spec["n_objects"])
    xs = np.linspace(-1.0, 1.4, n) + rng.uniform(-0.05, 0.05, n)
    ys = rng.uniform(-0.6, 0.6, n)
    img_dir = root / spec["id"]
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(CAM_X)):
        arr = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"f{i}.png")
    script: dict[int, list[tuple[float, float, float, float, float]]] = {}
    gt_of: dict[str, int] = {}
    for i, cx in enumerate(CAM_X):
        boxes = []
        for j, (ox, oy) in enumerate(zip(xs, ys, strict=True)):
            u = F * (ox - cx) / DEPTH + W / 2
            v = F * oy / DEPTH + H / 2
            if 4 <= u < W - 4 and 4 <= v < H - 4:
                gt_of[f"{i}:{len(boxes)}"] = j
                boxes.append((u - 3, v - 3, u + 3, v + 3, 0.9))
        script[i] = boxes
    return img_dir, script, gt_of


def _cluster_labels(res: Any, gt_of: dict[str, int]) -> tuple[list[int], list[int | None]]:
    cluster: dict[str, int] = {}
    for k, inst in enumerate(res.instances):
        for m in inst.members:
            cluster[m] = k
    ids = sorted(gt_of)
    labels = [cluster.get(d, -1 - i) for i, d in enumerate(ids)]
    return labels, [gt_of[d] for d in ids]


def _scene_row(
    pred: float, gt: float, f1: float | None, vram: float, latency: float
) -> dict[str, Any]:
    return {
        "pred": pred,
        "gt": gt,
        "mape": mape([pred], [gt]),
        "assoc_f1": f1,
        "peak_vram_mb": vram,
        "latency_s": latency,
    }


def run_synthetic(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for spec in verify_synthetic_pins(cfg):
            img_dir, script, gt_of = synthetic_scene(spec, root / "scenes")
            conf = Config(
                out_dir=root / "out" / spec["id"], max_frames=0, assoc=spec.get("assoc", "v2")
            )
            t0 = time.perf_counter()
            res = run(
                img_dir,
                spec.get("prompt", "widgets"),
                conf,
                detector=FakeDetector(script),
                reconstructor=PlaneRecon(),
            )
            latency = time.perf_counter() - t0
            labels, gt = _cluster_labels(res, gt_of)
            f1 = pairwise_prf(labels, gt)[2]
            out[spec["id"]] = _scene_row(
                float(res.count), float(spec["n_objects"]), f1, float(res.peak_vram_mb), latency
            )
    return out


def run_real(cfg: dict[str, Any], data_root: Path) -> dict[str, dict[str, Any]]:
    from countonce.ingest.frames import list_images

    out: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as td:
        for s in verify_real_pins(cfg, data_root):
            d = data_root / s["dir"]
            paths = list_images(d / "images" if (d / "images").is_dir() else d)
            conf = Config(out_dir=Path(td) / s["id"])
            t0 = time.perf_counter()
            res = run(paths, s["prompt"], conf)
            latency = time.perf_counter() - t0
            out[s["id"]] = _scene_row(
                float(res.count), float(s["gt_count"]), None, float(res.peak_vram_mb), latency
            )
    return out


def _mean(vals: Sequence[float | None]) -> float | None:
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else None


def summarize(scenes: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    rows = list(scenes.values())
    return {
        "mape": _mean([r["mape"] for r in rows]),
        "assoc_f1": _mean([r["assoc_f1"] for r in rows]),
        "peak_vram_mb": max((r["peak_vram_mb"] for r in rows), default=None),
        "latency_s": _mean([r["latency_s"] for r in rows]),
    }


def build_result(scenes: dict[str, dict[str, Any]], mode: str) -> dict[str, Any]:
    return {
        "version": RESULT_VERSION,
        "git_sha": git_sha(),
        "mode": mode,
        "scenes": scenes,
        "overall": summarize(scenes),
    }


# ---------------------------------------------------------------- compare


def _delta(metric: str, base: float | None, cur: float | None) -> float | None:
    if base is None or cur is None:
        return None
    _, kind = METRICS[metric]
    if kind == "abs":
        return cur - base
    return (cur - base) / base if base else (0.0 if cur == base else math.inf)


def _regressed(metric: str, delta: float | None, tol: float | None) -> bool:
    if delta is None or tol is None:
        return False
    direction, _ = METRICS[metric]
    worse = delta if direction == "up" else -delta
    return worse > tol + 1e-12


def compare(
    baseline: dict[str, Any], current: dict[str, Any], tol: dict[str, float | None]
) -> tuple[str, list[str]]:
    """Markdown delta table and the list of regression descriptions (empty == pass).

    A scene or overall metric that is missing in the current run but present in the baseline
    counts as a regression. Metrics with tolerance ``None`` are shown but never fail.
    """
    problems: list[str] = []
    lines = [
        "| scene | metric | baseline | current | delta | tolerance | status |",
        "|---|---|---|---|---|---|---|",
    ]

    def fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.4g}"

    def rows(name: str, b: dict[str, Any], c: dict[str, Any]) -> None:
        for m in METRICS:
            bv, cv = b.get(m), c.get(m)
            d = _delta(m, bv, cv)
            if bv is not None and cv is None:
                status = "MISSING"
                problems.append(f"{name}/{m}: missing in current run")
            elif _regressed(m, d, tol.get(m)):
                status = "REGRESSION"
                problems.append(
                    f"{name}/{m}: {fmt(bv)} -> {fmt(cv)} (delta {fmt(d)}, tol {tol[m]})"
                )
            else:
                status = "ok"
            unit = "%" if METRICS[m][1] == "rel" else ""
            dtxt = "n/a" if d is None else (f"{d * 100:+.1f}%" if unit else f"{d:+.4g}")
            t = tol.get(m)
            lines.append(
                f"| {name} | {m} | {fmt(bv)} | {fmt(cv)} | {dtxt} | "
                f"{'off' if t is None else t} | {status} |"
            )

    for sid, brow in baseline.get("scenes", {}).items():
        crow = current.get("scenes", {}).get(sid)
        if crow is None:
            problems.append(f"{sid}: scene missing in current run")
            lines.append(f"| {sid} | all | | | | | MISSING |")
            continue
        rows(sid, brow, crow)
    rows("**overall**", baseline.get("overall", {}), current.get("overall", {}))
    head = (
        f"Golden benchmark: baseline `{baseline.get('git_sha')}` "
        f"vs current `{current.get('git_sha')}`\n\n"
    )
    return head + "\n".join(lines) + "\n", problems


# ---------------------------------------------------------------- CLI


def _parse_tol(items: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = None if v.lower() in {"off", "none"} else float(v)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.golden")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the golden set and write results/golden/<sha>.json")
    r.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    r.add_argument("--cpu-synthetic", action="store_true")
    r.add_argument("--data-root", type=Path, default=Path("data"))
    r.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    r.add_argument("--out", type=Path, default=None, help="exact output file (overrides --out-dir)")
    c = sub.add_parser("compare", help="Markdown delta table; exit 1 on regression")
    c.add_argument("--baseline", type=Path, required=True)
    c.add_argument("--current", type=Path, default=None, help="default: run now (needs --mode)")
    c.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    c.add_argument("--cpu-synthetic", action="store_true")
    c.add_argument("--data-root", type=Path, default=Path("data"))
    c.add_argument("--tol", action="append", default=[], metavar="METRIC=VALUE|off")
    c.add_argument("--table-out", type=Path, default=None)
    f = sub.add_parser("freeze", help="write sha256 pins of real scenes into the config")
    f.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    f.add_argument("--data-root", type=Path, default=Path("data"))
    a = ap.parse_args(argv)

    if a.cmd == "freeze":
        raise SystemExit(_freeze(a.config, a.data_root))

    cfg = load_config(a.config)
    if a.cmd == "run":
        res = _run(cfg, a.cpu_synthetic, a.data_root)
        path = a.out or a.out_dir / f"{res['git_sha']}{'-cpu' if a.cpu_synthetic else ''}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=2, sort_keys=True))
        print(f"wrote {path}")
        return 0

    baseline = json.loads(a.baseline.read_text())
    current = (
        json.loads(a.current.read_text()) if a.current else _run(cfg, a.cpu_synthetic, a.data_root)
    )
    table, problems = compare(baseline, current, tolerances(cfg, _parse_tol(a.tol)))
    print(table)
    if a.table_out:
        a.table_out.parent.mkdir(parents=True, exist_ok=True)
        a.table_out.write_text(table)
    if problems:
        print("REGRESSIONS:\n" + "\n".join(f"- {p}" for p in problems), file=sys.stderr)
        return 1
    return 0


def _run(cfg: dict[str, Any], cpu: bool, data_root: Path) -> dict[str, Any]:
    if cpu:
        return build_result(run_synthetic(cfg), "cpu-synthetic")
    return build_result(run_real(cfg, data_root), "gpu")


def _freeze(config: Path, data_root: Path) -> int:
    cfg = load_config(config)
    for s in cfg.get("scenes") or []:
        s["sha256"] = scene_hash(data_root / s["dir"])
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"pinned {len(cfg.get('scenes') or [])} scenes in {config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
