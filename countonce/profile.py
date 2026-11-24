"""Per-stage wall time and peak VRAM profiling (PLAN 13 item 1).

``StageProfiler`` wraps stages with a timer and reads ``torch.cuda`` peak-memory stats only when
torch is already loaded and CUDA is available; otherwise VRAM is None. ``run_profile`` runs the
real pipeline on a configured input several times and writes one CSV row per (repeat, stage).
Nothing here is a result: numbers only exist after a real run.
"""

from __future__ import annotations

import csv
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from countonce.config import Config

MB = 2**20


def _cuda() -> Any | None:
    """``torch.cuda`` if torch is loaded and a GPU is usable, else None (never imports torch)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda


@dataclass
class StageStat:
    stage: str
    wall_s: float
    peak_vram_mb: float | None


class StageProfiler:
    def __init__(self) -> None:
        self.stats: list[StageStat] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        cuda = _cuda()
        if cuda is not None:
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            peak: float | None = None
            if cuda is not None:
                cuda.synchronize()
                peak = float(cuda.max_memory_allocated() / MB)
            self.stats.append(StageStat(name, time.perf_counter() - t0, peak))


@dataclass
class ProfileRow:
    repeat: int
    stage: str
    wall_s: float
    peak_vram_mb: float | None
    n_frames: int
    backbone: str
    detector: str
    assoc: str


class ProfileConfig:
    """YAML: ``inputs`` (dir/files/video), ``prompt``, ``repeats``, ``warmup``, ``out_csv``,
    and an optional nested ``run:`` block of ``Config`` fields."""

    def __init__(
        self,
        inputs: list[Path],
        prompt: str,
        repeats: int = 3,
        warmup: int = 1,
        out_csv: Path = Path("results/profile.csv"),
        run: Config | None = None,
    ) -> None:
        if repeats < 1 or warmup < 0:
            raise ValueError("repeats must be >= 1 and warmup >= 0")
        self.inputs, self.prompt = inputs, prompt
        self.repeats, self.warmup, self.out_csv = repeats, warmup, Path(out_csv)
        self.run = run if run is not None else Config()

    @classmethod
    def from_yaml(cls, path: Path | str) -> ProfileConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not raw.get("inputs") or not raw.get("prompt"):
            raise ValueError(f"{path}: `inputs` and `prompt` are required")
        ins = raw["inputs"]
        inputs = [Path(p) for p in ([ins] if isinstance(ins, str) else ins)]
        return cls(
            inputs=inputs,
            prompt=str(raw["prompt"]),
            repeats=int(raw.get("repeats", 3)),
            warmup=int(raw.get("warmup", 1)),
            out_csv=Path(raw.get("out_csv", "results/profile.csv")),
            run=Config(**(raw.get("run") or {})),
        )


COLUMNS = [f.name for f in fields(ProfileRow)]


def write_csv(rows: list[ProfileRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(
                [
                    r.repeat,
                    r.stage,
                    f"{r.wall_s:.6f}",
                    "" if r.peak_vram_mb is None else f"{r.peak_vram_mb:.1f}",
                    r.n_frames,
                    r.backbone,
                    r.detector,
                    r.assoc,
                ]
            )
    return path


def run_profile(cfg: ProfileConfig) -> tuple[Path, list[ProfileRow]]:
    """Run the pipeline ``warmup + repeats`` times; record per-stage wall time and peak VRAM.

    Stage times come from the pipeline's own ``timings`` (ingest, detect, recon, associate,
    confidence, total). Peak VRAM is the run-level peak (``CountResult.peak_vram_mb``) on the
    ``recon`` row and the whole-run torch peak on ``total``; None without CUDA.
    """
    from countonce.pipeline import run

    inputs: Any = cfg.inputs[0] if len(cfg.inputs) == 1 else cfg.inputs
    rows: list[ProfileRow] = []
    for i in range(cfg.warmup + cfg.repeats):
        prof = StageProfiler()
        with prof.stage("total"):
            result = run(inputs, cfg.prompt, cfg.run, export=False)
        total_peak = prof.stats[0].peak_vram_mb
        if i < cfg.warmup:
            continue
        has_cuda = total_peak is not None
        for stage, secs in result.timings.items():
            peak: float | None = None
            if has_cuda and stage == "recon":
                peak = result.peak_vram_mb or None
            elif has_cuda and stage == "total":
                peak = total_peak
            rows.append(
                ProfileRow(
                    i - cfg.warmup,
                    stage,
                    float(secs),
                    peak,
                    len(result.per_frame_counts),
                    result.model_versions.backbone,
                    result.model_versions.detector,
                    cfg.run.assoc,
                )
            )
    return write_csv(rows, cfg.out_csv), rows
