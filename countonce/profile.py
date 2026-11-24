"""Per-stage wall time and peak VRAM profiling (PLAN 13 item 1).

``StageProfiler`` wraps stages with a timer and reads ``torch.cuda`` peak-memory stats only when
torch is already loaded and CUDA is available; otherwise VRAM is None. ``run_profile`` runs the
real pipeline on a configured input several times and writes one CSV row per (repeat, stage).
Nothing here is a result: numbers only exist after a real run.
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from countonce.config import Config
from countonce.types import CountResult

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
    # Append-only (repeat-job caching, PLAN 13 item 6b). Empty on ordinary profile rows.
    cache_state: str = ""
    recon_cache_hit: bool | None = None
    det_cache_hit: bool | None = None


@dataclass
class Variant:
    """A named set of ``Config`` overrides on top of the base ``run:`` block."""

    name: str
    run: Config

    @classmethod
    def from_raw(cls, raw: dict[str, Any], base_run: dict[str, Any]) -> Variant:
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ValueError("each variant needs a `name`")
        return cls(str(raw["name"]), Config(**{**base_run, **(raw.get("run") or {})}))


@dataclass
class CacheProfile:
    """The ``cache:`` block: ``new_prompt`` (second prompt) and a scratch ``dir`` for the caches."""

    new_prompt: str
    dir: Path = Path("results/profile_cache")

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> CacheProfile:
        if not isinstance(raw, dict) or not raw.get("new_prompt"):
            raise ValueError("`cache:` needs a `new_prompt`")
        return cls(str(raw["new_prompt"]), Path(raw.get("dir", "results/profile_cache")))


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
        variants: list[Variant] | None = None,
        variants_csv: Path = Path("results/profile_variants.csv"),
        cache: CacheProfile | None = None,
    ) -> None:
        if repeats < 1 or warmup < 0:
            raise ValueError("repeats must be >= 1 and warmup >= 0")
        self.inputs, self.prompt = inputs, prompt
        self.repeats, self.warmup, self.out_csv = repeats, warmup, Path(out_csv)
        self.run = run if run is not None else Config()
        self.variants = variants or []
        self.variants_csv = Path(variants_csv)
        self.cache = cache
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variant names: {names}")

    @classmethod
    def from_yaml(cls, path: Path | str) -> ProfileConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not raw.get("inputs") or not raw.get("prompt"):
            raise ValueError(f"{path}: `inputs` and `prompt` are required")
        ins = raw["inputs"]
        inputs = [Path(p) for p in ([ins] if isinstance(ins, str) else ins)]
        base_run = raw.get("run") or {}
        return cls(
            inputs=inputs,
            prompt=str(raw["prompt"]),
            repeats=int(raw.get("repeats", 3)),
            warmup=int(raw.get("warmup", 1)),
            out_csv=Path(raw.get("out_csv", "results/profile.csv")),
            run=Config(**(raw.get("run") or {})),
            variants=[Variant.from_raw(v, base_run) for v in raw.get("variants") or []],
            variants_csv=Path(raw.get("variants_csv", "results/profile_variants.csv")),
            cache=CacheProfile.from_raw(raw["cache"]) if raw.get("cache") else None,
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
                    r.cache_state,
                    _flag(r.recon_cache_hit),
                    _flag(r.det_cache_hit),
                ]
            )
    return path


def _flag(x: bool | None) -> str:
    return "" if x is None else str(int(x))


def load_profile_csv(path: Path) -> list[ProfileRow]:
    """Read a profile CSV; files written before the cache columns existed load with defaults."""
    out: list[ProfileRow] = []
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            peak = r.get("peak_vram_mb") or ""
            rh, dh = r.get("recon_cache_hit") or "", r.get("det_cache_hit") or ""
            out.append(
                ProfileRow(
                    int(r["repeat"]),
                    r["stage"],
                    float(r["wall_s"]),
                    float(peak) if peak else None,
                    int(r["n_frames"]),
                    r["backbone"],
                    r["detector"],
                    r["assoc"],
                    r.get("cache_state") or "",
                    None if rh == "" else rh == "1",
                    None if dh == "" else dh == "1",
                )
            )
    return out


def _measure(
    cfg: ProfileConfig,
    run_cfg: Config,
    run_fn: Callable[[Config], CountResult] | None = None,
) -> list[tuple[int, CountResult, float | None]]:
    """``warmup + repeats`` runs; returns (repeat, result, torch peak MB) per timed run."""
    if run_fn is None:
        from countonce.pipeline import run

        inputs: Any = cfg.inputs[0] if len(cfg.inputs) == 1 else cfg.inputs

        def run_fn(c: Config) -> CountResult:
            return run(inputs, cfg.prompt, c, export=False)

    out: list[tuple[int, CountResult, float | None]] = []
    for i in range(cfg.warmup + cfg.repeats):
        prof = StageProfiler()
        with prof.stage("total"):
            result = run_fn(run_cfg)
        if i >= cfg.warmup:
            out.append((i - cfg.warmup, result, prof.stats[0].peak_vram_mb))
    return out


def _stage_rows(
    repeat: int, result: CountResult, total_peak: float | None, run_cfg: Config
) -> list[ProfileRow]:
    has_cuda = total_peak is not None
    rows = []
    for stage, secs in result.timings.items():
        peak: float | None = None
        if has_cuda and stage == "recon":
            peak = result.peak_vram_mb or None
        elif has_cuda and stage == "total":
            peak = total_peak
        rows.append(
            ProfileRow(
                repeat,
                stage,
                float(secs),
                peak,
                len(result.per_frame_counts),
                result.model_versions.backbone,
                result.model_versions.detector,
                run_cfg.assoc,
            )
        )
    return rows


def run_profile(cfg: ProfileConfig) -> tuple[Path, list[ProfileRow]]:
    """Run the pipeline ``warmup + repeats`` times; record per-stage wall time and peak VRAM.

    Stage times come from the pipeline's own ``timings`` (ingest, detect, recon, associate,
    confidence, total). Peak VRAM is the run-level peak (``CountResult.peak_vram_mb``) on the
    ``recon`` row and the whole-run torch peak on ``total``; None without CUDA.
    """
    rows: list[ProfileRow] = []
    for repeat, result, total_peak in _measure(cfg, cfg.run):
        rows.extend(_stage_rows(repeat, result, total_peak, cfg.run))
    return write_csv(rows, cfg.out_csv), rows


@dataclass
class VariantRow:
    variant: str
    repeat: int
    stage: str
    wall_s: float | None
    peak_vram_mb: float | None
    count: int | None
    count_delta: int | None  # count minus the baseline variant's count at the same repeat
    status: str = "ok"  # "ok" or "error: ..." (a failed variant does not stop the others)


VARIANT_COLUMNS = [f.name for f in fields(VariantRow)]


def write_variants_csv(rows: list[VariantRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(VARIANT_COLUMNS)
        for r in rows:
            w.writerow(
                [
                    r.variant,
                    r.repeat,
                    r.stage,
                    "" if r.wall_s is None else f"{r.wall_s:.6f}",
                    "" if r.peak_vram_mb is None else f"{r.peak_vram_mb:.1f}",
                    "" if r.count is None else r.count,
                    "" if r.count_delta is None else r.count_delta,
                    r.status,
                ]
            )
    return path


def run_profile_variants(
    cfg: ProfileConfig,
    run_fn: Callable[[Config], CountResult] | None = None,
) -> tuple[Path, list[VariantRow]]:
    """Profile each named variant (the first is the baseline) and write ``variants_csv``.

    Per variant: stage wall time and peak VRAM per timed repeat, plus the final count and its
    delta against the baseline's count at the same repeat (accuracy proxy: 0 means unchanged).
    ``run_fn`` replaces the real pipeline (tests use fake stages).
    """
    if not cfg.variants:
        raise ValueError("no `variants:` in the profile config")
    rows: list[VariantRow] = []
    baseline_counts: dict[int, int] = {}
    for vi, var in enumerate(cfg.variants):
        try:
            measured = _measure(cfg, var.run, run_fn)
        except Exception as exc:  # a variant (e.g. a failing compile) must not lose the others
            rows.append(VariantRow(var.name, 0, "total", None, None, None, None, f"error: {exc}"))
            continue
        for repeat, result, total_peak in measured:
            if vi == 0:
                baseline_counts[repeat] = result.count
            base = baseline_counts.get(repeat)
            delta = None if base is None else result.count - base
            for r in _stage_rows(repeat, result, total_peak, var.run):
                rows.append(
                    VariantRow(
                        var.name, repeat, r.stage, r.wall_s, r.peak_vram_mb, result.count, delta
                    )
                )
    return write_variants_csv(rows, cfg.variants_csv), rows


CACHE_STATES = ("cold", "new_prompt", "repeat")


def _n_cache_entries(cache_dir: Path) -> int:
    d = cache_dir / "detect"
    return sum(1 for p in d.rglob("*") if p.is_file()) if d.exists() else 0


def run_profile_cache(
    cfg: ProfileConfig,
    run_fn: Callable[[Config, str], CountResult] | None = None,
) -> tuple[Path, list[ProfileRow]]:
    """Measure repeat-job savings (PLAN 13 item 6b) and write ``out_csv``.

    Per repeat, with an empty cache: (1) ``cold``: the photos and ``prompt``; (2) ``new_prompt``:
    the same photos with ``cache.new_prompt`` (recon hit, detection miss); (3) ``repeat``: the
    first job again (both hit). ``warmup`` untimed passes run first in a separate cache dir. The
    recon hit flag is ``CountResult.recon_cache_hit``; the detection flag is "no new entry was
    written to the detection cache" (a miss always writes). ``run_fn`` replaces the pipeline.
    """
    if cfg.cache is None:
        raise ValueError("no `cache:` block in the profile config")
    if cfg.cache.new_prompt == cfg.prompt:
        raise ValueError("`cache.new_prompt` must differ from `prompt`")
    if run_fn is None:
        from countonce.pipeline import run

        inputs: Any = cfg.inputs[0] if len(cfg.inputs) == 1 else cfg.inputs

        def run_fn(c: Config, prompt: str) -> CountResult:
            return run(inputs, prompt, c, export=False)

    root = cfg.cache.dir
    prompts = {"cold": cfg.prompt, "new_prompt": cfg.cache.new_prompt, "repeat": cfg.prompt}
    rows: list[ProfileRow] = []
    for i in range(cfg.warmup + cfg.repeats):
        timed = i >= cfg.warmup
        cdir = root / ("run" if timed else "warmup")
        shutil.rmtree(cdir, ignore_errors=True)
        run_cfg = cfg.run.model_copy(update={"cache_dir": cdir})
        for state in CACHE_STATES:
            before = _n_cache_entries(cdir)
            prof = StageProfiler()
            with prof.stage("total"):
                result = run_fn(run_cfg, prompts[state])
            det_hit = _n_cache_entries(cdir) == before
            if not timed:
                continue
            for r in _stage_rows(i - cfg.warmup, result, prof.stats[0].peak_vram_mb, run_cfg):
                r.cache_state = state
                r.recon_cache_hit = result.recon_cache_hit
                r.det_cache_hit = det_hit
                rows.append(r)
    shutil.rmtree(root / "warmup", ignore_errors=True)
    return write_csv(rows, cfg.out_csv), rows
