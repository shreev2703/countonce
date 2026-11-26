"""Shadow-mode model routing (PLAN 16B.2, shadow deployments and model feature flags).

A YAML ``model_routing`` table names the live detector/backbone/associator and optional
shadow variants, each with a traffic percentage. A job id is hashed into a bucket, so the
same job always gets the same variant. ``run_with_shadow`` returns the live result and then
runs the selected shadow variant; the two share the stage caches (``Config.cache_dir``) and,
for injected components, the live detector/reconstructor whenever those stages match.
A shadow failure is logged and never reaches the caller. Backend only: no DB.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

from countonce.config import BackboneName, Config, DetectorName
from countonce.types import CountResult

log = logging.getLogger(__name__)

STAGES = ("detector", "backbone", "assoc")
_BUCKETS = 10_000  # percent resolution of 0.01


class LiveRoute(BaseModel):
    detector: DetectorName = "gdino"
    backbone: BackboneName = "vggt"
    assoc: str = Config.model_fields["assoc"].default


class ShadowRoute(BaseModel):
    """A shadow variant: unset stages inherit the live route."""

    name: str
    percent: float = Field(ge=0.0, le=100.0)
    detector: DetectorName | None = None
    backbone: BackboneName | None = None
    assoc: str | None = None
    v3_checkpoint: Path | None = None


class RoutingTable(BaseModel):
    live: LiveRoute = Field(default_factory=LiveRoute)
    shadows: list[ShadowRoute] = Field(default_factory=list)
    log_path: Path = Path("shadow_log.jsonl")

    @classmethod
    def from_yaml(cls, path: Path | str) -> RoutingTable:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        body = raw.get("model_routing", raw)
        if not isinstance(body, dict):
            raise ValueError(f"{path}: model_routing must be a mapping")
        table = cls(**body)
        names = [s.name for s in table.shadows]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate shadow names")
        return table

    def live_overrides(self) -> dict[str, Any]:
        return self.live.model_dump()

    def shadow_overrides(self, shadow: ShadowRoute) -> dict[str, Any]:
        out = self.live.model_dump()
        for stage in STAGES:
            v = getattr(shadow, stage)
            if v is not None:
                out[stage] = v
        if shadow.v3_checkpoint is not None:
            out["v3_checkpoint"] = shadow.v3_checkpoint
        return out

    def select_shadow(self, job_id: str) -> ShadowRoute | None:
        """First shadow whose percentage covers the job's hash bucket (deterministic)."""
        for shadow in self.shadows:
            if bucket(job_id, shadow.name) < shadow.percent:
                return shadow
        return None


def bucket(job_id: str, salt: str = "") -> float:
    """Stable bucket in [0, 100) from the job id; salted so shadows get independent samples."""
    h = hashlib.sha256(f"{salt}\0{job_id}".encode()).digest()
    return (int.from_bytes(h[:8], "big") % _BUCKETS) / (_BUCKETS / 100)


Runner = Callable[..., CountResult]


def _default_job_id(inputs: list[Path] | Path, prompt: str) -> str:
    paths = inputs if isinstance(inputs, list) else [inputs]
    return hashlib.sha256("\n".join([prompt, *map(str, paths)]).encode()).hexdigest()[:16]


def _summary(res: CountResult, wall: float) -> dict[str, Any]:
    return {
        "count": res.count,
        "interval": [res.interval.low, res.interval.high],
        "timings": {**res.timings, "wall": wall},
    }


def append_log(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def run_with_shadow(
    inputs: list[Path] | Path,
    prompt: str,
    routing: RoutingTable,
    *,
    config: Config | None = None,
    job_id: str | None = None,
    runner: Runner | None = None,
    detector: Any = None,
    reconstructor: Any = None,
    shadow_detector: Any = None,
    shadow_reconstructor: Any = None,
) -> CountResult:
    """Run the live route and return its result; then run the selected shadow, if any.

    ``detector``/``reconstructor`` are the live overrides (as in ``pipeline.run``). The
    shadow reuses them when its detector/backbone matches live; otherwise it takes the
    ``shadow_*`` overrides or builds its own. Live errors propagate; shadow errors do not.
    """
    if runner is None:
        from countonce.pipeline import run as default_runner

        runner = default_runner
    base = config if config is not None else Config()
    jid = job_id if job_id is not None else _default_job_id(inputs, prompt)
    live_cfg = base.with_overrides(**routing.live_overrides())

    kw: dict[str, Any] = {}
    if detector is not None:
        kw["detector"] = detector
    if reconstructor is not None:
        kw["reconstructor"] = reconstructor
    t0 = time.perf_counter()
    live = runner(inputs, prompt, live_cfg, **kw)
    live_summary = _summary(live, time.perf_counter() - t0)  # snapshot before the shadow runs

    try:
        shadow = routing.select_shadow(jid)
        if shadow is not None:
            _run_shadow(
                inputs, prompt, routing, shadow, base, jid, live_summary, runner,
                live_cfg=live_cfg,
                detector=detector, reconstructor=reconstructor,
                shadow_detector=shadow_detector, shadow_reconstructor=shadow_reconstructor,
            )  # fmt: skip
    except Exception:  # noqa: BLE001 - shadow machinery must never affect the live result
        log.exception("shadow bookkeeping failed for job %s", jid)
    return live


def _run_shadow(
    inputs: list[Path] | Path,
    prompt: str,
    routing: RoutingTable,
    shadow: ShadowRoute,
    base: Config,
    jid: str,
    live_summary: dict[str, Any],
    runner: Runner,
    *,
    live_cfg: Config,
    detector: Any,
    reconstructor: Any,
    shadow_detector: Any,
    shadow_reconstructor: Any,
) -> None:
    over = routing.shadow_overrides(shadow)
    # separate output dir so the shadow never overwrites live artifacts; caches stay shared
    over["out_dir"] = Path(base.out_dir) / "shadow" / shadow.name
    cfg = base.with_overrides(**over)
    kw: dict[str, Any] = {"export": False}
    same_det = cfg.detector == live_cfg.detector
    same_bb = cfg.backbone == live_cfg.backbone
    det = detector if same_det else shadow_detector
    rec = reconstructor if same_bb else shadow_reconstructor
    if det is not None:
        kw["detector"] = det
    if rec is not None:
        kw["reconstructor"] = rec

    record: dict[str, Any] = {
        "job": jid,
        "live": routing.live_overrides(),
        "shadow": {"name": shadow.name, "percent": shadow.percent} | {s: over[s] for s in STAGES},
        "shared_stages": [s for s, same in (("detector", same_det), ("backbone", same_bb)) if same],
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        res = runner(inputs, prompt, cfg, **kw)
        sh = _summary(res, time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - shadow failures are recorded, never raised
        log.warning("shadow %s failed for job %s: %s", shadow.name, jid, exc)
        sh = None
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["counts"] = {"live": live_summary["count"], "shadow": sh["count"] if sh else None}
    record["intervals"] = {
        "live": live_summary["interval"],
        "shadow": sh["interval"] if sh else None,
    }
    record["timings"] = {"live": live_summary["timings"], "shadow": sh["timings"] if sh else None}
    append_log(routing.log_path, record)


# ---------------------------------------------------------------- report


def load_log(path: Path | str) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i}: invalid JSON ({exc.msg})") from exc
    return out


def _boot_ci(
    x: Sequence[float], n_boot: int, seed: int, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over jobs; NaNs when there is no data."""
    if not len(x):
        return (float("nan"),) * 3
    a = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(a.mean()), float(lo), float(hi)


def _overlap(a: Sequence[int], b: Sequence[int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def summarize(
    records: list[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Agreement of shadow vs live, per shadow name, with bootstrap CIs over jobs."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_name.setdefault(r["shadow"]["name"], []).append(r)
    report: dict[str, Any] = {}
    for name, rs in sorted(by_name.items()):
        ok = [r for r in rs if r.get("error") is None and r["counts"]["shadow"] is not None]
        d = [abs(r["counts"]["shadow"] - r["counts"]["live"]) for r in ok]
        ov = [float(_overlap(r["intervals"]["live"], r["intervals"]["shadow"])) for r in ok]
        lat = [
            r["timings"]["shadow"]["wall"] - r["timings"]["live"]["wall"]
            for r in ok
            if "wall" in r["timings"]["shadow"] and "wall" in r["timings"]["live"]
        ]
        entry: dict[str, Any] = {"n": len(rs), "n_ok": len(ok), "n_failed": len(rs) - len(ok)}
        for key, vals in (
            ("mean_abs_delta_count", d),
            ("interval_overlap", ov),
            ("latency_delta_s", lat),
        ):
            m, lo, hi = _boot_ci(vals, n_boot, seed)
            entry[key] = {"mean": m, "ci95": [lo, hi]}
        report[name] = entry
    return report


def format_report(report: dict[str, Any]) -> str:
    if not report:
        return "shadow-report: no shadow records"
    lines = []
    for name, e in report.items():
        lines.append(f"shadow {name}: {e['n_ok']}/{e['n']} ok, {e['n_failed']} failed")
        for key, label in (
            ("mean_abs_delta_count", "mean |d count|"),
            ("interval_overlap", "interval overlap"),
            ("latency_delta_s", "latency delta (s)"),
        ):
            v = e[key]
            lines.append(f"  {label}: {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    return "\n".join(lines)
