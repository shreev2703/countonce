"""Cold-start harness: process start -> import -> model loads -> first warm inference -> p50/p95.

Loaders and the warm call are injected so CPU tests run with fakes. ``load_weights`` is the
default file loader: ``.safetensors`` files are opened with ``safe_open`` (memory-mapped, lazy),
anything else is read as bytes. Rows are appended to ``results/coldstart.csv`` with a fixed schema.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = ("run_id", "timestamp", "worker_kind", "phase", "name", "seconds", "n", "note")
PHASES = ("process_start", "import", "model_load", "first_inference", "warm_p50", "warm_p95")


@dataclass(frozen=True)
class Row:
    run_id: str
    timestamp: str
    worker_kind: str
    phase: str
    name: str
    seconds: float
    n: int = 1
    note: str = ""


def percentile(values: Sequence[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        raise ValueError("no values")
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def load_weights(path: str | Path) -> Any:
    """Load one weights file; ``.safetensors`` is memory-mapped (``safe_open``), not read."""
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors import safe_open  # type: ignore[import-not-found,unused-ignore]

        return safe_open(str(p), framework="np")  # mmap-backed, tensors materialise lazily
    return p.read_bytes()


def run_coldstart(
    loaders: Mapping[str, Callable[[], Any]],
    warm: Callable[[Mapping[str, Any]], Any],
    *,
    importer: Callable[[], Any] | None = None,
    process_start: float | None = None,
    warm_iters: int = 20,
    worker_kind: str = "cloud",
    run_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
    wall: Callable[[], float] = time.time,
) -> list[Row]:
    """Time each cold-start phase and return rows (never invented: every value is measured)."""
    if warm_iters < 1:
        raise ValueError("warm_iters must be >= 1")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(wall()))
    rid = run_id or f"cs-{int(wall())}"
    rows: list[Row] = []

    def add(phase: str, name: str, seconds: float, n: int = 1, note: str = "") -> None:
        rows.append(Row(rid, stamp, worker_kind, phase, name, seconds, n, note))

    if process_start is not None:
        add("process_start", "interpreter", max(0.0, wall() - process_start))
    t = clock()
    if importer is not None:
        importer()
    add("import", "pipeline", clock() - t)

    models: dict[str, Any] = {}
    for name, loader in loaders.items():
        t = clock()
        models[name] = loader()
        add("model_load", name, clock() - t)

    t = clock()
    warm(models)
    add("first_inference", "warm", clock() - t)

    lat: list[float] = []
    for _ in range(warm_iters):
        t = clock()
        warm(models)
        lat.append(clock() - t)
    add("warm_p50", "warm", percentile(lat, 50), n=warm_iters)
    add("warm_p95", "warm", percentile(lat, 95), n=warm_iters)
    return rows


def append_rows(rows: Sequence[Row], path: str | Path) -> Path:
    """Append-only CSV; the header is written once and the schema is fixed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(SCHEMA)
        for r in rows:
            w.writerow(
                [
                    r.run_id,
                    r.timestamp,
                    r.worker_kind,
                    r.phase,
                    r.name,
                    f"{r.seconds:.6f}",
                    r.n,
                    r.note,
                ]
            )
    return p


def weight_loaders(paths: Mapping[str, str | Path]) -> dict[str, Callable[[], Any]]:
    return {name: (lambda p=p: load_weights(p)) for name, p in paths.items()}  # type: ignore[misc]


def touch_warm(models: Mapping[str, Any]) -> int:
    """Default warm call: touch every loaded handle (first tensor of a safetensors file)."""
    total = 0
    for m in models.values():
        if hasattr(m, "keys"):
            keys = list(m.keys())
            if keys:
                total += int(m.get_tensor(keys[0]).size)
        else:
            total += len(m)
    return total
