"""Human baseline study harness (PLAN §9.3 item 10). The study itself is human-run.

Three steps::

    python -m bench.human_baseline sheets   --manifest scenes.json --out study/
    python -m bench.human_baseline score    --manifest scenes.json --responses r.csv \
        --bench-csv results/bench.csv --method countonce-v2
    python -m bench.human_baseline figure   (same inputs plus --out-fig accuracy_vs_time.png)

Manifest (``scenes.json``): ``{"scenes": {"s1": {"images": ["a.jpg", ...], "count": 12}}}``.
``count`` (ground truth) is only used for scoring and is never written to a sheet.
Response CSV columns: ``participant_id, scene_id, count, seconds[, notes]``.
Participants see photos only (no 3D view) under a fixed timing protocol (see ``PROTOCOL``).
Reported numbers are produced at run time; nothing here hard-codes a result.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bench import metrics, stats

PROTOCOL = (
    "Work alone, without a phone or counting software. Use only the photos on the sheet.",
    "Read the instructions, then start the timer when you turn to the first scene.",
    "For each scene, count how many distinct physical objects of the target kind appear "
    "across ALL photos. The same object seen in several photos counts once.",
    "Stop the timer when you write down your final count and record the elapsed seconds.",
    "Do not go back to earlier scenes and do not revise a recorded count.",
    "If you are unsure, give your best single number; note any doubts in the Notes column.",
)


class StudyError(ValueError):
    """Malformed manifest, response file, or scoring inputs; the message says why."""


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images: list[str] = Field(min_length=1)
    count: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenes: dict[str, Scene] = Field(min_length=1)
    target: str = Field(default="objects", min_length=1, description="what to count")


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1)
    scene_id: str = Field(min_length=1)
    count: float = Field(ge=0, allow_inf_nan=False)
    seconds: float = Field(gt=0, allow_inf_nan=False)
    notes: str = ""


def load_manifest(path: str | Path) -> Manifest:
    try:
        return Manifest.model_validate(json.loads(Path(path).read_text()))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        raise StudyError(f"bad manifest {path}: {e}") from e


# --- (a) printable sheets -------------------------------------------------------------------

_CSS = (
    "body{font-family:sans-serif;margin:1.5cm}h1,h2{margin:.2em 0}"
    ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}"
    ".grid img{width:100%;border:1px solid #888}"
    ".box{border:2px solid #000;padding:8px;margin-top:12px;display:flex;gap:2em}"
    ".box span{display:inline-block;min-width:6em;border-bottom:1px solid #000}"
    "@media print{.scene{page-break-after:always}}"
)


def _page(title: str, body: str) -> str:
    t = html.escape(title)
    return (
        f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><title>{t}</title>'
        f"<style>{_CSS}</style></head><body>\n{body}\n</body></html>\n"
    )


def instructions_html(manifest: Manifest) -> str:
    steps = "".join(f"<li>{html.escape(s)}</li>" for s in PROTOCOL)
    target = html.escape(manifest.target)
    body = (
        f"<h1>Counting study: instructions</h1><p>Target to count: <b>{target}</b>. "
        f"There are {len(manifest.scenes)} scenes.</p><ol>{steps}</ol>"
        "<p>Participant ID: <span style='border-bottom:1px solid #000;padding:0 6em'></span></p>"
    )
    return _page("Instructions", body)


def scene_sheet_html(scene_id: str, scene: Scene, target: str = "objects") -> str:
    imgs = "".join(
        f'<img src="{html.escape(p, quote=True)}" alt="photo {i + 1}">'
        for i, p in enumerate(scene.images)
    )
    body = (
        f'<div class="scene"><h2>Scene {html.escape(scene_id)}</h2>'
        f"<p>Count the distinct {html.escape(target)} across all {len(scene.images)} photos "
        f'(same object in several photos counts once).</p><div class="grid">{imgs}</div>'
        '<div class="box"><div>Count: <span></span></div>'
        "<div>Seconds: <span></span></div><div>Notes: <span style='min-width:14em'></span></div>"
        "</div></div>"
    )
    return _page(f"Scene {scene_id}", body)


def write_sheets(manifest: Manifest, out_dir: str | Path) -> list[Path]:
    """Write ``instructions.html`` and one ``scene_<id>.html`` per scene (static, no scripts)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "instructions.html"]
    paths[0].write_text(instructions_html(manifest))
    for sid, sc in manifest.scenes.items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)
        p = out / f"scene_{safe}.html"
        p.write_text(scene_sheet_html(sid, sc, manifest.target))
        paths.append(p)
    return paths


# --- (b) responses ----------------------------------------------------------------------------

_REQUIRED = ("participant_id", "scene_id", "count", "seconds")


def load_responses(path: str | Path, known_scenes: Sequence[str]) -> list[Response]:
    """Parse and validate the response CSV. Unknown scenes and duplicate
    (participant, scene) rows raise :class:`StudyError` naming the CSV line."""
    known = set(known_scenes)
    out: list[Response] = []
    seen: dict[tuple[str, str], int] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise StudyError(f"{path}: missing column(s) {missing}")
        for line, row in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in row.items() if k in (*_REQUIRED, "notes")}
            try:
                r = Response.model_validate(row)
            except ValueError as e:
                raise StudyError(f"{path} line {line}: {e}") from e
            if r.scene_id not in known:
                raise StudyError(f"{path} line {line}: unknown scene {r.scene_id!r}")
            key = (r.participant_id, r.scene_id)
            if key in seen:
                raise StudyError(
                    f"{path} line {line}: duplicate row for participant {key[0]!r}, scene "
                    f"{key[1]!r} (first seen on line {seen[key]})"
                )
            seen[key] = line
            out.append(r)
    if not out:
        raise StudyError(f"{path}: no responses")
    return out


# --- (c) scoring ----------------------------------------------------------------------------


def _gt(manifest: Manifest) -> dict[str, float]:
    gt = {s: sc.count for s, sc in manifest.scenes.items() if sc.count is not None}
    if not gt:
        raise StudyError("manifest has no ground-truth counts; cannot score")
    return gt


def _summ(ci: stats.CI) -> dict[str, float]:
    return {"estimate": ci.estimate, "low": ci.low, "high": ci.high, "n": ci.n}


def _score(
    scenes: Sequence[str],
    pred: Mapping[str, float],
    gt: Mapping[str, float],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    p = [pred[s] for s in scenes]
    g = [gt[s] for s in scenes]
    pct = 100.0 * metrics.pct_errors(p, g)
    return {
        "n_scenes": len(scenes),
        "mae": _summ(stats.bootstrap_ci(metrics.abs_errors(p, g), n_boot=n_boot, seed=seed)),
        "mape": _summ(stats.bootstrap_ci(pct, n_boot=n_boot, seed=seed))
        if not np.isnan(pct).all()
        else None,
        "within_10pct": _summ(
            stats.bootstrap_ci(
                metrics.within_tolerance(p, g, 0.10).astype(float), n_boot=n_boot, seed=seed
            )
        ),
    }


def _per_scene_mean(resp: Sequence[Response], attr: str) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in resp:
        acc[r.scene_id].append(getattr(r, attr))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def method_scenes(
    bench_csv: str | Path, method: str, n_views: int | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-scene (prediction, runtime seconds) for one method from ``results/bench.csv``;
    rows of a scene (samples) are averaged."""
    from bench.figures import load_rows  # lazy: pulls in matplotlib styling

    rows = [
        r
        for r in load_rows(bench_csv)
        if r.method == method and (n_views is None or r.n_views == n_views)
    ]
    if not rows:
        raise StudyError(f"no ok rows for method {method!r} in {bench_csv}")
    preds: dict[str, list[float]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        preds[r.scene].append(r.pred)
        if r.runtime_s is not None:
            times[r.scene].append(r.runtime_s)
    return (
        {s: float(np.mean(v)) for s, v in preds.items()},
        {s: float(np.mean(v)) for s, v in times.items()},
    )


def ratio_statistic(human_s: np.ndarray, method_s: np.ndarray) -> Callable[[np.ndarray], float]:
    """Statistic over resampled scene *indices*: mean human seconds / CountOnce p50 seconds."""

    def stat(idx: np.ndarray) -> float:
        i = idx.astype(int)
        return float(np.mean(human_s[i]) / np.median(method_s[i]))

    return stat


def score_study(
    manifest: Manifest,
    responses: Sequence[Response],
    method_pred: Mapping[str, float] | None = None,
    method_seconds: Mapping[str, float] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    gt = _gt(manifest)
    scored = [r for r in responses if r.scene_id in gt]
    if not scored:
        raise StudyError("no responses on scenes that have ground truth")
    by_p: dict[str, list[Response]] = defaultdict(list)
    for r in scored:
        by_p[r.participant_id].append(r)

    out: dict[str, Any] = {"participants": {}, "n_participants": len(by_p)}
    for pid, rs in sorted(by_p.items()):
        pred = {r.scene_id: r.count for r in rs}
        res = _score(sorted(pred), pred, gt, n_boot, seed)
        res["mean_seconds_per_scene"] = float(np.mean([r.seconds for r in rs]))
        out["participants"][pid] = res

    # Pooled: scene-level mean over participants, scenes are the resampling unit.
    pooled_pred = _per_scene_mean(scored, "count")
    pooled_sec = _per_scene_mean(scored, "seconds")
    scenes = sorted(pooled_pred)
    pooled = _score(scenes, pooled_pred, gt, n_boot, seed)
    # Pooled error metrics average per-participant errors (not the error of the mean count).
    per_scene_err: dict[str, list[float]] = defaultdict(list)
    per_scene_ok: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        per_scene_err[r.scene_id].append(abs(r.count - gt[r.scene_id]))
        tol = 0.10 * gt[r.scene_id] + 1e-12
        per_scene_ok[r.scene_id].append(float(abs(r.count - gt[r.scene_id]) <= tol))
    abs_err = np.array([np.mean(per_scene_err[s]) for s in scenes])
    pooled["mae"] = _summ(stats.bootstrap_ci(abs_err, n_boot=n_boot, seed=seed))
    pooled["within_10pct"] = _summ(
        stats.bootstrap_ci(
            [float(np.mean(per_scene_ok[s])) for s in scenes], n_boot=n_boot, seed=seed
        )
    )
    pooled["mean_seconds_per_scene"] = float(np.mean([pooled_sec[s] for s in scenes]))
    out["pooled"] = pooled

    if method_pred is not None:
        common = [s for s in scenes if s in method_pred]
        if len(common) < 2:
            raise StudyError("fewer than 2 scenes shared between humans and the method")
        g = [gt[s] for s in common]
        m_err = metrics.abs_errors([method_pred[s] for s in common], g)
        h_err = np.array([np.mean(per_scene_err[s]) for s in common])
        paired = stats.paired_compare(h_err, m_err, n_boot=n_boot, seed=seed)
        out["vs_method"] = {
            "note": "diff = human abs err - method abs err; diff > 0 means the method is better",
            "method_mae": _summ(stats.bootstrap_ci(m_err, n_boot=n_boot, seed=seed)),
            "scenes": common,
            **paired.__dict__,
        }
        if method_seconds:
            tcommon = [s for s in common if s in method_seconds]
            if len(tcommon) >= 2:
                h = np.array([pooled_sec[s] for s in tcommon])
                m = np.array([method_seconds[s] for s in tcommon])
                ci = stats.bootstrap_ci(
                    np.arange(len(tcommon), dtype=float),
                    statistic=ratio_statistic(h, m),
                    n_boot=n_boot,
                    seed=seed,
                )
                out["time_ratio"] = {
                    "definition": "mean human seconds / CountOnce p50 seconds",
                    "method_p50_seconds": float(np.median(m)),
                    **_summ(ci),
                }
    return out


# --- (d) figure ---------------------------------------------------------------------------------


def fig_accuracy_vs_time(
    result: Mapping[str, Any],
    out: Path,
    methods: Mapping[str, tuple[float, float]] | None = None,
) -> Path:
    """Scatter: x = seconds per scene, y = MAPE (%); one point per participant and one per
    method (``methods`` maps name -> (seconds per scene, MAPE %)). No values are annotated."""
    from countonce.plotting import apply_style, color_for, plt, save

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in result["participants"].values():
        if r["mape"] is None:
            continue
        ax.scatter(r["mean_seconds_per_scene"], r["mape"]["estimate"], color="#888888", s=36)
    ax.scatter([], [], color="#888888", label="human participant")
    for i, (name, (sec, mape)) in enumerate((methods or {}).items()):
        ax.scatter(sec, mape, marker="D", s=60, color=color_for(name, i), label=name)
    ax.set_xscale("log")
    ax.set_xlabel("seconds per scene (log)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Accuracy vs. time: TODO: fill from results")
    ax.legend(fontsize=8)
    save(fig, out)
    return out


def method_point(
    bench_csv: str | Path, method: str, gt: Mapping[str, float], n_views: int | None = None
) -> tuple[float, float]:
    """(mean seconds per scene, MAPE %) of one method over the scenes with GT."""
    pred, secs = method_scenes(bench_csv, method, n_views)
    common = sorted(s for s in pred if s in gt and s in secs)
    if not common:
        raise StudyError(f"method {method!r} has no timed scenes with ground truth")
    return (
        float(np.mean([secs[s] for s in common])),
        metrics.mape([pred[s] for s in common], [gt[s] for s in common]),
    )


# --- CLI --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--out", type=Path, default=Path("study"))
    for name in ("score", "figure"):
        q = sub.add_parser(name)
        q.add_argument("--manifest", type=Path, required=True)
        q.add_argument("--responses", type=Path, required=True)
        q.add_argument("--bench-csv", type=Path, default=Path("results/bench.csv"))
        q.add_argument("--method", default=None, help="CountOnce method name in bench.csv")
        q.add_argument("--n-views", type=int, default=None)
        q.add_argument("--n-boot", type=int, default=10_000)
        if name == "figure":
            q.add_argument(
                "--out-fig", type=Path, default=Path("results/figures/human_vs_method.png")
            )
    a = ap.parse_args(argv)
    try:
        man = load_manifest(a.manifest)
        if a.cmd == "sheets":
            for p in write_sheets(man, a.out):
                print(p)
            return 0
        resp = load_responses(a.responses, list(man.scenes))
        pred = secs = None
        if a.method:
            pred, secs = method_scenes(a.bench_csv, a.method, a.n_views)
        res = score_study(man, resp, pred, secs, n_boot=a.n_boot)
        if a.cmd == "score":
            print(json.dumps(res, indent=2, default=float))
            return 0
        pts = (
            {a.method: method_point(a.bench_csv, a.method, _gt(man), a.n_views)} if a.method else {}
        )
        print(fig_accuracy_vs_time(res, a.out_fig, pts))
        return 0
    except StudyError as e:
        print(f"error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
