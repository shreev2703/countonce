"""Serialize a ``CountResult`` to the versioned wire formats."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from countonce.schemas import CURRENT_VERSION, MarkersV1, ResultV1
from countonce.types import CountResult, Detection


def to_result(result: CountResult) -> ResultV1:
    data = result.to_dict()
    data["schema_version"] = CURRENT_VERSION
    data["per_frame_counts"] = {str(k): v for k, v in result.per_frame_counts.items()}
    return ResultV1.model_validate(data)


def to_markers(result: CountResult, scene_scale: float | None = None) -> MarkersV1:
    markers = [
        {
            "id": i.id,
            "xyz": [float(v) for v in i.xyz],
            "size": float(i.size),
            "conf": i.conf,
            "n_views": i.n_views,
            "members": list(i.members),
            "possibly_moving": i.possibly_moving,
        }
        for i in result.instances
    ]
    return MarkersV1.model_validate(
        {
            "schema_version": CURRENT_VERSION,
            "count": result.count,
            "interval_low": result.interval.low,
            "interval_high": result.interval.high,
            "scene_scale": scene_scale,
            "markers": markers,
        }
    )


def _write_json(path: Path, model: ResultV1 | MarkersV1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return path


def write_result_json(result: CountResult, path: Path) -> Path:
    return _write_json(Path(path), to_result(result))


def write_markers_json(result: CountResult, path: Path, scene_scale: float | None = None) -> Path:
    return _write_json(Path(path), to_markers(result, scene_scale))


def write_counts_csv(result: CountResult, path: Path) -> Path:
    """One row per frame: detections by status, plus a final ``total`` row.

    The ``new`` column sums to the final count (each instance is counted in
    exactly one frame: the one holding its ``new`` detection).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_frame: dict[int, Counter[str]] = {fid: Counter() for fid in result.per_frame_counts}
    for det_id, status in result.detection_status.items():
        per_frame.setdefault(int(det_id.split(":")[0]), Counter())[status] += 1
    cols = ("new", "duplicate", "rejected")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "detections", *cols])
        totals: Counter[str] = Counter()
        for fid in sorted(per_frame):
            c = per_frame[fid]
            totals.update(c)
            w.writerow([fid, sum(c.values()), *(c[s] for s in cols)])
        w.writerow(["total", sum(totals.values()), *(totals[s] for s in cols)])
    return path


def export_all(
    result: CountResult,
    out_dir: Path,
    frames: Sequence | None = None,
    detections: Sequence[Detection] | None = None,
    scene_scale: float | None = None,
) -> dict[str, Path | list[Path]]:
    """Write every artifact into ``out_dir``; overlays only if frames+detections given."""
    out = Path(out_dir)
    paths: dict[str, Path | list[Path]] = {
        "result": write_result_json(result, out / "result.json"),
        "markers": write_markers_json(result, out / "markers.json", scene_scale),
        "counts": write_counts_csv(result, out / "counts.csv"),
    }
    if frames is not None and detections is not None:
        from countonce.export.overlay import render_overlays

        paths["overlays"] = render_overlays(frames, detections, result, out / "overlays")
    return paths
