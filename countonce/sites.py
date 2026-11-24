"""Sites and visits (PLAN 14.8, 14.3 ``sites``): group runs by plot and compare counts over time.

A :class:`Site` holds a list of :class:`Visit` entries, each pointing at a finished run output
directory plus a ``captured_at`` time. The index lives in a single ``sites.json``. Counts and
intervals are read from each run's ``result.json`` when a report is built, never copied, so the
report always reflects the run on disk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class SiteError(ValueError):
    """Base class for site/visit failures."""


class SiteNotFoundError(SiteError):
    pass


class DuplicateSiteError(SiteError):
    pass


class RunDirError(SiteError):
    """The run directory (or its ``result.json``) is missing or unreadable."""


class CapturedAtError(SiteError):
    """No usable ``captured_at``: not given and not in the result metadata."""


def parse_time(value: str) -> datetime:
    """Parse ISO 8601; naive values are taken as UTC so ordering is always well defined."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CapturedAtError(f"invalid captured_at {value!r}: expected ISO 8601") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@dataclass(frozen=True)
class Visit:
    run_dir: str
    captured_at: str  # ISO 8601

    @property
    def when(self) -> datetime:
        return parse_time(self.captured_at)


@dataclass
class Site:
    id: str
    name: str
    description: str = ""
    visits: list[Visit] = field(default_factory=list)

    def ordered_visits(self) -> list[Visit]:
        return sorted(self.visits, key=lambda v: (v.when, v.run_dir))


@dataclass(frozen=True)
class VisitRow:
    captured_at: str
    run_dir: str
    count: int
    low: int
    high: int
    level: float
    report_html: str | None


@dataclass(frozen=True)
class Delta:
    from_run: str
    to_run: str
    change: int
    within_noise: bool


def intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _read_result(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "result.json"
    if not run_dir.is_dir() or not path.is_file():
        raise RunDirError(f"run dir {run_dir} has no result.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunDirError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunDirError(f"{path} is not a JSON object")
    return data


def result_captured_at(run_dir: Path) -> str | None:
    cfg = _read_result(run_dir).get("config")
    ts = cfg.get("timestamp_utc") if isinstance(cfg, dict) else None
    return ts if isinstance(ts, str) else None


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise SiteError(f"cannot derive a site id from {name!r}")
    return slug


class SiteIndex:
    """The ``sites.json`` file: load, mutate, save."""

    def __init__(self, path: Path | str, sites: list[Site] | None = None) -> None:
        self.path = Path(path)
        self.sites: list[Site] = sites or []

    @classmethod
    def load(cls, path: Path | str) -> SiteIndex:
        p = Path(path)
        if not p.is_file():
            return cls(p)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            sites = [
                Site(
                    id=s["id"],
                    name=s["name"],
                    description=s.get("description", ""),
                    visits=[Visit(v["run_dir"], v["captured_at"]) for v in s.get("visits", [])],
                )
                for s in raw["sites"]
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SiteError(f"{p} is not a valid sites.json: {exc!r}") from exc
        return cls(p, sites)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sites": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "visits": [
                        {"run_dir": v.run_dir, "captured_at": v.captured_at} for v in s.visits
                    ],
                }
                for s in self.sites
            ],
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def get(self, site_id: str) -> Site:
        for s in self.sites:
            if s.id == site_id:
                return s
        raise SiteNotFoundError(f"no site with id {site_id!r}")

    def add(self, name: str, description: str = "", site_id: str | None = None) -> Site:
        sid = site_id or slugify(name)
        if any(s.id == sid for s in self.sites):
            raise DuplicateSiteError(f"site {sid!r} already exists")
        site = Site(sid, name, description)
        self.sites.append(site)
        return site

    def attach(self, site_id: str, run_dir: Path | str, captured_at: str | None = None) -> Visit:
        """Add a visit. ``captured_at`` defaults to the result's ``timestamp_utc`` metadata."""
        site = self.get(site_id)
        rd = Path(run_dir).resolve()
        ts = captured_at if captured_at is not None else result_captured_at(rd)
        if ts is None:
            raise CapturedAtError(f"{rd}/result.json has no timestamp_utc; pass --captured-at")
        parse_time(ts)  # validate
        _read_result(rd)  # the run must exist even when the time is given
        if any(v.run_dir == str(rd) for v in site.visits):
            raise SiteError(f"{rd} is already attached to {site_id!r}")
        visit = Visit(str(rd), ts)
        site.visits.append(visit)
        return visit


def visit_rows(site: Site) -> list[VisitRow]:
    """Rows ordered by ``captured_at``. Raises :class:`RunDirError` for a missing run dir."""
    rows = []
    for v in site.ordered_visits():
        rd = Path(v.run_dir)
        res = _read_result(rd)
        try:
            iv = res["interval"]
            row = VisitRow(
                v.captured_at,
                v.run_dir,
                int(res["count"]),
                int(iv["low"]),
                int(iv["high"]),
                float(iv["level"]),
                str(rd / "report.html") if (rd / "report.html").is_file() else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunDirError(f"{rd}/result.json lacks count/interval: {exc!r}") from exc
        rows.append(row)
    return rows


def deltas(rows: list[VisitRow]) -> list[Delta]:
    """Change between consecutive visits; ``within_noise`` when the intervals overlap."""
    return [
        Delta(
            a.run_dir,
            b.run_dir,
            b.count - a.count,
            intervals_overlap((a.low, a.high), (b.low, b.high)),
        )
        for a, b in zip(rows, rows[1:], strict=False)
    ]


def render_table(site: Site, rows: list[VisitRow], ds: list[Delta]) -> str:
    lines = [
        f"# {site.name} ({site.id})",
        "",
        "| Captured at | Count | Interval | Change vs previous | Report |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows):
        if i == 0:
            change = "-"
        else:
            d = ds[i - 1]
            change = f"{d.change:+d}" + (" (within noise)" if d.within_noise else "")
        link = f"[report]({r.report_html})" if r.report_html else "-"
        lines.append(
            f"| {r.captured_at} | {r.count} | {int(r.level * 100)}%: {r.low} to {r.high} "
            f"| {change} | {link} |"
        )
    return "\n".join(lines) + "\n"


def plot_counts(site: Site, rows: list[VisitRow], path: Path) -> None:
    """Count with its interval per visit, in the shared ``plotting.py`` style."""
    from countonce import plotting

    plotting.apply_style()
    fig, ax = plotting.plt.subplots(figsize=(6, 3.5))
    xs = [parse_time(r.captured_at) for r in rows]
    ys = [r.count for r in rows]
    err = [[r.count - r.low for r in rows], [r.high - r.count for r in rows]]
    ax.errorbar(xs, ys, yerr=err, fmt="o-", color=plotting.METHOD_COLORS["vggt"], capsize=4)
    ax.set_xlabel("captured at")
    ax.set_ylabel("count")
    ax.set_title(f"{site.name}: count per visit")
    fig.autofmt_xdate()
    plotting.save(fig, path)


def write_report(site: Site, out_dir: Path | str, figure: bool = True) -> dict[str, Path]:
    """Write ``<id>_report.md`` and (when matplotlib is available) ``<id>_counts.png``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = visit_rows(site)
    ds = deltas(rows)
    paths = {"table": out / f"{site.id}_report.md"}
    paths["table"].write_text(render_table(site, rows, ds), encoding="utf-8")
    if figure and rows:
        try:
            plot_counts(site, rows, out / f"{site.id}_counts.png")
            paths["figure"] = out / f"{site.id}_counts.png"
        except ImportError:
            pass
    return paths
