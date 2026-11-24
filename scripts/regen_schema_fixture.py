"""Regenerate the stored schema-compatibility fixtures (PLAN 8.10, section 18).

Runs the CPU end-to-end pipeline on the generated synthetic scene from
``tests/test_pipeline.py`` and writes ``result.json`` and ``markers.json`` for the
*current* ``schema_version`` into ``tests/fixtures/schemas/v<version>/``.

Older version directories are never touched, and an existing directory for the current
version is never overwritten (pass ``--force`` to replace it on purpose, e.g. while a
version is still unreleased). Absolute paths and machine-specific provenance are scrubbed.

    python scripts/regen_schema_fixture.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "schemas"


def scrub(raw: dict[str, Any]) -> dict[str, Any]:
    """Make a freshly produced result.json machine-independent (relative paths only)."""
    cfg = raw.get("config", {})
    inner = cfg.get("config", {})
    inner["out_dir"] = "out"
    inner["cache_dir"] = "cache"
    cfg["git_commit"] = "fixture"
    cfg["git_dirty"] = False
    cfg["timestamp_utc"] = "2000-01-01T00:00:00+00:00"
    cfg["torch"] = None
    cfg["gpu"] = None
    return raw


def produce(work: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
    from test_pipeline import Config, FakeDetector, PlaneRecon, make_scene, run

    img_dir, script = make_scene(work)
    cfg = Config(out_dir=work / "out", cache_dir=work / "cache", max_frames=0)
    run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    result = json.loads((work / "out" / "result.json").read_text())
    markers = json.loads((work / "out" / "markers.json").read_text())
    return scrub(result), markers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="replace the current version's fixture")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from countonce.schemas import CURRENT_VERSION

    dest = FIXTURE_ROOT / f"v{CURRENT_VERSION}"
    if dest.exists() and not args.force:
        print(f"{dest} already exists; refusing to overwrite (use --force).", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        result, markers = produce(Path(tmp))
    dest.mkdir(parents=True, exist_ok=True)
    for name, doc in (("result.json", result), ("markers.json", markers)):
        (dest / name).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
