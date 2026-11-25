"""Build the static demo assets under ``web/public/demo/`` (PLAN §0.2, M13).

    PYTHONPATH=. .venv/bin/python scripts/build_demo.py            # tiny scene, CPU, no weights
    PYTHONPATH=. .venv/bin/python scripts/build_demo.py --check    # CI: versions + hashes + schemas
    PYTHONPATH=. .venv/bin/python scripts/build_demo.py --real     # GroundingDINO + VGGT (GPU)

The tiny scene is a synthetic test scene and is labelled as such. Own captures (P4) are added
through ``--scene ID=DIR:PROMPT`` once they exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from countonce.export.demo import DemoVersionError, build_scene, check_demo

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "examples" / "tiny_scene"
TINY_LABEL = "synthetic test scene"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "demo")
    ap.add_argument("--check", action="store_true", help="validate built assets and versions only")
    ap.add_argument(
        "--real", action="store_true", help="GroundingDINO + VGGT instead of CPU stand-ins"
    )
    ap.add_argument("--scene", action="append", default=[], metavar="ID=DIR:PROMPT",
                    help="extra scene (own captures); label 'own capture'")  # fmt: skip
    a = ap.parse_args(argv)

    if a.check:
        try:
            scenes = check_demo(a.out, ROOT)
        except DemoVersionError as e:
            print(f"demo check FAILED: {e}", file=sys.stderr)
            return 1
        print(f"demo check ok: {', '.join(scenes)}")
        return 0

    gt = json.loads((TINY / "scene.json").read_text())["gt_count"]
    # the five views are deliberately different: keep all of them (default dedup would keep one)
    kw: dict[str, Any] = {"extra_config": {"dedup_threshold": 0}}
    if a.real:
        kw = {
            "detector": "gdino",
            "backbone": "vggt",
            "extra_config": {"dedup_threshold": 0, "score_threshold": 0.5, "nms_iou": 0.7},
        }
    build_scene(TINY, a.out, scene_id="tiny", prompt="red sphere", label=TINY_LABEL,
                gt_count=gt, repo_root=ROOT, **kw)  # fmt: skip
    for spec in a.scene:
        sid, rest = spec.split("=", 1)
        d, prompt = rest.rsplit(":", 1)
        build_scene(Path(d), a.out, scene_id=sid, prompt=prompt, label="own capture",
                    detector="gdino", backbone="vggt", repo_root=ROOT)  # fmt: skip
    print(f"built demo assets in {a.out}")
    check_demo(a.out, ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
