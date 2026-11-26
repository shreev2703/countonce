"""Draw the README figures from the tiny synthetic scene and two finished runs.

    countonce run examples/tiny_scene --prompt "red sphere" --detector color_blob \\
        --backbone fixture --assoc v2 --config examples/tiny_scene/all_views.yaml --out results/tiny
    countonce run examples/tiny_scene --prompt "red sphere" --assoc v2 \\
        --config configs/readme_real_failure.yaml --out results/tiny_real     # GPU
    python scripts/make_readme_figures.py --standin results/tiny --real results/tiny_real

Numbers on the pictures (instance IDs, 1-based as on the website) come from each ``result.json``;
nothing is placed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "examples" / "tiny_scene"
VIEWS = 5
SCALE = 3  # the source views are 128 x 96
GAP = 6
COLORS = ["#ffd400", "#00e5ff", "#ff4fd8", "#7cff4f", "#ffffff", "#ff9f1c", "#9d8cff"]


def _views() -> list[Image.Image]:
    out = []
    for i in range(VIEWS):
        img = Image.open(SCENE / f"view_{i}.png").convert("RGB")
        out.append(img.resize((img.width * SCALE, img.height * SCALE), Image.Resampling.NEAREST))
    return out


def _strip(tiles: list[Image.Image], out: Path) -> None:
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * len(tiles) + GAP * (len(tiles) - 1), h), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * (w + GAP), 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)


def _with_ids(result: dict[str, Any]) -> list[Image.Image]:
    """Mark every counted detection at its centre with its instance number."""
    ids = {m: n for n, inst in enumerate(result["instances"], 1) for m in inst["members"]}
    # detection centres are in the backbone's pixel grid; its principal point is the grid centre
    grid = {c["view"]: (2 * c["intrinsics"][0][2], 2 * c["intrinsics"][1][2])
            for c in result["cameras"]}  # fmt: skip
    tiles = _views()
    font = ImageFont.load_default(size=22)
    for det in result["detections"]:
        n = ids.get(det["id"])
        if n is None:
            continue
        tile = tiles[det["view"]]
        gw, gh = grid[det["view"]]
        x, y = det["uv"][0] * tile.width / gw, det["uv"][1] * tile.height / gh
        color = COLORS[(n - 1) % len(COLORS)]
        draw = ImageDraw.Draw(tile)
        draw.ellipse((x - 16, y - 16, x + 16, y + 16), outline=color, width=3)
        draw.text((x, y - 28), str(n), fill=color, font=font, anchor="mm",
                  stroke_width=2, stroke_fill="black")  # fmt: skip
    return tiles


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--standin", type=Path, required=True, help="CPU stand-in run directory")
    ap.add_argument("--real", type=Path, required=True, help="GroundingDINO + VGGT run directory")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "assets")
    a = ap.parse_args(argv)
    _strip(_views(), a.out / "tiny_inputs.png")
    for run, name in ((a.standin, "tiny_ids_standin.png"), (a.real, "tiny_failure_real.png")):
        result = json.loads((run / "result.json").read_text())
        _strip(_with_ids(result), a.out / name)
        print(f"{name}: count {result['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
