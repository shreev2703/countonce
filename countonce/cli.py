"""``countonce`` console script: ``run`` and ``env`` (PLAN 8.11)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from countonce.config import Config


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="countonce", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="count instances in images or a video")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=Path, nargs="+", help="image directory or files")
    src.add_argument("--video", type=Path, help="video file")
    r.add_argument("--prompt", required=True)
    r.add_argument("--config", type=Path, help="YAML Config file; flags override it")
    r.add_argument("--detector", choices=["countgd", "gdino", "countgdpp"])
    r.add_argument("--backbone", choices=["vggt", "vggt_omega", "vggt_t3", "colmap"])
    r.add_argument("--assoc", choices=["v1", "v2", "v3"])
    r.add_argument("--out", type=Path)
    r.add_argument("--fps", type=float)

    e = sub.add_parser("env", help="check the Python/GPU environment")
    e.add_argument("--out", type=Path, default=Path("results/env.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "env":
        from countonce import env_check

        return env_check.main(args.out)

    from countonce.pipeline import run

    base = Config.from_yaml(args.config) if args.config else Config()
    config = base.with_overrides(
        detector=args.detector,
        backbone=args.backbone,
        assoc=args.assoc,
        out_dir=args.out,
        fps=args.fps,
    )
    inputs = args.images if args.images is not None else args.video
    if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
    try:
        result = run(inputs, args.prompt, config)
    except (NotImplementedError, FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(f"{result.prompt}: {result.count} (results in {config.out_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
