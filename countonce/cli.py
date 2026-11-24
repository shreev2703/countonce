"""``countonce`` console script: ``run`` and ``env`` (PLAN 8.11)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from countonce.oom import configure_allocator

configure_allocator()  # before anything can import torch

from countonce.config import Config  # noqa: E402


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
    r.add_argument("--assoc", choices=["v1", "v2", "v3", "sync"])
    r.add_argument("--v3-checkpoint", type=Path, help="matcher checkpoint for --assoc v3")
    r.add_argument("--out", type=Path)
    r.add_argument("--fps", type=float)

    e = sub.add_parser("env", help="check the Python/GPU environment")
    e.add_argument("--out", type=Path, default=Path("results/env.json"))

    b = sub.add_parser("bench", help="run the benchmark sweep, or render figures from its CSV")
    b.add_argument("--config", type=Path, default=Path("configs/bench.yaml"))
    b.add_argument("--dry-run", action="store_true", help="print the job grid size and exit")
    b.add_argument(
        "--allow-paid", action="store_true", help="permit paid VLM API calls (kind: vlm methods)"
    )
    b.add_argument(
        "--spend-cap-usd", type=float, help="per-run estimated spend cap; required for paid calls"
    )
    b.add_argument(
        "--figures", type=Path, metavar="DIR", help="render figures/gallery from the CSV"
    )

    b.add_argument(
        "--synth-assoc",
        nargs="?",
        const=Path("configs/synth_assoc.yaml"),
        type=Path,
        metavar="CONFIG",
        help="run v1/v2/sync/v3 on held-out synthetic episodes (association metrics)",
    )

    b.add_argument(
        "--tune-warnings",
        nargs="?",
        const=Path("configs/warnings.yaml"),
        type=Path,
        metavar="OUT",
        help="tune quality-warning thresholds on the val split of the bench CSV",
    )
    b.add_argument(
        "--tune-method", default="co_v2_vggt", help="bench method used by --tune-warnings"
    )

    a = sub.add_parser("ablate", help="v3 arm and feature-group ablation on synthetic episodes")
    a.add_argument("--config", type=Path, default=Path("configs/synth_ablation.yaml"))

    t = sub.add_parser("train", help="train the CountOnce-Match v3 matcher")
    t.add_argument("--config", type=Path, default=Path("configs/match.yaml"))

    f = sub.add_parser("profile", help="time each pipeline stage and record peak VRAM")
    f.add_argument("--config", type=Path, default=Path("configs/profile.yaml"))

    y = sub.add_parser("synth", help="write synthetic geometry episodes for the matcher")
    y.add_argument("--config", type=Path, default=Path("configs/synth.yaml"))
    return p


def _train(args: argparse.Namespace) -> int:
    from countonce.match.train import TrainConfig, train

    try:
        history = train(TrainConfig.from_yaml(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(f"train: {len(history)} epochs logged")
    return 0


def _synth(args: argparse.Namespace) -> int:
    import yaml

    from synth.geometry import GeometryConfig, write_dataset

    try:
        raw = yaml.safe_load(args.config.read_text()) or {}
        geo = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in (raw.get("geometry") or {}).items()
        }
        cfg = GeometryConfig(**geo)
        seed, out = int(raw.get("seed", 0)), Path(raw.get("out_dir", "data/synth"))
        offset = 0
        for split, n in (raw.get("splits") or {"train": 100}).items():
            paths = write_dataset(out / split, int(n), seed0=seed + offset, cfg=cfg)
            offset += int(n)
            print(f"synth: {split}: {len(paths)} episodes -> {out / split}")
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    return 0


def _synth_assoc(args: argparse.Namespace) -> int:
    from bench.synth_assoc import SynthAssocConfig, run_synth_assoc

    try:
        cfg = SynthAssocConfig.from_yaml(args.synth_assoc)
        ran, skipped, errored = run_synth_assoc(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(f"synth-assoc: ran {ran}, skipped {skipped}, errored {errored} -> {cfg.csv}")
    return 0


def _ablate(args: argparse.Namespace) -> int:
    from bench.synth_ablation import AblationConfig, run_ablation

    try:
        summary, per_seed = run_ablation(AblationConfig.from_yaml(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(f"ablate: summary -> {summary}, per-seed -> {per_seed}")
    return 0


def _bench(args: argparse.Namespace) -> int:
    if args.synth_assoc is not None:
        return _synth_assoc(args)
    from bench.sweep import BenchConfig, build_jobs, discover_scenes, run_bench

    cfg = BenchConfig.from_yaml(args.config)
    if args.tune_warnings is not None:
        from bench.tune_warnings import tune_from_csv

        try:
            doc = tune_from_csv(cfg.csv, args.tune_warnings, method=args.tune_method)
        except (FileNotFoundError, ValueError) as exc:
            print(f"countonce: error: {exc}", file=sys.stderr)
            return 2
        n_tuned = len(doc["warnings"])
        print(
            f"tune-warnings: {n_tuned} tuned from {doc['n_scenes']} val scenes "
            f"-> {args.tune_warnings}"
        )
        return 0
    if args.allow_paid:
        cfg = cfg.model_copy(update={"allow_paid": True})
    if args.spend_cap_usd is not None:
        cfg = cfg.model_copy(update={"spend_cap_usd": args.spend_cap_usd})
    if args.figures is not None:
        from bench.figures import make_all
        from bench.gallery import write_gallery

        try:
            for f in make_all(cfg.csv, args.figures, cfg.ablation_base):
                print(f)
            write_gallery(cfg.csv, args.figures / "failure_gallery.md", runs_dir=cfg.runs_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"countonce: error: {exc}", file=sys.stderr)
            return 2
        return 0
    try:
        scenes = discover_scenes(cfg.scenes_dir)
    except FileNotFoundError as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"{len(scenes)} scenes, {len(build_jobs(cfg, scenes))} jobs -> {cfg.csv}")
        return 0
    ran, skipped, errored = run_bench(cfg, scenes)
    print(f"bench: ran {ran}, skipped {skipped}, errored {errored} -> {cfg.csv}")
    return 0


def _profile(args: argparse.Namespace) -> int:
    from countonce.profile import ProfileConfig, run_profile, run_profile_variants

    try:
        pcfg = ProfileConfig.from_yaml(args.config)
        if pcfg.variants:
            path, vrows = run_profile_variants(pcfg)
            print(f"profile: {len(vrows)} variant rows ({len(pcfg.variants)} variants) -> {path}")
            return 0
        path, rows = run_profile(pcfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(f"profile: {len(rows)} stage rows -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_allocator()
    args = _parser().parse_args(argv)
    if args.cmd == "env":
        from countonce import env_check

        return env_check.main(args.out)
    if args.cmd == "bench":
        return _bench(args)
    if args.cmd == "ablate":
        return _ablate(args)
    if args.cmd == "train":
        return _train(args)
    if args.cmd == "synth":
        return _synth(args)
    if args.cmd == "profile":
        return _profile(args)

    from countonce.pipeline import run

    base = Config.from_yaml(args.config) if args.config else Config()
    config = base.with_overrides(
        detector=args.detector,
        backbone=args.backbone,
        assoc=args.assoc,
        v3_checkpoint=args.v3_checkpoint,
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
