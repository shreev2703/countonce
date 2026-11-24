"""``countonce`` console script: ``run`` and ``env`` (PLAN 8.11)."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

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
    t.add_argument(
        "--corrections",
        type=Path,
        help="JSONL of consented correction pairs (countonce correct --pairs) to mix in",
    )

    f = sub.add_parser("profile", help="time each pipeline stage and record peak VRAM")
    f.add_argument("--config", type=Path, default=Path("configs/profile.yaml"))

    y = sub.add_parser("synth", help="write synthetic geometry episodes for the matcher")
    y.add_argument("--config", type=Path, default=Path("configs/synth.yaml"))

    d = sub.add_parser("drift", help="population-stability drift of current jobs vs a reference")
    d.add_argument("--reference", type=Path, required=True, help="bench CSV or dir of result.json")
    d.add_argument("--current", type=Path, required=True, help="dir of result.json files")
    d.add_argument("--out", type=Path, help="write the drift report JSON here")
    d.add_argument("--warn", type=float, default=0.1, help="PSI warn threshold")
    d.add_argument("--alert", type=float, default=0.25, help="PSI alert threshold")

    x = sub.add_parser("explain", help="explain why two detections were or were not merged")
    x.add_argument("out_dir", type=Path, help="directory holding result.json")
    x.add_argument("det_a")
    x.add_argument("det_b")
    x.add_argument("--json", action="store_true", help="print the explanation as JSON")

    r = sub.add_parser("report", help="write a self-contained report.html for a finished run")
    r.add_argument("out_dir", type=Path, help="directory holding result.json")
    r.add_argument("--pdf", action="store_true", help="also write report.pdf (needs weasyprint)")

    c = sub.add_parser("correct", help="apply split/merge/reject corrections to a finished run")
    c.add_argument("out_dir", type=Path, help="directory holding result.json")
    c.add_argument("--file", type=Path, required=True, help="corrections.json")
    c.add_argument("--pairs", type=Path, help="also write consented training pairs (JSONL)")

    q = sub.add_parser("review-queue", help="rank finished jobs by matcher uncertainty")
    q.add_argument("jobs", type=Path, help="a job dir, or a dir of job dirs with result.json")
    q.add_argument("--out", type=Path, default=Path("results/review_queue.jsonl"))
    q.add_argument("--top", type=int, help="keep only the N most uncertain jobs")
    q.add_argument("--include-reviewed", action="store_true", help="keep jobs with corrected/")

    m = sub.add_parser("models", help="matcher checkpoint registry")
    m.add_argument("--root", type=Path, default=Path("models"))
    msub = m.add_subparsers(dest="models_cmd", required=True)
    msub.add_parser("list", help="list registered versions and the current one")
    mr = msub.add_parser("register", help="register a checkpoint (does not promote it)")
    mr.add_argument("checkpoint", type=Path)
    mr.add_argument("--dataset-version", default="")
    mr.add_argument("--recipe-version", default="")
    mr.add_argument("--config", type=Path, help="YAML/JSON training config to record")
    mr.add_argument("--metrics", type=Path, help="JSON file of metrics to record")
    mp = msub.add_parser("promote", help="gate a version against the current model, then promote")
    mp.add_argument("version")
    mp.add_argument("--frozen-gt", type=Path, help="frozen-split ground truth JSON")
    mp.add_argument(
        "--frozen-predictions", type=Path, help="dir holding <version>.json predictions"
    )
    mp.add_argument("--golden-config", type=Path, help="golden set config (default: built in)")
    mp.add_argument("--no-golden", action="store_true", help="skip the CPU golden set")
    mp.add_argument("--allow-no-frozen", action="store_true", help="do not require a frozen eval")
    mp.add_argument("--tol-mae-rel", type=float, default=0.05)
    mp.add_argument("--tol-f1", type=float, default=0.01)
    mp.add_argument("--tol-golden-mape", type=float, default=0.02)
    mp.add_argument("--tol-golden-f1", type=float, default=0.02)
    return p


def _review_queue(args: argparse.Namespace) -> int:
    from countonce.active import export_review_queue, find_jobs, rank_jobs
    from countonce.errors import CountOnceError

    try:
        items = rank_jobs(find_jobs(args.jobs), skip_reviewed=not args.include_reviewed)
    except (FileNotFoundError, ValueError, CountOnceError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    export_review_queue(items, args.out, args.top)
    print(f"review-queue: {len(items[: args.top])} of {len(items)} jobs -> {args.out}")
    return 0


def _read_doc(path: Path | None) -> dict[str, Any]:
    import yaml

    return (yaml.safe_load(path.read_text()) or {}) if path else {}


def _models(args: argparse.Namespace) -> int:

    from countonce.match.registry import (
        GateConfig,
        Registry,
        RegistryError,
        frozen_evaluator,
        golden_evaluator,
    )

    reg = Registry(args.root)
    try:
        if args.models_cmd == "list":
            rows = reg.models()
            if not rows:
                print("models: registry is empty")
            for m in rows:
                print(
                    f"{'*' if m['current'] else ' '} {m['version']}  {m['sha256'][:12]}  "
                    f"data={m['dataset_version'] or '-'}  git={(m['git_sha'] or '-')[:8]}"
                )
            return 0
        if args.models_cmd == "register":
            entry = reg.register(
                args.checkpoint,
                dataset_version=args.dataset_version,
                recipe_version=args.recipe_version,
                config=_read_doc(args.config),
                metrics=_read_doc(args.metrics),
            )
            print(f"models: registered {entry['version']} ({entry['sha256'][:12]})")
            return 0
        gate = GateConfig(
            frozen_mae_rel=args.tol_mae_rel,
            frozen_assoc_f1_abs=args.tol_f1,
            require_frozen=not args.allow_no_frozen,
        )
        gate.golden["mape"] = args.tol_golden_mape
        gate.golden["assoc_f1"] = args.tol_golden_f1
        frozen = None
        if args.frozen_gt and args.frozen_predictions:
            frozen = frozen_evaluator(args.frozen_gt, args.frozen_predictions)
        golden = None if args.no_golden else golden_evaluator(args.golden_config)
        decision = reg.promote(args.version, gate=gate, frozen_eval=frozen, golden_eval=golden)
    except (FileNotFoundError, ValueError, KeyError, RegistryError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    if decision.passed:
        print(f"models: promoted {decision.candidate} (was {decision.incumbent})")
        return 0
    print(f"models: REFUSED {decision.candidate}: logged to {reg.log_path}", file=sys.stderr)
    for r in decision.reasons:
        print(f"  - {r}", file=sys.stderr)
    return 1


def _correct(args: argparse.Namespace) -> int:
    import json

    from countonce.corrections import (
        apply_corrections,
        export_training_pairs,
        load_corrections,
        write_corrected,
        write_training_pairs_jsonl,
    )
    from countonce.errors import CountOnceError
    from countonce.explain import load_result_dir

    try:
        result = load_result_dir(args.out_dir)
        corrections = load_corrections(args.file)
        corrected = apply_corrections(result, corrections)
    except (FileNotFoundError, ValueError, KeyError, CountOnceError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    scale = None
    mpath = Path(args.out_dir) / "markers.json"
    if mpath.is_file():
        scale = json.loads(mpath.read_text(encoding="utf-8")).get("scene_scale")
    dest = Path(args.out_dir) / "corrected"
    write_corrected(corrected, dest, scale)
    print(f"correct: count {result.count} -> {corrected.count}; wrote {dest}")
    if args.pairs:
        jobs = {c.job_id: result for c in corrections}
        pairs = export_training_pairs(corrections, jobs)
        write_training_pairs_jsonl(pairs, args.pairs)
        print(f"correct: wrote {len(pairs)} consented pairs to {args.pairs}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from countonce.export.report import write_report

    try:
        paths = write_report(args.out_dir, pdf=args.pdf)
    except (FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    for p in paths:
        print(f"report: wrote {p}")
    return 0


def _explain(args: argparse.Namespace) -> int:
    import json

    from countonce.explain import explain_pair, load_result_dir

    try:
        exp = explain_pair(load_result_dir(args.out_dir), args.det_a, args.det_b)
    except (FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(exp.to_dict(), indent=2) if args.json else exp.to_text())
    return 0


def _drift(args: argparse.Namespace) -> int:
    from countonce.monitor.drift import DriftThresholds, run_drift

    try:
        rep = run_drift(
            args.reference, args.current, args.out, DriftThresholds(args.warn, args.alert)
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    for name, v in rep["features"].items():
        print(f"drift: {name}: psi={v['psi']:.4f} {v['level']}")
    print(f"drift: status {rep['status']} ({len(rep['flagged'])} flagged)")
    return 0


def _train(args: argparse.Namespace) -> int:
    from countonce.match.train import TrainConfig, train

    try:
        tcfg = TrainConfig.from_yaml(args.config)
        if args.corrections:
            tcfg = dataclasses.replace(tcfg, corrections=str(args.corrections))
        history = train(tcfg)
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
        geo: dict[str, Any] = {
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
    if args.cmd == "drift":
        return _drift(args)
    if args.cmd == "explain":
        return _explain(args)
    if args.cmd == "correct":
        return _correct(args)
    if args.cmd == "models":
        return _models(args)
    if args.cmd == "review-queue":
        return _review_queue(args)
    if args.cmd == "report":
        return _report(args)

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
