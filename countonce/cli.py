"""``countonce`` console script: ``run`` and ``env`` (PLAN 8.11)."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

from countonce.oom import configure_allocator
from countonce.resources import resolve_config

configure_allocator()  # before anything can import torch

from countonce.config import Config  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="countonce", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="count instances in images or a video")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("source", type=Path, nargs="?", help="image directory, image or video file")
    src.add_argument("--images", type=Path, nargs="+", help="image directory or files")
    src.add_argument("--video", type=Path, help="video file")
    r.add_argument("--prompt", required=True)
    r.add_argument("--config", type=Path, help="YAML Config file; flags override it")
    r.add_argument("--detector", choices=["countgd", "gdino", "countgdpp", "color_blob"])
    r.add_argument("--backbone", choices=["vggt", "vggt_omega", "vggt_t3", "colmap", "fixture"])
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
    t.add_argument(
        "--profile-memory",
        action="store_true",
        help="one step per activation_checkpointing mode; print saved-activation MB, no training",
    )
    t.add_argument("--activation-checkpointing", choices=["none", "attention", "all"])

    f = sub.add_parser("profile", help="time each pipeline stage and record peak VRAM")
    f.add_argument("--config", type=Path, default=Path("configs/profile.yaml"))
    f.add_argument(
        "--variant",
        choices=["cache"],
        help="cache: cold vs new-prompt vs exact-repeat jobs (needs a `cache:` block)",
    )

    y = sub.add_parser("synth", help="write synthetic geometry episodes for the matcher")
    y.add_argument("--config", type=Path, default=Path("configs/synth.yaml"))

    ct = sub.add_parser("catalog", help="DuckDB catalog over synthetic scenes and features")
    ct.add_argument("catalog_cmd", choices=["query"])
    ct.add_argument("sql", help="SQL over the `scenes` (and `detections`) views")
    ct.add_argument("--dir", type=Path, default=Path("data/dag"), help="DAG work dir")

    g = sub.add_parser("dag", help="rerunnable synthetic-data DAG with lineage")
    g.add_argument("dag_cmd", choices=["run", "status"])
    g.add_argument("--config", type=Path, default=Path("configs/dag.yaml"))
    g.add_argument("--from", dest="from_stage", help="rerun this stage and everything after it")
    g.add_argument("--force", action="store_true", help="rerun every stage")

    d = sub.add_parser("drift", help="population-stability drift of current jobs vs a reference")
    d.add_argument("--reference", type=Path, required=True, help="bench CSV or dir of result.json")
    d.add_argument("--current", type=Path, required=True, help="dir of result.json files")
    d.add_argument("--out", type=Path, help="write the drift report JSON here")
    d.add_argument("--warn", type=float, default=0.1, help="PSI warn threshold")
    d.add_argument("--alert", type=float, default=0.25, help="PSI alert threshold")

    sr = sub.add_parser("shadow-report", help="summarize shadow-vs-live agreement from a log")
    sr.add_argument("log", type=Path, help="shadow_log.jsonl")
    sr.add_argument("--json", action="store_true", help="print the report as JSON")
    sr.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    sr.add_argument("--seed", type=int, default=0)

    x = sub.add_parser("explain", help="explain why two detections were or were not merged")
    x.add_argument("out_dir", type=Path, help="directory holding result.json")
    x.add_argument("det_a")
    x.add_argument("det_b")
    x.add_argument("--json", action="store_true", help="print the explanation as JSON")

    r = sub.add_parser("report", help="write a self-contained report.html for a finished run")
    r.add_argument("out_dir", type=Path, help="directory holding result.json")
    r.add_argument("--pdf", action="store_true", help="also write report.pdf (needs weasyprint)")

    st = sub.add_parser("sites", help="group runs into sites and compare counts over visits")
    st.add_argument("--file", type=Path, default=Path("sites.json"), help="sites.json index")
    ssub = st.add_subparsers(dest="sites_cmd", required=True)
    sa = ssub.add_parser("add", help="create a site")
    sa.add_argument("name")
    sa.add_argument("--id", dest="site_id")
    sa.add_argument("--description", default="")
    ssub.add_parser("list", help="list sites and their visits")
    sat = ssub.add_parser("attach", help="attach a run output dir as a visit")
    sat.add_argument("site_id")
    sat.add_argument("run_dir", type=Path)
    sat.add_argument("--captured-at", help="ISO 8601; default from result metadata")
    srp = ssub.add_parser("report", help="counts-over-time table and figure")
    srp.add_argument("site_id")
    srp.add_argument("--out", type=Path, default=Path("."), help="output directory")

    pf = sub.add_parser("preflight", help="check photos for blur, duplicates and overlap")
    pf.add_argument("images", nargs="+", type=Path)
    pf.add_argument("--config", type=Path, help="YAML config with a preflight: threshold block")
    pf.add_argument("--json", action="store_true", help="print the report as JSON")

    lb = sub.add_parser("label", help="write a static label.html for identity labeling a run")
    lb.add_argument("out_dir", type=Path, help="directory holding result.json")
    lb.add_argument("--prelabel", action="store_true", help="seed ids from the run's instances")
    lb.add_argument("--resume", type=Path, help="seed from an exported identity_labels.json")
    lb.add_argument("--dest", type=Path, help="output path (default: <out_dir>/label.html)")

    c = sub.add_parser("correct", help="apply split/merge/reject corrections to a finished run")
    c.add_argument("out_dir", type=Path, help="directory holding result.json")
    c.add_argument("--file", type=Path, required=True, help="corrections.json")
    c.add_argument("--pairs", type=Path, help="also write consented training pairs (JSONL)")

    pr = sub.add_parser("pilot-report", help="write the pilot usage report (docs/PILOT.md)")
    pr.add_argument("--db", type=Path, required=True)
    pr.add_argument("--since", help="ISO date/time; only jobs created at or after it")
    pr.add_argument("--corrections", type=Path, help="corrections.json (not stored in the db)")
    pr.add_argument("--out", type=Path, default=Path("docs/PILOT.md"))

    wk = sub.add_parser("worker", help="run a queue worker against a job database")
    wk.add_argument("--db", type=Path, required=True)
    wk.add_argument("--once", action="store_true", help="drain claimable jobs, then exit")
    wk.add_argument("--out", type=Path, help="job output root (default: <db dir>/outputs)")
    wk.add_argument("--poll", type=float, default=1.0, help="idle poll interval, seconds")
    wk.add_argument("--tracing", choices=["none", "jsonl", "otlp"], default="none")
    wk.add_argument("--trace-dir", type=Path, default=Path("runs/traces"))

    sj = sub.add_parser("submit", help="queue a counting job in a job database")
    sj.add_argument("--db", type=Path, required=True)
    sj.add_argument("--prompt", required=True)
    sj.add_argument("--images", type=Path, nargs="+", required=True)
    sj.add_argument("--user", default="local")
    sj.add_argument("--idempotency-key")
    sj.add_argument("--config", type=Path, help="YAML Config stored as the job options")

    up = sub.add_parser("upload", help="resumable chunked upload into a job database")
    up.add_argument("file", type=Path)
    up.add_argument("--db", type=Path, required=True)
    up.add_argument("--user", default="local")
    up.add_argument("--resume", metavar="SESSION", help="continue an interrupted session")
    up.add_argument("--chunk-mb", type=float, default=4.0, help="chunk size in MiB (default 4)")
    up.add_argument("--job-id", help="attach to this queued job instead of creating one")
    up.add_argument("--prompt", help="prompt for the new job (needed without --job-id)")
    up.add_argument("--limits", type=Path, default=Path("configs/uploads.yaml"))

    tr = sub.add_parser("trace", help="inspect stage traces written by a traced worker or run")
    tsub = tr.add_subparsers(dest="trace_cmd", required=True)
    ts = tsub.add_parser("show", help="print an indented waterfall for a job's trace")
    ts.add_argument("job_id")
    ts.add_argument("--dir", type=Path, default=Path("runs/traces"))

    jb = sub.add_parser("jobs", help="job retention, sharing and API keys on a job database")
    jb.add_argument("--db", type=Path, required=True)
    jb.add_argument("--out", type=Path, help="job output root (default: <db dir>/outputs)")
    jsub = jb.add_subparsers(dest="jobs_cmd", required=True)
    jsub.add_parser("delete", help="delete a job's files and mark it deleted").add_argument(
        "job_id"
    )
    jp = jsub.add_parser("purge", help="delete files of jobs older than the retention window")
    jp.add_argument("--retention-days", type=float, default=30.0)
    jsub.add_parser("share", help="create a read-only share token").add_argument("job_id")
    jsub.add_parser("revoke-share", help="revoke a share token").add_argument("token")
    jk = jsub.add_parser("apikey", help="create or revoke API keys").add_subparsers(
        dest="apikey_cmd", required=True
    )
    jk.add_parser("create", help="create a key (plaintext shown once)").add_argument("user")
    jk.add_parser("revoke", help="revoke a key by id").add_argument("key_id")

    ad = sub.add_parser("admin-stats", help="job-store metrics and alerts (JSON / static HTML)")
    ad.add_argument("--db", type=Path, help="live sqlite job store (default source)")
    ad.add_argument(
        "--export", type=Path, help="read this Parquet export (DuckDB), not the live db"
    )
    ad.add_argument("--alerts", type=Path, default=Path("configs/alerts.yaml"))
    ad.add_argument("--cost", type=Path, default=Path("configs/cost.yaml"), help="cost_per_hour")
    ad.add_argument("--json", action="store_true", help="print metrics and alerts as JSON")
    ad.add_argument("--html", nargs="?", const=Path("admin.html"), type=Path, help="write HTML")
    ad.add_argument("--reference", type=Path, help="drift reference (bench CSV or result.json dir)")
    ad.add_argument("--current", type=Path, help="current result.json dir for drift")

    et = sub.add_parser(
        "etl", help="OLTP -> OLAP: export job telemetry to date-partitioned Parquet"
    )
    et.add_argument("etl_cmd", choices=["export"])
    et.add_argument("--db", type=Path, default=Path("jobs.db"))
    et.add_argument("--out", type=Path, required=True, help="export directory")
    et.add_argument("--since", help="ISO date/time or epoch; skip rows older than this")
    et.add_argument("--corrections", type=Path, help="corrections.json (not stored in the db)")
    et.add_argument("--cost", type=Path, default=Path("configs/cost.yaml"), help="cost_per_hour")

    q = sub.add_parser("review-queue", help="rank finished jobs by matcher uncertainty")
    q.add_argument("jobs", type=Path, help="a job dir, or a dir of job dirs with result.json")
    q.add_argument("--out", type=Path, default=Path("results/review_queue.jsonl"))
    q.add_argument("--top", type=int, help="keep only the N most uncertain jobs")
    q.add_argument("--include-reviewed", action="store_true", help="keep jobs with corrected/")

    rt = sub.add_parser(
        "retrain", help="fine-tune the promoted matcher on opted-in corrections, gate, promote"
    )
    rt.add_argument("--config", type=Path, default=Path("configs/retrain.yaml"))

    sa = sub.add_parser("serve-api", help="serve the HTTP API over a sqlite job store (api extra)")
    sa.add_argument("--db", type=Path, default=Path("jobs.db"))
    sa.add_argument("--upload-root", type=Path, help="default: <db dir>/uploads")
    sa.add_argument("--host", default="127.0.0.1")
    sa.add_argument("--port", type=int, default=8000)
    sa.add_argument("--rate-limit", type=int, help="job submits per minute per user")
    sa.add_argument("--request-rate-limit", type=int, help="requests per minute per user")
    sa.add_argument("--write-docs", type=Path, help="write the OpenAPI spec as Markdown and exit")

    lt = sub.add_parser("loadtest", help="queue load test with fake workers (sqlite job store)")
    lt.add_argument("--config", type=Path, default=Path("configs/loadtest.yaml"))

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


def _find_config(path: Path) -> Path | None:
    """``path`` if it exists, else the packaged copy of a ``configs/<name>`` default, else None."""
    try:
        return resolve_config(path)
    except FileNotFoundError:
        return None


def _read_doc(path: Path | None) -> dict[str, Any]:
    import yaml

    return (yaml.safe_load(path.read_text()) or {}) if path else {}


def _retrain(args: argparse.Namespace) -> int:
    from countonce.errors import CountOnceError
    from countonce.match.registry import RegistryError
    from countonce.match.retrain import (
        RetrainConfig,
        RetrainError,
        evaluators_from_config,
        retrain,
    )

    try:
        cfg = RetrainConfig.from_yaml(args.config)
        frozen, golden = evaluators_from_config(cfg)
        out = retrain(cfg, frozen_evals=frozen, golden_eval=golden)
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        RegistryError,
        RetrainError,
        CountOnceError,
    ) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    row = out.row
    print(
        f"retrain: {out.incumbent} -> {out.candidate} "
        f"corrections={row['n_corrections']} promoted={out.decision.passed}; log {cfg.log_csv}"
    )
    if out.decision.passed:
        return 0
    for r in out.decision.reasons:
        print(f"  - {r}", file=sys.stderr)
    return 1


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


def _serve_api(args: argparse.Namespace) -> int:
    try:
        from countonce.api.app import open_app
        from countonce.api.docs import render_api_md
    except ImportError:
        print(
            "countonce: error: serve-api needs the api extra: pip install .[api]", file=sys.stderr
        )
        return 2
    app = open_app(
        args.db,
        args.upload_root,
        rate_limit=args.rate_limit,
        request_rate_limit=args.request_rate_limit,
    )
    if args.write_docs:
        args.write_docs.parent.mkdir(parents=True, exist_ok=True)
        args.write_docs.write_text(render_api_md(app))
        print(f"serve-api: wrote {args.write_docs}")
        return 0
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _loadtest(args: argparse.Namespace) -> int:
    from countonce.loadtest import LoadTestConfig, format_summary, run_load_test

    try:
        cfg = LoadTestConfig.from_yaml(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    summary = run_load_test(cfg)
    print(format_summary(summary))
    return 0 if summary["reconciled"] else 1


def _pilot_report(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from countonce.monitor.pilot import write_pilot_report

    since = None
    if args.since:
        try:
            dt = datetime.fromisoformat(args.since)
        except ValueError:
            print(f"countonce: error: bad --since {args.since!r}", file=sys.stderr)
            return 2
        since = (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()
    if not args.db.exists():
        print(f"countonce: error: no such db {args.db}", file=sys.stderr)
        return 2
    path = write_pilot_report(args.db, args.out, since=since, corrections_path=args.corrections)
    print(f"pilot-report: wrote {path}")
    return 0


def _worker(args: argparse.Namespace) -> int:
    from countonce.jobs.store import SqliteJobStore
    from countonce.jobs.worker import Worker, pipeline_runner

    store = SqliteJobStore(args.db)
    out = args.out or args.db.resolve().parent / "outputs"
    from countonce.monitor.tracing import TracingConfig

    tcfg = TracingConfig(args.tracing, args.trace_dir)
    worker = Worker(store, pipeline_runner(out), tracing_cfg=tcfg)
    try:
        n = worker.run_loop(until_idle=args.once, poll_s=args.poll)
    except KeyboardInterrupt:
        return 0
    print(f"worker {worker.worker_id}: processed {n} job(s); {store.counts()}")
    return 0


def _trace(args: argparse.Namespace) -> int:
    from countonce.monitor.tracing import load_trace, waterfall

    path = args.dir / f"{args.job_id}.jsonl"
    if not path.exists():
        print(f"countonce: error: no trace for job {args.job_id} in {args.dir}", file=sys.stderr)
        return 2
    print(waterfall(load_trace(path)))
    return 0


def _etl(args: argparse.Namespace) -> int:
    from countonce.monitor import admin, etl

    if not args.db.exists():
        print(f"countonce: error: no such database: {args.db}", file=sys.stderr)
        return 2
    try:
        cost_p = _find_config(args.cost)
        rates = admin.load_cost_rates(cost_p) if cost_p else None
        summary = etl.export(args.db, args.out, args.since, args.corrections, rates)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    detail = ", ".join(f"{t}={n}" for t, n in summary.rows.items())
    print(f"etl: exported {summary.total} new rows in {summary.files} files ({detail})")
    return 0


def _admin_stats(args: argparse.Namespace) -> int:
    import json
    import time

    from countonce.jobs.store import SqliteJobStore
    from countonce.monitor import admin

    now = time.time()
    try:
        if args.export:
            from countonce.monitor.etl import ParquetStore

            store: Any = ParquetStore(args.export)
        elif args.db is None or not args.db.exists():
            print(f"countonce: error: no such database: {args.db}", file=sys.stderr)
            return 2
        else:
            store = SqliteJobStore(args.db)
        alerts_p, cost_p = _find_config(args.alerts), _find_config(args.cost)
        rules = admin.load_rules(alerts_p) if alerts_p else []
        rates = admin.load_cost_rates(cost_p) if cost_p else None
        drift = None
        if args.reference:
            from countonce.monitor.drift import run_drift

            drift = run_drift(args.reference, args.current or args.reference)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    metrics = admin.compute_metrics(store, now, rates)
    alerts = admin.evaluate_alerts(store, rules, now, rates)
    if args.html:
        args.html.write_text(admin.render_html(metrics, alerts, drift))
        print(f"admin-stats: wrote {args.html}")
    if args.json:
        payload = {"metrics": metrics, "alerts": admin.alerts_to_dicts(alerts), "drift": drift}
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not args.html:
        print(
            f"admin-stats: {metrics['n_jobs']} jobs, success rate {metrics['success_rate']}, "
            f"queue depth {metrics['queue_depth']}"
        )
        for a in alerts:
            print(f"ALERT [{a.severity}] {a.rule}: {a.message}")
    return 0


def _submit(args: argparse.Namespace) -> int:
    from countonce.jobs.store import SqliteJobStore

    options = Config.from_yaml(args.config).to_dict() if args.config else {}
    store = SqliteJobStore(args.db)
    job_id, created = store.submit(
        args.user,
        args.prompt,
        [p.resolve() for p in args.images],
        options=options,
        idempotency_key=args.idempotency_key,
    )
    print(f"{job_id} {'queued' if created else 'duplicate (existing job)'}")
    return 0


def _upload(args: argparse.Namespace) -> int:
    from countonce.errors import CountOnceError
    from countonce.jobs.uploads import LocalTransport, open_service, upload_file

    if not args.file.is_file():
        print(f"upload: no such file: {args.file}", file=sys.stderr)
        return 1
    if args.job_id is None and not args.prompt:
        print("upload: pass --prompt (new job) or --job-id", file=sys.stderr)
        return 2
    try:
        service = open_service(args.db, limits_path=args.limits)
        out = upload_file(
            LocalTransport(service),
            args.user,
            args.file,
            chunk_size=max(1, int(args.chunk_mb * 1024 * 1024)),
            resume=args.resume,
            job_id=args.job_id,
            prompt=args.prompt,
            on_session=lambda sid: print(f"upload session {sid}", flush=True),
            on_chunk=lambda i, n: print(f"chunk {i + 1}/{n}", flush=True),
        )
    except CountOnceError as exc:
        print(f"upload: {exc}\n  tip: {exc.tip}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"upload: {exc}", file=sys.stderr)
        return 1
    print(f"upload: job {out['job_id']}")
    return 0


def _jobs(args: argparse.Namespace) -> int:
    from countonce.jobs.store import SqliteJobStore

    store = SqliteJobStore(args.db, output_root=args.out or args.db.parent / "outputs")
    try:
        if args.jobs_cmd == "delete":
            ok = store.delete_job(args.job_id)
            print(f"jobs: {'deleted' if ok else 'not found or already deleted'} {args.job_id}")
            return 0 if ok else 1
        if args.jobs_cmd == "purge":
            ids = store.purge_expired(retention_days=args.retention_days)
            print(f"jobs: purged {len(ids)} job(s) older than {args.retention_days:g} days")
        elif args.jobs_cmd == "share":
            print(store.create_share_token(args.job_id))
        elif args.jobs_cmd == "revoke-share":
            ok = store.revoke_share_token(args.token)
            print(f"jobs: {'revoked' if ok else 'unknown or already revoked'}")
            return 0 if ok else 1
        elif args.apikey_cmd == "create":
            key_id, plaintext = store.create_api_key(args.user)
            print(f"id: {key_id}\nkey: {plaintext}\n(shown once; only a salted hash is stored)")
        else:
            ok = store.revoke_api_key(args.key_id)
            print(f"jobs: {'revoked' if ok else 'unknown or already revoked'} key {args.key_id}")
            return 0 if ok else 1
    except (KeyError, ValueError) as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 1
    return 0


def _sites(args: argparse.Namespace) -> int:
    from countonce import sites

    try:
        idx = sites.SiteIndex.load(args.file)
        if args.sites_cmd == "add":
            site = idx.add(args.name, args.description, args.site_id)
            idx.save()
            print(f"sites: added {site.id}")
        elif args.sites_cmd == "list":
            for s in idx.sites:
                print(f"{s.id}\t{s.name}\t{len(s.visits)} visits")
        elif args.sites_cmd == "attach":
            v = idx.attach(args.site_id, args.run_dir, args.captured_at)
            idx.save()
            print(f"sites: attached {v.run_dir} at {v.captured_at}")
        else:
            paths = sites.write_report(idx.get(args.site_id), args.out)
            for p in paths.values():
                print(f"sites: wrote {p}")
    except sites.SiteError as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    return 0


def _preflight(args: argparse.Namespace) -> int:
    import json

    from countonce.ingest.preflight import load_thresholds, preflight

    try:
        overrides: dict[str, float] = {}
        if args.config:
            import yaml

            overrides = (yaml.safe_load(args.config.read_text()) or {}).get("preflight", {})
        rep = preflight(args.images, load_thresholds(overrides))
    except (OSError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print(f"preflight: {len(rep.photos)} photos, diversity {rep.diversity:.3f}")
        for i, (n, s) in enumerate(zip(rep.photos, rep.sharpness, strict=True), 1):
            print(f"  {i}. {n}: sharpness {s:.1f}")
        print("\n".join(rep.tips) if rep.tips else "Looks good.")
    return 0 if rep.ok else 1


def _label(args: argparse.Namespace) -> int:
    from bench.label.html import write_label_html
    from bench.label.io import LabelError

    try:
        path = write_label_html(args.out_dir, args.dest, args.prelabel, args.resume)
    except (FileNotFoundError, ValueError, LabelError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(f"label: wrote {path}; open it in a browser and export identity_labels.json")
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


def _shadow_report(args: argparse.Namespace) -> int:
    import json

    from countonce.routing import format_report, load_log, summarize

    try:
        rep = summarize(load_log(args.log), n_boot=args.boot, seed=args.seed)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(rep, indent=2) if args.json else format_report(rep))
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
        if args.activation_checkpointing:
            tcfg = dataclasses.replace(tcfg, activation_checkpointing=args.activation_checkpointing)
        if args.profile_memory:
            from countonce.match.train import profile_memory

            for row in profile_memory(tcfg.validated()):
                vram = "n/a" if row["peak_vram_mb"] is None else f"{row['peak_vram_mb']:.1f}"
                print(
                    f"profile-memory: {row['mode']:<9} saved={row['saved_mb']:.2f} MB "
                    f"step={row['step_s']:.3f}s peak_vram_mb={vram}"
                )
            return 0
        history = train(tcfg.validated())
    except (FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(f"train: {len(history)} epochs logged")
    return 0


def _catalog(args: argparse.Namespace) -> int:
    from synth.catalog import CatalogError, connect, format_table, run_query

    try:
        con = connect(
            args.dir / "catalog" / "scenes.parquet", args.dir / "features" / "features.parquet"
        )
        cols, rows = run_query(con, args.sql)
    except (CatalogError, OSError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(format_table(cols, rows))
    return 0


def _dag(args: argparse.Namespace) -> int:
    from synth.dag import Dag, DagConfig, DagError, format_status

    try:
        dag = Dag(DagConfig.from_yaml(args.config))
        if args.dag_cmd == "status":
            print(format_status(dag.status()))
            return 0
        for stage, state in dag.run(from_stage=args.from_stage, force=args.force).items():
            print(f"dag: {stage}: {state}")
    except (FileNotFoundError, DagError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
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
    from bench.sweep import (
        BenchConfig,
        build_jobs,
        dataset_scenes,
        discover_scenes,
        run_bench,
    )

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
        scenes = discover_scenes(cfg.scenes_dir) + dataset_scenes(cfg)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"{len(scenes)} scenes, {len(build_jobs(cfg, scenes))} jobs -> {cfg.csv}")
        return 0
    ran, skipped, errored = run_bench(cfg, scenes)
    print(f"bench: ran {ran}, skipped {skipped}, errored {errored} -> {cfg.csv}")
    return 0


def _profile(args: argparse.Namespace) -> int:
    from countonce.profile import (
        ProfileConfig,
        run_profile,
        run_profile_cache,
        run_profile_variants,
    )

    try:
        pcfg = ProfileConfig.from_yaml(args.config)
        if pcfg.variants and args.variant != "cache":
            path, vrows = run_profile_variants(pcfg)
            print(f"profile: {len(vrows)} variant rows ({len(pcfg.variants)} variants) -> {path}")
            return 0
        if args.variant == "cache":
            path, rows = run_profile_cache(pcfg)
            print(f"profile: {len(rows)} cache-state rows -> {path}")
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
    cfg_arg = getattr(args, "config", None)
    if isinstance(cfg_arg, Path) and not cfg_arg.exists():
        try:
            args.config = resolve_config(cfg_arg)
        except FileNotFoundError as exc:
            print(f"countonce: error: {exc}", file=sys.stderr)
            return 2
    if args.cmd == "env":
        from countonce import env_check

        return env_check.main(args.out)
    if args.cmd == "bench":
        return _bench(args)
    if args.cmd == "ablate":
        return _ablate(args)
    if args.cmd == "train":
        return _train(args)
    if args.cmd == "dag":
        return _dag(args)
    if args.cmd == "catalog":
        return _catalog(args)
    if args.cmd == "synth":
        return _synth(args)
    if args.cmd == "profile":
        return _profile(args)
    if args.cmd == "pilot-report":
        return _pilot_report(args)
    if args.cmd == "drift":
        return _drift(args)
    if args.cmd == "shadow-report":
        return _shadow_report(args)
    if args.cmd == "explain":
        return _explain(args)
    if args.cmd == "correct":
        return _correct(args)
    if args.cmd == "models":
        return _models(args)
    if args.cmd == "retrain":
        return _retrain(args)
    if args.cmd == "review-queue":
        return _review_queue(args)
    if args.cmd == "preflight":
        return _preflight(args)
    if args.cmd == "label":
        return _label(args)
    if args.cmd == "sites":
        return _sites(args)
    if args.cmd == "worker":
        return _worker(args)
    if args.cmd == "serve-api":
        return _serve_api(args)
    if args.cmd == "loadtest":
        return _loadtest(args)
    if args.cmd == "etl":
        return _etl(args)
    if args.cmd == "admin-stats":
        return _admin_stats(args)
    if args.cmd == "submit":
        return _submit(args)
    if args.cmd == "upload":
        return _upload(args)
    if args.cmd == "jobs":
        return _jobs(args)
    if args.cmd == "trace":
        return _trace(args)
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
    inputs = args.images if args.images is not None else (args.source or args.video)
    if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
    if config.backbone == "fixture" and "scene" not in config.backbone_options:
        scene_dir = inputs if isinstance(inputs, Path) and inputs.is_dir() else None
        if scene_dir is not None:  # test backend: scene.json sits next to the views
            opts = {**config.backbone_options, "scene": str(scene_dir / "scene.json")}
            config = config.model_copy(update={"backbone_options": opts})
    try:
        result = run(inputs, args.prompt, config)
    except (NotImplementedError, FileNotFoundError, ValueError) as exc:
        print(f"countonce: error: {exc}", file=sys.stderr)
        return 2
    print(f"{result.prompt}: {result.count} (results in {config.out_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
