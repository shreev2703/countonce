"""Local CountOnce API plus a fake worker, for the web live end-to-end test (web/e2e/live).

Real FastAPI app, sqlite store and chunked-upload service. The "worker" is a thread that claims
queued jobs, reports progress and completes them with the stored v1.1 result fixture, so the web
flow runs without a GPU. The worker starts offline while ``--pause-file`` exists (no heartbeats,
no claims); delete the file to bring it online, touch it to pause it again.

    .venv/bin/python scripts/e2e_api_server.py --port 4020 --pause-file /tmp/worker.pause
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path

import uvicorn

from countonce.api.app import create_app
from countonce.jobs.store import SqliteJobStore
from countonce.jobs.uploads import UploadService

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/schemas/v1.1/result.json"
STAGES = ("ingest", "detecting", "recon", "matching", "counting")


def fake_worker(store: SqliteJobStore, pause_file: Path | None, stop: threading.Event) -> None:
    raw = json.loads(FIXTURE.read_text())
    wid = "e2e-fake"
    while not stop.is_set():
        if pause_file is not None and pause_file.exists():
            time.sleep(0.1)  # offline: no heartbeat, so /health reports processing_paused
            continue
        store.register_worker(wid, name="fake", kind="e2e")
        job = store.claim(wid)
        if job is None:
            time.sleep(0.1)
            continue
        for i, stage in enumerate(STAGES):
            store.set_progress(job["id"], wid, stage, (i + 1) / (len(STAGES) + 1))
            time.sleep(0.15)
        res = {**raw, "prompt": job["prompt"]}
        iv = res["interval"]
        store.complete(
            job["id"],
            wid,
            {
                "schema_version": res["schema_version"],
                "count": res["count"],
                "interval_low": iv["low"],
                "interval_high": iv["high"],
                "interval_level": iv["level"],
                "interval_method": iv["method"],
                "result_json": json.dumps(res),
                "warnings_json": json.dumps(res["warnings"]),
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4020)
    ap.add_argument("--pause-file", type=Path)
    ap.add_argument("--secret", default=None, help="default: $COUNTONCE_AUTH_SECRET")
    args = ap.parse_args()
    import os

    secret = args.secret or os.environ.get("COUNTONCE_AUTH_SECRET")
    if not secret:
        raise SystemExit("set COUNTONCE_AUTH_SECRET or --secret")
    if args.pause_file:
        args.pause_file.touch()  # start with the worker offline; a test deletes the file to resume
    root = Path(tempfile.mkdtemp(prefix="countonce-e2e-"))
    store = SqliteJobStore(root / "jobs.db", output_root=root / "out")
    app = create_app(
        store, UploadService(store, root / "uploads"), sse_poll_s=0.1, auth_secret=secret
    )
    stop = threading.Event()
    threading.Thread(target=fake_worker, args=(store, args.pause_file, stop), daemon=True).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
