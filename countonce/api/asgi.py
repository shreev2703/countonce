"""Module-level ASGI app for serverless hosts (Vercel entrypoint ``countonce.api.asgi:app``).

Long work never runs here: this app only stores jobs and files and serves status; the GPU worker
(``countonce worker``) picks queued jobs up from the shared store. Server-Sent Events are capped by
``COUNTONCE_SSE_MAX_S`` (set it below the function's max duration); the web client reconnects or
falls back to polling. See docs/DEPLOY.md.

Uses Postgres when ``COUNTONCE_PG_URL`` is set. Otherwise it falls back to a SQLite file under the
system temp dir, which is ephemeral on serverless hosts: fine for a smoke check, not for real jobs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from countonce.api.app import open_app


def build_app() -> FastAPI:
    root = Path(os.environ.get("COUNTONCE_DATA_DIR", tempfile.gettempdir())) / "countonce"
    root.mkdir(parents=True, exist_ok=True)
    kind = "postgres" if os.environ.get("COUNTONCE_PG_URL") else "sqlite"
    app = open_app(root / "jobs.db", root / "uploads", store_kind=kind)

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def index() -> RedirectResponse:
        """The API has no page of its own: send people to the site (``COUNTONCE_WEB_URL``), or to
        the API docs when that is not set. Probes follow the redirect or use ``/health``."""
        web = (os.environ.get("COUNTONCE_WEB_URL") or "").rstrip("/")
        return RedirectResponse(web or "/docs", status_code=307)

    return app


app = build_app()
