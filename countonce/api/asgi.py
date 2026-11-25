"""Module-level ASGI app for serverless hosts (Vercel entrypoint ``countonce.api.asgi:app``).

Uses Postgres when ``COUNTONCE_PG_URL`` is set. Otherwise it falls back to a SQLite file under the
system temp dir, which is ephemeral on serverless hosts: fine for a smoke check, not for real jobs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI

from countonce.api.app import open_app


def build_app() -> FastAPI:
    root = Path(os.environ.get("COUNTONCE_DATA_DIR", tempfile.gettempdir())) / "countonce"
    root.mkdir(parents=True, exist_ok=True)
    kind = "postgres" if os.environ.get("COUNTONCE_PG_URL") else "sqlite"
    app = open_app(root / "jobs.db", root / "uploads", store_kind=kind)

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def index() -> dict[str, str]:
        """Landing probe so the bare production URL answers 200 (GET and HEAD)."""
        return {"service": "countonce", "health": "/health", "docs": "/docs"}

    return app


app = build_app()
