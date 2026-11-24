"""Database URL helpers shared by Alembic, the Postgres store and the tests."""

from __future__ import annotations


def sqlalchemy_url(url: str) -> str:
    """``postgresql://`` / ``postgres://`` -> the psycopg 3 SQLAlchemy dialect; others unchanged."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def psycopg_url(url: str) -> str:
    """The plain libpq URL psycopg expects (no SQLAlchemy driver suffix)."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    return url
