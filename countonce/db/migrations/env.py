"""Alembic environment. URL precedence: ``-x url=...``, ``sqlalchemy.url`` in the config
(``migrate()`` sets it), then ``COUNTONCE_PG_URL``."""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine, pool

from countonce.db.models import Base
from countonce.db.url import sqlalchemy_url

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    url = context.get_x_argument(as_dictionary=True).get("url")
    url = url or config.get_main_option("sqlalchemy.url") or os.environ.get("COUNTONCE_PG_URL")
    if not url:
        raise RuntimeError("no database URL: pass -x url=..., or set COUNTONCE_PG_URL")
    return sqlalchemy_url(url)


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
