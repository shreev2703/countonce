"""Alembic migrations on SQLite and the SQLAlchemy models against PLAN 14.3."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")

from alembic.autogenerate import compare_metadata  # noqa: E402
from alembic.migration import MigrationContext  # noqa: E402
from sqlalchemy import create_engine, inspect  # noqa: E402

from countonce.db import migrate  # noqa: E402
from countonce.db.models import EXTRA_COLUMNS, EXTRA_TABLES, PLAN_COLUMNS, Base  # noqa: E402
from countonce.jobs.store import SqliteJobStore  # noqa: E402

PLAN = Path(__file__).resolve().parents[1] / "docs" / "PLAN.md"


def _plan_tables() -> dict[str, set[str]]:
    """Parse the §14.3 table from PLAN.md (backticked names, bold markers stripped)."""
    text = PLAN.read_text()
    section = text[text.index("### 14.3 Database") : text.index("### 14.4")]
    out: dict[str, set[str]] = {}
    for line in section.splitlines():
        m = re.match(r"\| `(\w+)` \| (.*) \| (Strong|Moon)[^|]*\|$", line)
        if not m:
            continue
        cols = re.sub(r"\([^)]*\)", "", m.group(2)).replace("*", "").replace("`", "")
        out[m.group(1)] = {c.strip() for c in cols.split(",") if c.strip()}
    return out


def test_plan_columns_constant_matches_plan_md() -> None:
    plan = _plan_tables()
    assert len(plan) == 13  # every table in 14.3, Strong and Moon
    assert {t: set(c) for t, c in PLAN_COLUMNS.items()} == plan


def test_models_have_every_plan_column_and_only_declared_extras() -> None:
    tables = Base.metadata.tables
    assert set(PLAN_COLUMNS) <= set(tables)
    assert set(tables) - set(PLAN_COLUMNS) == set(EXTRA_TABLES)
    for name, cols in PLAN_COLUMNS.items():
        actual = {c.name for c in tables[name].columns}
        assert actual == set(cols) | set(EXTRA_COLUMNS.get(name, ())), name


def test_idempotency_key_is_unique_per_user() -> None:
    jobs = Base.metadata.tables["jobs"]
    uniques = {tuple(c.name for c in u.columns) for u in jobs.constraints if hasattr(u, "columns")}
    assert ("user_id", "idempotency_key") in uniques


def test_sqlite_store_schema_is_a_subset_of_models(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "q.db")
    insp = inspect(create_engine(f"sqlite:///{store.path}"))
    for table in insp.get_table_names():
        model_cols = {c.name for c in Base.metadata.tables[table].columns}
        assert {c["name"] for c in insp.get_columns(table)} <= model_cols, table


def test_upgrade_head_then_downgrade_base_on_sqlite(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    engine = create_engine(url)
    migrate.upgrade(url)
    tables = set(inspect(engine).get_table_names())
    assert set(Base.metadata.tables) <= tables
    with engine.connect() as conn:  # the migration and the models describe the same schema
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []
    migrate.downgrade(url)
    assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    migrate.upgrade(url)  # and it re-applies cleanly
    assert set(Base.metadata.tables) <= set(inspect(engine).get_table_names())


def test_single_initial_revision() -> None:
    versions = list((migrate.MIGRATIONS_DIR / "versions").glob("*.py"))
    assert len(versions) == 1


def test_pg_store_translates_placeholders() -> None:
    pytest.importorskip("psycopg")
    from countonce.jobs.pg_store import PostgresJobStore, to_pyformat

    assert to_pyformat("SELECT * FROM t WHERE a = ? AND b LIKE '50%'") == (
        "SELECT * FROM t WHERE a = %s AND b LIKE '50%%'"
    )
    assert "FOR UPDATE SKIP LOCKED" in PostgresJobStore._CLAIM_SQL
    assert "FOR UPDATE" not in SqliteJobStore._CLAIM_SQL
