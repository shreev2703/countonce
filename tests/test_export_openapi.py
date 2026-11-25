"""web/openapi.json (input of the frontend's generated API types) must match the backend."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts/export_openapi.py")
assert spec and spec.loader
export_openapi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_openapi)


def test_committed_openapi_is_current():
    assert export_openapi.main(["--check"]) == 0, "run: python scripts/export_openapi.py"


def test_operation_ids_are_unique():
    import json

    ids = [
        op["operationId"]
        for ops in json.loads(export_openapi.render())["paths"].values()
        for op in ops.values()
    ]
    assert len(ids) == len(set(ids))
