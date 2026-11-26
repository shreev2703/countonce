"""MCP integration: pure count_objects contract, typed errors, lazy mcp import, packaging."""

import importlib
import importlib.metadata
import importlib.util
import tomllib
from pathlib import Path

import pytest
from integrations import mcp_server
from integrations.mcp_server import build_server, count_objects

from countonce.detect.base import FakeDetector
from tests.test_pipeline import OBJECTS, PlaneRecon, make_scene


def test_count_objects_with_fakes(tmp_path):
    img_dir, script = make_scene(tmp_path)
    images = sorted(str(p) for p in img_dir.iterdir())
    out = count_objects(
        images,
        "widgets",
        {"out_dir": str(tmp_path / "out"), "max_frames": 0},
        detector=FakeDetector(script),
        reconstructor=PlaneRecon(),
    )
    assert out["ok"] and out["count"] == len(OBJECTS)
    assert set(out) == {"ok", "count", "zero_count", "interval", "warnings", "result_path"}
    assert set(out["interval"]) == {"low", "high", "level", "method"}
    assert out["interval"]["low"] <= out["count"] <= out["interval"]["high"]
    assert isinstance(out["warnings"], list)
    assert Path(out["result_path"]).is_file()


def test_default_out_dir_is_temporary(tmp_path):
    img_dir, script = make_scene(tmp_path)
    out = count_objects(
        sorted(str(p) for p in img_dir.iterdir()),
        "widgets",
        {"max_frames": 0},
        detector=FakeDetector(script),
        reconstructor=PlaneRecon(),
    )
    assert out["ok"] and Path(out["result_path"]).is_file()


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_is_typed_error(tmp_path, prompt):
    out = count_objects([str(tmp_path)], prompt, None)
    assert not out["ok"] and out["error"]["code"] == "ERR_INVALID_INPUT"
    assert set(out["error"]) == {"code", "message", "tip", "retryable"}
    assert out["error"]["retryable"] is False


def test_bad_paths_and_inputs_are_typed_errors(tmp_path):
    out = count_objects([str(tmp_path / "nope.jpg")], "apples", None)
    assert out["error"]["code"] == "ERR_INVALID_INPUT" and "nope.jpg" in out["error"]["message"]
    assert count_objects([], "apples", None)["error"]["code"] == "ERR_INVALID_INPUT"
    bad_cfg = count_objects([str(tmp_path)], "apples", {"bogus": 1})
    assert bad_cfg["error"]["code"] == "ERR_INVALID_INPUT"


def test_unsupported_media_keeps_its_code(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    out = count_objects([str(f)], "apples", {"out_dir": str(tmp_path / "o")})
    assert not out["ok"] and out["error"]["code"] == "ERR_UNSUPPORTED_MEDIA"


def test_module_imports_without_mcp():
    importlib.reload(mcp_server)  # import-time must not need mcp
    if importlib.util.find_spec("mcp") is None:
        with pytest.raises(RuntimeError, match=r"countonce\[mcp\]"):
            build_server()


def test_server_registers_tool_when_mcp_present():
    pytest.importorskip("mcp")
    assert build_server() is not None


def test_mcp_entry_point_is_not_packaged():
    # Quarantined (PLAN §28.5): the console script is gone until the Moon tier starts.
    meta = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    assert "countonce-mcp" not in meta["project"]["scripts"]
    assert any(d.startswith("mcp") for d in meta["project"]["optional-dependencies"]["mcp"])
