"""One-click report: renders from the stored 1.0 fixture, has count + warnings, no external URLs."""

import json
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image

from countonce.cli import main
from countonce.export.report import write_report

FIX = Path(__file__).parent / "fixtures" / "schemas"


def _setup(tmp_path: Path, version: str = "1.0", overlays: bool = True) -> Path:
    shutil.copy(FIX / f"v{version}" / "result.json", tmp_path / "result.json")
    if overlays:
        (tmp_path / "overlays").mkdir()
        Image.new("RGB", (64, 48), (10, 120, 30)).save(tmp_path / "overlays" / "frame_0.png")
    return tmp_path


def test_report_from_1_0_fixture(tmp_path):
    pytest.importorskip("matplotlib")
    out = _setup(tmp_path)
    doc_json = json.loads((out / "result.json").read_text())
    (path,) = write_report(out)
    doc = path.read_text()
    assert f">{doc_json['count']}<" in doc
    for w in doc_json["warnings"]:
        assert w["code"] in doc and w["fix"].split(".")[0][:20] in doc
    assert "data:image/jpeg;base64," in doc and "data:image/png;base64," in doc
    assert "sensitivity" in doc and "fake@0" in doc and "fixture" in doc
    assert not re.findall(r"(?:src|href)\s*=\s*['\"]?(?:https?:)?//", doc)
    assert "http://" not in doc.replace("http://www.w3.org", "")


def test_report_without_overlays_and_1_1(tmp_path):
    pytest.importorskip("matplotlib")
    out = _setup(tmp_path, "1.1", overlays=False)
    (path,) = write_report(out)
    assert "Annotated overlays" not in path.read_text()


def test_cli_report_and_pdf_skips_cleanly(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = _setup(tmp_path)
    assert main(["report", str(out), "--pdf"]) == 0
    assert (out / "report.html").is_file()
    assert main(["report", str(tmp_path / "missing")]) == 2
