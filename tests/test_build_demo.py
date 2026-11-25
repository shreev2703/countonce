"""M13: demo exporter writes valid assets and the version check fails on mismatch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from countonce.export import demo
from countonce.export.demo import DemoVersionError, build_scene, check_demo, read_points_bin

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "examples" / "tiny_scene"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    build_scene(
        TINY, out, scene_id="tiny", prompt="red sphere", label="synthetic test scene",
        gt_count=5, extra_config={"dedup_threshold": 0}, repo_root=ROOT,
    )  # fmt: skip
    return out


def test_assets_written_and_valid(built: Path) -> None:
    d = built / "tiny"
    for name in ("result.json", "markers.json", "points.bin", "evidence.json", "provenance.json"):
        assert (d / name).is_file()
    man = json.loads((d / "manifest.json").read_text())
    assert man["label"] == "synthetic test scene"
    assert man["count"] == 5 == man["gt_count"]
    assert len(list((d / "overlays").iterdir())) == 5 == len(list((d / "thumbs").iterdir()))
    xyz, rgb = read_points_bin(d / "points.bin")
    assert len(xyz) == len(rgb) == man["n_points"] > 0
    assert json.loads((d / "provenance.json").read_text())["synthetic"] is True
    assert check_demo(built, ROOT) == ["tiny"]


def test_committed_demo_assets_are_current() -> None:
    """The CI check: fails if the committed assets were built for another schema version."""
    assert "tiny" in check_demo(ROOT / "web" / "public" / "demo", ROOT)


def test_stale_manifest_version_fails(built: Path, tmp_path: Path) -> None:
    shutil.copytree(built, tmp_path / "d")
    mf = tmp_path / "d" / "tiny" / "manifest.json"
    man = json.loads(mf.read_text())
    man["schema_version"] = "1.0"
    mf.write_text(json.dumps(man))
    with pytest.raises(DemoVersionError, match="rebuild"):
        check_demo(tmp_path / "d", ROOT)


def test_tampered_file_fails(built: Path, tmp_path: Path) -> None:
    shutil.copytree(built, tmp_path / "d")
    with (tmp_path / "d" / "tiny" / "evidence.json").open("a") as fh:
        fh.write(" ")
    with pytest.raises(DemoVersionError, match="hash differs"):
        check_demo(tmp_path / "d", ROOT)


def test_backend_web_version_mismatch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = tmp_path / demo.ZOD_SCHEMA_FILE
    web.parent.mkdir(parents=True)
    web.write_text('export const CURRENT_VERSION = "9.9";\n')
    with pytest.raises(DemoVersionError, match="mismatch"):
        demo.assert_versions_match(tmp_path)
    assert demo.assert_versions_match(ROOT) == demo.CURRENT_VERSION
