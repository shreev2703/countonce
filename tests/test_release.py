"""HF release scaffolding: Space handler, dataset layout/card, model card. No network."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

from countonce import release
from countonce.cli import main
from countonce.config import Config
from countonce.detect.base import FakeDetector
from tests.test_pipeline import PlaneRecon, make_scene

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("network call attempted")

    try:
        import requests

        monkeypatch.setattr(requests.sessions.Session, "request", boom)
    except ImportError:
        pass
    monkeypatch.setattr(socket.socket, "connect", boom)
    try:
        import huggingface_hub

        for name in ("HfApi", "hf_hub_download", "upload_folder", "snapshot_download"):
            if hasattr(huggingface_hub, name):
                monkeypatch.setattr(huggingface_hub, name, boom)
    except ImportError:
        pass


def _load_app():
    spec = importlib.util.spec_from_file_location("space_app", ROOT / "space" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["space_app"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_handler_with_fake_backends_without_gradio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "gradio", None)  # any import of gradio raises ImportError
    app = _load_app()
    img_dir, script = make_scene(tmp_path)
    images = sorted(img_dir.glob("*.png"))
    cfg = Config(out_dir=tmp_path / "out")
    text, overlays, raw = app.count_handler(
        [str(p) for p in images],
        "widgets",
        config=cfg,
        detector=FakeDetector(script),
        reconstructor=PlaneRecon(),
        workdir=tmp_path / "out",
    )
    doc = json.loads(raw)
    assert text.startswith(f"{doc['count']} widgets")
    assert doc["count"] > 0
    assert len(overlays) == len(images) and all(Path(o).is_file() for o in overlays)


def test_handler_input_errors(tmp_path: Path) -> None:
    app = _load_app()
    assert app.count_handler([], "x")[0].startswith("Upload")
    assert app.count_handler(["a.png"], "  ")[0].startswith("Enter")
    text, ov, raw = app.count_handler(
        [str(tmp_path / "missing.png")],
        "x",
        workdir=tmp_path / "o",
        detector=FakeDetector({}),
        reconstructor=PlaneRecon(),
    )
    assert text.startswith("Error") and ov == [] and "error" in json.loads(raw)


def test_space_files() -> None:
    readme = (ROOT / "space" / "README.md").read_text()
    assert readme.startswith("---\n") and "sdk: gradio" in readme and "app_file: app.py" in readme
    assert "gradio" in (ROOT / "space" / "requirements.txt").read_text()


def _make_shards(path: Path) -> None:
    pytest.importorskip("pyarrow")
    from synth.dag import proxy_scene
    from synth.recipe import make_recipe
    from synth.shards import pack_shards, scene_sample

    pack_shards(
        (
            scene_sample(
                f"scene_{s:06d}",
                proxy_scene(make_recipe(s, n_objects=5, n_views=3)),
                {"split": "train" if s < 3 else "val"},
            )
            for s in range(5)
        ),
        path,
    )


def test_synth_dataset_layout_and_card(tmp_path: Path) -> None:
    from synth.recipe import RECIPE_VERSION
    from synth.shards import read_shards

    _make_shards(tmp_path / "src")
    out = tmp_path / "ds"
    rc = main(
        [
            "release",
            "synth-dataset",
            "--out",
            str(out),
            "--shards",
            str(tmp_path / "src"),
            "--n-scenes",
            "2",
            "--license",
            "cc-by-4.0",
        ]
    )
    assert rc == 0
    assert (out / "data" / "index.json").is_file()
    assert len(list(read_shards(out / "data"))) == 2
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-4.0\n")
    assert f'countonce_recipe_version: "{RECIPE_VERSION}"' in card
    assert "countonce_n_scenes: 2" in card
    assert "Composition" in card and "TODO: fill from results" in card
    assert card.count("\n# ") == 0 or card.count("# CountOnce-Synth") == 1


def test_synth_dataset_split_and_errors(tmp_path: Path) -> None:
    _make_shards(tmp_path / "src")
    info = release.build_synth_dataset(tmp_path / "v", tmp_path / "src", split="val")
    assert info["n_scenes"] == 2
    with pytest.raises(release.ReleaseError):
        release.build_synth_dataset(tmp_path / "x", tmp_path / "nope")
    assert (
        main(
            [
                "release",
                "synth-dataset",
                "--out",
                str(tmp_path / "y"),
                "--shards",
                str(tmp_path / "nope"),
            ]
        )
        == 2
    )


def _registry(tmp_path: Path, metrics: dict | None = None) -> Path:
    from countonce.match.registry import Registry

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    root = tmp_path / "models"
    Registry(root).register(
        ckpt, dataset_version="d1", recipe_version="1.0", metrics=metrics or {}, git_sha="abc123"
    )
    return root


def test_model_card_todo_without_results(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    out = tmp_path / "card.md"
    assert main(["release", "model-card", "--root", str(root), "--out", str(out)]) == 0
    card = out.read_text()
    assert "Version: `v001`" in card and "abc123" in card and "d1" in card
    assert card.startswith("---\nlicense: other\n")
    assert "| Count MAE | TODO: fill from results |" in card


def test_model_card_uses_results_file(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    res = tmp_path / "r.json"
    res.write_text(json.dumps({"count_mae": 1.25, "assoc_f1": 0.9}))
    out = tmp_path / "card.md"
    assert (
        main(
            ["release", "model-card", "--root", str(root), "--out", str(out), "--results", str(res)]
        )
        == 0
    )
    card = out.read_text()
    assert "| Count MAE | 1.25 |" in card and "| Association pairwise F1 | 0.9 |" in card
    assert "| MAPE | TODO: fill from results |" in card


def test_model_card_empty_registry(tmp_path: Path) -> None:
    assert (
        main(
            [
                "release",
                "model-card",
                "--root",
                str(tmp_path / "none"),
                "--out",
                str(tmp_path / "c.md"),
            ]
        )
        == 2
    )
