"""Static check: the worker image's pip extra covers what the default gdino + vggt path imports."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker" / "worker.Dockerfile"
# Files behind build_detector("gdino") and the VGGT adapter (imports are lazy; scan whole files).
SOURCES = [
    "countonce/detect/groundingdino.py",
    "countonce/detect/_grounding.py",
    "countonce/recon/vggt.py",
]
# import name -> distribution name where they differ
DIST = {"PIL": "pillow", "yaml": "pyyaml", "cv2": "opencv-python-headless"}
# Optional alternative backend (the `groundingdino` package), guarded and not on the default path.
OPTIONAL = {"groundingdino"}


def _extra_name(dockerfile: str) -> str:
    m = re.search(r'pip install\s+"\.\[([^\]]+)\]"', dockerfile)
    assert m, 'Dockerfile has no `pip install ".[extra]"` line'
    return m.group(1)


def _dist_names(reqs: list[str]) -> set[str]:
    return {
        re.split(r"[\s\[<>=!~@;]", r.strip(), maxsplit=1)[0].lower().replace("_", "-") for r in reqs
    }


def _third_party_imports() -> set[str]:
    mods: set[str] = set()
    for rel in SOURCES:
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    return {
        m for m in mods if m not in sys.stdlib_module_names and m not in {"countonce", "__future__"}
    }


def test_dockerfile_extra_covers_default_path_imports() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extra = _extra_name(DOCKERFILE.read_text())
    covered = _dist_names(project["dependencies"]) | _dist_names(
        project["optional-dependencies"][extra]
    )
    missing = sorted(
        m
        for m in _third_party_imports() - OPTIONAL
        if DIST.get(m, m).lower().replace("_", "-") not in covered
    )
    assert not missing, f"extra [{extra}] lacks: {missing}"


def test_dockerfile_and_readme_name_the_same_extra() -> None:
    extra = _extra_name(DOCKERFILE.read_text())
    assert extra == "worker"
    assert f"[dev,{extra}]" in (ROOT / "README.md").read_text()


def test_vggt_is_pinned_and_git_is_installed() -> None:
    reqs = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"][
        "worker"
    ]
    assert any(r.startswith("vggt @ git+") and re.search(r"@[0-9a-f]{40}$", r) for r in reqs)
    assert re.search(r"\bgit\b", DOCKERFILE.read_text().split("pip install")[0])
