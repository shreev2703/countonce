"""Fail when code reads an environment variable that is missing from ``.env.example``."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("countonce", "bench", "synth", "scripts", "examples")
# Variables of the OS or tooling that are not project configuration.
IGNORED = {"PATH", "HOME"}


def documented() -> set[str]:
    text = (ROOT / ".env.example").read_text()
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, flags=re.M))


def _is_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _read_vars(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    consts = {
        t.id: n.value.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
        for t in n.targets
        if isinstance(t, ast.Name)
    }

    def resolve(arg: ast.AST) -> str | None:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.Name):
            return consts.get(arg.id)
        return None

    found: set[str] = set()
    for n in ast.walk(tree):
        arg: ast.AST | None = None
        if isinstance(n, ast.Subscript) and _is_environ(n.value):
            arg = n.slice
        elif isinstance(n, ast.Call) and n.args:
            f = n.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr in {"get", "pop", "setdefault"}
                and _is_environ(f.value)
            ) or (
                isinstance(f, ast.Attribute)
                and f.attr == "getenv"
                and isinstance(f.value, ast.Name)
                and f.value.id == "os"
            ):
                arg = n.args[0]
            elif isinstance(f, ast.Name) and f.id == "ProviderInfo":
                arg = n.args[0]  # API-key variable read indirectly via os.environ[env_var]
        name = resolve(arg) if arg is not None else None
        if name:
            found.add(name)
    return found


def code_env_vars() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for d in SCAN_DIRS:
        for py in sorted((ROOT / d).rglob("*.py")):
            for name in _read_vars(py):
                out.setdefault(name, set()).add(str(py.relative_to(ROOT)))
    return out


def test_every_env_var_read_in_code_is_in_env_example() -> None:
    used = code_env_vars()
    assert used, "scanner found no environment variables; it is probably broken"
    missing = {k: sorted(v) for k, v in used.items() if k not in documented() | IGNORED}
    assert not missing, f"add these to .env.example: {missing}"


def test_known_variables_are_detected() -> None:
    used = code_env_vars()
    for name in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    ):
        assert name in used


def test_scanner_catches_an_undocumented_variable(tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_text('import os\nX = "FOO_BAR"\na = os.environ.get(X)\nb = os.getenv("BAZ")\n')
    assert _read_vars(f) == {"FOO_BAR", "BAZ"}


def test_env_example_has_no_real_secrets() -> None:
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "KEY" in line and "=" in line:
            assert line.split("=", 1)[1].startswith("your-")


def test_env_example_is_not_git_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text()
    assert "!.env.example" in ignore
