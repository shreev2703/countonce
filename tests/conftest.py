"""Session guard: tests must never write into the checkout's ``results/`` or ``docs/``.

Most of ``results/`` is git-ignored, so a test that overwrites a real result file leaves no
trace in ``git status``. The baseline is taken when this file is imported (before collection,
so import-time writes count too) and compared after the last test. Point every output path of
a script or CLI under test at ``tmp_path``.

Anything else writing to those directories while the suite runs trips the guard as well.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GUARDED = ("results", "docs")


def _snapshot() -> dict[str, tuple[int, int]]:
    """``{repo-relative path: (mtime_ns, size)}`` for every file under the guarded dirs."""
    snap: dict[str, tuple[int, int]] = {}
    for top in _GUARDED:
        for path in (_ROOT / top).rglob("*"):
            if path.is_file():
                st = path.stat()
                snap[path.relative_to(_ROOT).as_posix()] = (st.st_mtime_ns, st.st_size)
    return snap


_BASELINE = _snapshot()


@pytest.fixture(scope="session", autouse=True)
def _guard_results_and_docs() -> Iterator[None]:
    yield
    now = _snapshot()
    leaks = [
        f"  {'appeared' if p not in _BASELINE else 'removed' if p not in now else 'changed'}: {p}"
        for p in sorted(_BASELINE.keys() | now.keys())
        if _BASELINE.get(p) != now.get(p)
    ]
    if leaks:
        pytest.fail(
            "files under results/ or docs/ changed during the test run. A test must write to "
            "tmp_path instead; if another process (e.g. a running benchmark) wrote them, "
            "leave them alone and rerun once it is idle:\n" + "\n".join(leaks),
            pytrace=False,
        )
