"""Fail when pytest skipped a test for a reason outside the allow-list.

Usage: pytest -rs | tee log; python scripts/check_skips.py log
Reads a pytest ``-rs`` log (file argument or stdin) and checks each
``SKIPPED [n] file:line: reason`` line.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED = ("pycolmap", "mcp", "cuda", "gpu", "countonce_pg_url")
_LINE = re.compile(r"^SKIPPED \[(\d+)\] (\S+): (.*)$")


def check(text: str) -> tuple[int, list[str]]:
    """Return (total skips, offending lines)."""
    total = 0
    bad: list[str] = []
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        total += int(m.group(1))
        reason = m.group(3).lower()
        if not any(a in reason for a in ALLOWED):
            bad.append(line.strip())
    return total, bad


def main(argv: list[str]) -> int:
    text = Path(argv[1]).read_text() if len(argv) > 1 else sys.stdin.read()
    total, bad = check(text)
    print(f"{total} skipped; {len(bad)} outside allow-list {ALLOWED}")
    for b in bad:
        print("  UNEXPECTED:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
