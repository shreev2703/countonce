"""Write the FastAPI OpenAPI schema to ``web/openapi.json`` (input of ``npm run gen:api``).

Deterministic (sorted keys, no timestamps). ``--check`` exits 1 when the file is stale, so CI
can fail when the backend and the generated frontend types drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "web" / "openapi.json"


def render() -> str:
    from countonce.api.app import open_app

    with tempfile.TemporaryDirectory() as tmp:
        app = open_app(Path(tmp) / "jobs.db", Path(tmp) / "uploads")
        return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="fail if --out is out of date")
    args = ap.parse_args(argv)
    text = render()
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            print(f"{args.out} is stale; run scripts/export_openapi.py", file=sys.stderr)
            return 1
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
