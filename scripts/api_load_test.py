"""HTTP API load test with fake workers. Same as ``countonce loadtest-api``."""

import argparse
import sys
from pathlib import Path

from countonce.loadtest_api import ApiLoadConfig, format_summary, run_api_load


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/loadtest_api.yaml"))
    args = ap.parse_args(argv)
    try:
        cfg = ApiLoadConfig.from_yaml(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"api_load_test: error: {exc}", file=sys.stderr)
        return 2
    summary = run_api_load(cfg)
    print(format_summary(summary))
    return 0 if summary["reconciled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
