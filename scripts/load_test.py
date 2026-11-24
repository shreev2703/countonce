"""Queue load test with fake workers. Same as ``countonce loadtest``."""

import argparse
import sys
from pathlib import Path

from countonce.loadtest import LoadTestConfig, format_summary, run_load_test


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/loadtest.yaml"))
    args = ap.parse_args(argv)
    try:
        cfg = LoadTestConfig.from_yaml(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"load_test: error: {exc}", file=sys.stderr)
        return 2
    summary = run_load_test(cfg)
    print(format_summary(summary))
    return 0 if summary["reconciled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
