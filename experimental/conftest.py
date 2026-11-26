"""Quarantined Moon-tier modules (PLAN §28.5) are importable as top-level ``llm`` and
``integrations`` only while running ``pytest experimental``; they are not in the package."""

import sys
from pathlib import Path

HERE = Path(__file__).parent
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
