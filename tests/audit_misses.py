"""Explain why labeled flakes are missing from the current detector run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.verification.audit_misses import main


if __name__ == "__main__":
    raise SystemExit(main())
