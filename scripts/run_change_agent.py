#!/usr/bin/env python3
"""Run the SatQuery Change-Agent adapter from a source checkout."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.change_agent import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
