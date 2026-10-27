#!/usr/bin/env python3
"""Run the SatQuery Sentinel-2 Level-2A NDVI comparison."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.vegetation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
