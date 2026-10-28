#!/usr/bin/env python3
"""Run the documented GeoLens historical vegetation demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.historical import (  # noqa: E402
    DEMO_BOUNDS,
    HistoricalNetworkError,
    HistoricalValidationError,
    build_request,
    confirm_aoi,
    run_historical,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Alappuzha historical NDVI demo.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=DEMO_BOUNDS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "geolens" / "historical",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "geolens-historical",
    )
    args = parser.parse_args()
    try:
        request = build_request(confirm_aoi(args.bounds), args.start_year, args.end_year)
        record = run_historical(request, args.output_dir, args.cache_dir)
    except (HistoricalValidationError, HistoricalNetworkError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": record["status"],
        "bounds": record["request"]["aoi"]["bounds_wgs84"],
        "available_years": record["available_years"],
        "missing_years": record["missing_years"],
        "common_valid_trend": record["common_valid_trend"],
        "execution_record": record["outputs"]["execution_record"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
