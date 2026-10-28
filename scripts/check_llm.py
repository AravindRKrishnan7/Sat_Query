#!/usr/bin/env python3
"""Run a real, read-only GeoLens interpretation compatibility check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.language import (  # noqa: E402
    LanguageModelError,
    OllamaProvider,
    build_tool_plan,
    discover_ollama,
    interpret_request,
)
from satquery.historical import DEMO_AOI_LABEL, DEMO_BOUNDS, confirm_aoi  # noqa: E402


CASES = (
    (
        "Compare vegetation from 2021 to 2025.",
        {"analyses": ["vegetation"], "start_year": 2021, "end_year": 2025, "missing": []},
    ),
    (
        "Compare construction and roads here from 2021 to 2023.",
        {"analyses": ["construction", "roads"], "start_year": 2021, "end_year": 2023, "missing": []},
    ),
    (
        "Compare vegetation and construction here from 2022 to 2025.",
        {"analyses": ["vegetation", "construction"], "start_year": 2022, "end_year": 2025, "missing": []},
    ),
    (
        "Compare vegetation here.",
        {
            "analyses": ["vegetation"],
            "start_year": None,
            "end_year": None,
            "missing": ["start_year", "end_year"],
        },
    ),
    (
        "Compare vegetation in Kochi from 2021 to 2025.",
        {
            "analyses": ["vegetation"],
            "start_year": 2021,
            "end_year": 2025,
            "missing": ["named_location_confirmation"],
        },
    ),
)


def main() -> int:
    info = discover_ollama()
    if not info.available:
        print(json.dumps(info.record(), indent=2))
        return 2
    provider = OllamaProvider(info)
    outputs = []
    try:
        for text, expected in CASES:
            interpreted = interpret_request(provider, text)
            plan = build_tool_plan(
                interpreted,
                confirm_aoi(DEMO_BOUNDS),
                active_boundary_label=DEMO_AOI_LABEL,
                active_boundary_source="demo_preset",
            )
            actual = {
                "analyses": list(interpreted.analyses),
                "start_year": interpreted.start_year,
                "end_year": interpreted.end_year,
                "missing": list(plan.boundary_resolution.missing_information),
            }
            if actual != expected:
                raise RuntimeError(f"compatibility mismatch for {text!r}: {actual!r} != {expected!r}")
            outputs.append({"request": text, "interpretation": actual})
    except (LanguageModelError, RuntimeError) as exc:
        print(f"compatibility check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"provider": info.identifier, "cases": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
