from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.historical import (  # noqa: E402
    DEMO_BOUNDS,
    AnnualData,
    HistoricalNetworkError,
    HistoricalValidationError,
    build_analysis_grid,
    build_request,
    common_trend,
    confirm_aoi,
    load_asset,
    resolve_reflectance_scaling,
    run_historical,
    search_candidates,
    update_aoi_state,
)


def _item(year: int) -> dict:
    bands = [{"scale": 0.0001, "offset": -0.1}]
    return {
        "id": f"S2-test-{year}",
        "collection": "sentinel-2-c1-l2a",
        "properties": {
            "datetime": f"{year}-02-01T05:30:00Z",
            "eo:cloud_cover": 1.5,
            "s2:processing_baseline": "05.11",
            "platform": "sentinel-2b",
            "grid:code": "MGRS-43PFL",
        },
        "assets": {
            key: {"href": f"https://example.invalid/{year}/{key}.tif", "raster:bands": bands}
            for key in ("blue", "green", "red", "nir", "scl")
        },
    }


def _fake_asset_reader(item: dict, asset_key: str, grid):
    year = int(item["properties"]["datetime"][:4])
    if asset_key == "scl":
        values = np.where(grid.aoi_mask, 4, 0).astype(np.uint8)
        return values, grid.aoi_mask.copy(), {"asset": asset_key, "item_id": item["id"]}
    base = {"blue": 0.10, "green": 0.15, "red": 0.20, "nir": 0.55}[asset_key]
    if asset_key == "nir":
        base += (year - 2021) * 0.01
    values = np.full(grid.aoi_mask.shape, np.nan, dtype=np.float64)
    values[grid.aoi_mask] = base
    return values, grid.aoi_mask.copy(), {
        "asset": asset_key,
        "item_id": item["id"],
        "applied_scale": 0.0001,
        "applied_offset": -0.1,
        "scale_offset_source": "test fixture",
    }


class HistoricalValidationTests(unittest.TestCase):
    def test_aoi_must_be_confirmed_and_within_limit(self) -> None:
        with self.assertRaisesRegex(HistoricalValidationError, "confirm"):
            build_request(None, 2021, 2025, current_year=2026)
        with self.assertRaisesRegex(HistoricalValidationError, "demo limit"):
            confirm_aoi((76.0, 9.0, 77.0, 10.0))
        confirmed = confirm_aoi(DEMO_BOUNDS)
        self.assertLess(confirmed.area_square_km, 25.0)

    def test_changed_aoi_invalidates_confirmation_request_and_result(self) -> None:
        state = {
            "historical_draft_bounds": DEMO_BOUNDS,
            "historical_confirmed_aoi": object(),
            "historical_request": object(),
            "historical_result": object(),
            "assistant_plan": object(),
            "assistant_execution": object(),
            "historical_answer": object(),
        }
        changed = update_aoi_state(state, (76.341, 9.48, 76.361, 9.50))
        self.assertTrue(changed)
        self.assertNotIn("historical_confirmed_aoi", state)
        self.assertNotIn("historical_request", state)
        self.assertNotIn("historical_result", state)
        self.assertNotIn("assistant_plan", state)
        self.assertNotIn("assistant_execution", state)
        self.assertNotIn("historical_answer", state)
        self.assertFalse(update_aoi_state(state, (76.341, 9.48, 76.361, 9.50)))

    def test_invalid_dates_and_unsupported_analysis_are_rejected(self) -> None:
        aoi = confirm_aoi(DEMO_BOUNDS)
        with self.assertRaisesRegex(HistoricalValidationError, "later"):
            build_request(aoi, 2025, 2021, current_year=2026)
        with self.assertRaisesRegex(HistoricalValidationError, "integers"):
            build_request(aoi, 2021.5, 2025, current_year=2026)  # type: ignore[arg-type]
        with self.assertRaisesRegex(HistoricalValidationError, "limit"):
            build_request(aoi, 2020, 2025, current_year=2026)
        with self.assertRaisesRegex(HistoricalValidationError, "unsupported"):
            build_request(aoi, 2021, 2025, analysis="construction", current_year=2026)


class HistoricalScalingTests(unittest.TestCase):
    def test_collection_one_requires_item_and_cog_scaling_to_match(self) -> None:
        asset = {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}
        scale, offset, source = resolve_reflectance_scaling(
            "sentinel-2-c1-l2a", {}, asset, 0.0001, -0.1
        )
        self.assertEqual((scale, offset), (0.0001, -0.1))
        self.assertIn("independently matched", source)
        with self.assertRaisesRegex(HistoricalValidationError, "disagrees"):
            resolve_reflectance_scaling("sentinel-2-c1-l2a", {}, asset, 1.0, 0.0)
        with self.assertRaisesRegex(HistoricalValidationError, "explicit"):
            resolve_reflectance_scaling("sentinel-2-c1-l2a", {}, {}, 0.0001, -0.1)

    def test_legacy_representation_flag_controls_offset(self) -> None:
        asset = {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}
        self.assertEqual(
            resolve_reflectance_scaling(
                "sentinel-2-l2a", {"earthsearch:boa_offset_applied": True}, asset, 1.0, 0.0
            )[:2],
            (0.0001, 0.0),
        )
        self.assertEqual(
            resolve_reflectance_scaling(
                "sentinel-2-l2a", {"earthsearch:boa_offset_applied": False}, asset, 1.0, 0.0
            )[:2],
            (0.0001, -0.1),
        )
        with self.assertRaisesRegex(HistoricalValidationError, "ambiguous"):
            resolve_reflectance_scaling("sentinel-2-l2a", {}, asset, 1.0, 0.0)


class HistoricalCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.request = build_request(confirm_aoi(DEMO_BOUNDS), 2021, 2022, current_year=2026)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_search_network_failure_uses_only_matching_aoi_cache(self) -> None:
        def success(_url, _payload):
            return {"features": [_item(2021)]}

        items, status = search_candidates(self.request, 2021, self.root, transport=success)
        self.assertEqual(status, "retrieved")
        self.assertEqual(items[0]["id"], "S2-test-2021")

        def failure(_url, _payload):
            raise HistoricalNetworkError("offline")

        items, status = search_candidates(
            self.request, 2021, self.root, transport=failure, refresh=True
        )
        self.assertEqual(status, "cache_after_network_error")
        self.assertEqual(items[0]["id"], "S2-test-2021")

        other_request = build_request(
            confirm_aoi((76.35, 9.48, 76.37, 9.50)), 2021, 2022, current_year=2026
        )
        with self.assertRaises(HistoricalNetworkError):
            search_candidates(other_request, 2021, self.root, transport=failure, refresh=True)

    def test_asset_network_failure_with_and_without_matching_cache(self) -> None:
        grid = build_analysis_grid(self.request.aoi)
        item = _item(2021)
        first = load_asset(
            item, "red", grid, self.request.aoi.cache_key, self.root, reader=_fake_asset_reader
        )
        self.assertEqual(first.cache_status, "retrieved")

        def invalid_scaling(_item_value, _asset_key, _grid):
            raise HistoricalValidationError("scaling mismatch")

        with self.assertRaisesRegex(HistoricalValidationError, "scaling mismatch"):
            load_asset(
                item,
                "red",
                grid,
                self.request.aoi.cache_key,
                self.root,
                reader=invalid_scaling,
                refresh=True,
            )

        def failure(_item_value, _asset_key, _grid):
            raise HistoricalNetworkError("offline")

        cached = load_asset(
            item,
            "red",
            grid,
            self.request.aoi.cache_key,
            self.root,
            reader=failure,
            refresh=True,
        )
        self.assertEqual(cached.cache_status, "cache_after_network_error")
        with self.assertRaises(HistoricalNetworkError):
            load_asset(
                _item(2022),
                "red",
                grid,
                self.request.aoi.cache_key,
                self.root,
                reader=failure,
                refresh=True,
            )


class HistoricalAnalysisTests(unittest.TestCase):
    def test_missing_year_is_explicit_and_not_bridged_as_adjacent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = build_request(confirm_aoi(DEMO_BOUNDS), 2021, 2023, current_year=2026)

            def transport(_url, payload):
                year = int(payload["datetime"][:4])
                return {"features": [] if year == 2022 else [_item(year)]}

            record = run_historical(
                request,
                root / "outputs",
                root / "cache",
                transport=transport,
                asset_reader=_fake_asset_reader,
            )
            self.assertEqual(record["status"], "partial")
            self.assertEqual(record["missing_years"], [2022])
            self.assertEqual(
                [entry["status"] for entry in record["adjacent_differences"]],
                ["unavailable", "unavailable"],
            )
            self.assertTrue(record["first_to_last_difference"]["status"] == "available")

    def test_insufficient_common_valid_coverage_withholds_statistics(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        first_valid = np.array([[True, False], [False, False]])
        second_valid = np.array([[False, True], [False, False]])

        def observation(year: int, valid: np.ndarray) -> AnnualData:
            ndvi = np.full((2, 2), np.nan)
            ndvi[valid] = 0.5
            return AnnualData(year, f"{year}-02-01", str(year), ndvi, valid, np.empty((2, 2, 3)), valid, {})

        _common, record = common_trend(
            [observation(2021, first_valid), observation(2022, second_valid)], mask, 0.5
        )
        self.assertFalse(record["sufficient"])
        self.assertEqual(record["coverage"], 0.0)
        self.assertEqual(record["annual_statistics"], [])


if __name__ == "__main__":
    unittest.main()
