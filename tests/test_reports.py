from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satquery.reports import (  # noqa: E402
    ReportBuildError,
    build_historical_pdf,
    build_infrastructure_pdf,
)


def _png(path: Path, colour: tuple[int, int, int]) -> str:
    Image.new("RGB", (48, 36), colour).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReportTests(unittest.TestCase):
    def _infrastructure_record(self, root: Path) -> dict:
        observation_root = root / "observations"
        comparison_root = root / "comparisons" / "one"
        observation_root.mkdir(parents=True)
        comparison_root.mkdir(parents=True)
        before = observation_root / "before.png"
        after = observation_root / "after.png"
        before_hash = _png(before, (30, 60, 90))
        after_hash = _png(after, (90, 60, 30))
        overlay = comparison_root / "overlay.png"
        _png(overlay, (120, 30, 30))
        execution = comparison_root / "run.json"
        execution.write_text(
            json.dumps(
                {
                    "inputs": {
                        "before": {"path": str(before), "sha256": before_hash},
                        "after": {"path": str(after), "sha256": after_hash},
                    },
                    "model_outputs": {
                        "caption": {"text": "some houses are newly built"},
                        "mask_class_distribution": {
                            "0": {"name": "background", "pixels": 100},
                            "1": {"name": "road_change", "pixels": 0},
                            "2": {"name": "building_change", "pixels": 20},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        answer = root / "timeline-answer.md"
        answer.write_text(
            "The model suggests that houses were added between the two observations.\n",
            encoding="utf-8",
        )
        timeline = root / "timeline.json"
        record = {
            "created_at_utc": "2026-10-29T00:00:00+00:00",
            "status": "partial",
            "question": "What building changes are predicted?",
            "request_fingerprint": "abc123",
            "execution_policy": {"endpoint_requested": False},
            "observations": [
                {
                    "id": "observation-01",
                    "chronological_order": 1,
                    "acquisition_date": "2023-01-01",
                    "original_upload": {
                        "filename": "before.png",
                        "sha256": "a" * 64,
                    },
                    "preparation": {"resampling": "none"},
                    "prepared_file": {"path": str(before), "sha256": before_hash},
                },
                {
                    "id": "observation-02",
                    "chronological_order": 2,
                    "acquisition_date": "2025-01-01",
                    "original_upload": {
                        "filename": "after.png",
                        "sha256": "b" * 64,
                    },
                    "preparation": {"resampling": "none"},
                    "prepared_file": {"path": str(after), "sha256": after_hash},
                },
            ],
            "comparisons": [
                {
                    "id": "comparison-01",
                    "kind": "consecutive",
                    "before_observation": "observation-01",
                    "after_observation": "observation-02",
                    "before_date": "2023-01-01",
                    "after_date": "2025-01-01",
                    "status": "success",
                    "answer": "The model suggests that houses were added.",
                    "execution_record": str(execution),
                    "overlay": str(overlay),
                },
                {
                    "id": "comparison-02",
                    "kind": "endpoint",
                    "before_observation": "observation-01",
                    "after_observation": "observation-02",
                    "before_date": "2023-01-01",
                    "after_date": "2025-01-01",
                    "status": "failed",
                    "error": "example later-comparison failure",
                },
            ],
            "outputs": {
                "timeline_answer": str(answer),
                "timeline_record": str(timeline),
            },
        }
        timeline.write_text(json.dumps(record), encoding="utf-8")
        return record

    def test_infrastructure_pdf_uses_saved_identity_and_preserves_partial_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self._infrastructure_record(root)
            output = build_infrastructure_pdf(record, root / "report.pdf")
            content = output.read_bytes()
            self.assertTrue(content.startswith(b"%PDF-"))
            self.assertGreater(len(content), 5_000)

    def test_infrastructure_pdf_rejects_prepared_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self._infrastructure_record(root)
            record["observations"][0]["prepared_file"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ReportBuildError, "recorded SHA-256"):
                build_infrastructure_pdf(record, root / "report.pdf")

    def test_historical_pdf_handles_missing_year_and_saved_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annual = root / "annual"
            differences = root / "differences"
            annual.mkdir()
            differences.mkdir()
            observation_records = []
            for year, colour in ((2021, (20, 90, 60)), (2023, (30, 120, 70))):
                rgb = annual / f"{year}-rgb.png"
                ndvi = annual / f"{year}-ndvi.png"
                _png(rgb, colour)
                _png(ndvi, (colour[1], colour[0], colour[2]))
                observation_records.append(
                    {
                        "year": year,
                        "status": "available",
                        "acquisition_date": f"{year}-01-10",
                        "item_id": f"sentinel-{year}",
                        "platform": "sentinel-2",
                        "valid_coverage": 0.9,
                        "outputs": {
                            "true_color_png": str(rgb),
                            "ndvi_png": str(ndvi),
                        },
                    }
                )
            observation_records.insert(
                1,
                {"year": 2022, "status": "unavailable", "reason": "quality rules not met"},
            )
            difference = differences / "2021-to-2023.png"
            _png(difference, (100, 100, 180))
            execution = root / "run.json"
            record = {
                "status": "partial",
                "request": {
                    "aoi": {
                        "bounds_wgs84": [76.34, 9.48, 76.36, 9.5],
                        "area_square_km": 4.878,
                    },
                    "years": [2021, 2022, 2023],
                },
                "annual_observations": observation_records,
                "adjacent_differences": [
                    {
                        "label": "2021-to-2022",
                        "status": "unavailable",
                        "reason": "2022 observation unavailable",
                    }
                ],
                "first_to_last_difference": {
                    "label": "2021-to-2023-first-last",
                    "status": "available",
                    "before_date": "2021-01-10",
                    "after_date": "2023-01-10",
                    "statistics": {"mean_difference": 0.02, "valid_pixels": 200},
                    "outputs": {"difference_png": str(difference)},
                },
                "provider": {"collection": "sentinel-2-c1-l2a"},
                "grid": {"resolution_metres": 10},
                "available_years": [2021, 2023],
                "missing_years": [2022],
                "common_valid_trend": {"coverage": 0.8, "sufficient": True},
                "outputs": {"execution_record": str(execution)},
            }
            execution.write_text(json.dumps(record), encoding="utf-8")
            output = build_historical_pdf(
                record,
                root / "report.pdf",
                original_request="Compare vegetation from 2021 to 2023.",
                answer="Average satellite-observed greenness was higher on the later date.",
                area_label="Alappuzha demo area",
            )
            content = output.read_bytes()
            self.assertTrue(content.startswith(b"%PDF-"))
            self.assertGreater(len(content), 5_000)


if __name__ == "__main__":
    unittest.main()
