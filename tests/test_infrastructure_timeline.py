from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.geolens import prepare_uploaded_image  # noqa: E402
from satquery.infrastructure import InfrastructureInterpretation  # noqa: E402
from satquery.infrastructure_timeline import (  # noqa: E402
    InfrastructureTimelineError,
    execute_infrastructure_timeline,
    timeline_request_fingerprint,
    validate_timeline_observations,
)


def image_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(output, format="PNG")
    return output.getvalue()


def interpretation() -> InfrastructureInterpretation:
    return InfrastructureInterpretation(
        original_request="Compare building and road changes over time",
        requested_classes=("building_change", "road_change"),
        general_change_requested=False,
        instance_count_requested=False,
        spatial_description_requested=False,
        evidence={
            "buildings": "building",
            "roads": "road",
            "general_change": None,
            "instance_count": None,
            "spatial_description": None,
        },
        provider="test",
        model="test",
        runtime={},
    )


def observations(count: int, dates: list[str] | None = None):
    raw = [image_bytes((index * 30, 80, 120)) for index in range(1, count + 1)]
    prepared = [
        prepare_uploaded_image(value, f"image {index}", resize=False)
        for index, value in enumerate(raw, start=1)
    ]
    values = dates or [f"202{index}-01-01" for index in range(1, count + 1)]
    result = validate_timeline_observations(
        [f"observation-{index}.png" for index in range(1, count + 1)],
        prepared,
        values,
    )
    return raw, result


class FakeRunner:
    def __init__(self, *, fail_calls: set[int] | None = None) -> None:
        self.calls = 0
        self.fail_calls = fail_calls or set()

    def __call__(self, before: Path, after: Path, output: Path) -> dict:
        self.calls += 1
        if self.calls in self.fail_calls:
            raise RuntimeError(f"deliberate failure {self.calls}")
        output.mkdir(parents=True, exist_ok=True)
        building_pixels = 10 if self.calls == 1 else 0
        caption = "some houses are built" if self.calls == 1 else "nothing changed"
        paths = {
            "indexed_mask": output / "class_mask.png",
            "color_mask_with_legend": output / "color_mask_with_legend.png",
            "overlay": output / "overlay.png",
            "comparison": output / "comparison.png",
            "caption": output / "caption.txt",
            "execution_record": output / "run.json",
        }
        for name, path in paths.items():
            if name not in {"caption", "execution_record"}:
                path.write_bytes(b"png")
        paths["caption"].write_text(caption + "\n", encoding="utf-8")
        record = {
            "status": "success",
            "inputs": {
                "before": {
                    "path": str(before.resolve()),
                    "sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
                },
                "after": {
                    "path": str(after.resolve()),
                    "sha256": hashlib.sha256(after.read_bytes()).hexdigest(),
                },
            },
            "model_outputs": {
                "mask_class_distribution": {
                    "0": {
                        "pixels": 100 - building_pixels,
                        "fraction": (100 - building_pixels) / 100,
                    },
                    "1": {"pixels": 0, "fraction": 0.0},
                    "2": {"pixels": building_pixels, "fraction": building_pixels / 100},
                },
                "caption": {"text": caption},
            },
            "outputs": {name: str(path.resolve()) for name, path in paths.items()},
        }
        paths["execution_record"].write_text(json.dumps(record), encoding="utf-8")
        return record


class TimelineValidationTests(unittest.TestCase):
    def test_two_three_and_four_images_sort_chronologically(self) -> None:
        for count in (2, 3, 4):
            with self.subTest(count=count):
                values = [f"202{index}-01-01" for index in range(count, 0, -1)]
                _, result = observations(count, values)
                self.assertEqual(
                    sorted(values),
                    [item.acquisition_date.isoformat() for item in result],
                )
                self.assertEqual(
                    list(range(count, 0, -1)),
                    [item.upload_order for item in result],
                )

    def test_missing_duplicate_and_invalid_counts_are_rejected(self) -> None:
        raw = image_bytes((1, 2, 3))
        prepared = prepare_uploaded_image(raw, "image", resize=False)
        with self.assertRaisesRegex(InfrastructureTimelineError, "between 2 and 4"):
            validate_timeline_observations(["one.png"], [prepared], ["2024-01-01"])
        with self.assertRaisesRegex(InfrastructureTimelineError, "required"):
            validate_timeline_observations(
                ["one.png", "two.png"], [prepared, prepared], ["2024-01-01", ""]
            )
        with self.assertRaisesRegex(InfrastructureTimelineError, "unique"):
            validate_timeline_observations(
                ["one.png", "two.png"],
                [prepared, prepared],
                ["2024-01-01", "2024-01-01"],
            )
        with self.assertRaisesRegex(InfrastructureTimelineError, "between 2 and 4"):
            validate_timeline_observations(
                [f"{index}.png" for index in range(5)],
                [prepared] * 5,
                [f"2024-01-{index + 1:02d}" for index in range(5)],
            )

    def test_fingerprint_invalidates_every_controlling_input(self) -> None:
        base = {
            "question": "compare",
            "upload_bytes": [b"a", b"b"],
            "date_values": ["2023-01-01", "2024-01-01"],
            "resize": False,
            "include_endpoint": False,
        }
        expected = timeline_request_fingerprint(**base)
        self.assertEqual(expected, timeline_request_fingerprint(**base))
        changes = (
            {**base, "question": "other"},
            {**base, "upload_bytes": [b"changed", b"b"]},
            {**base, "date_values": ["2022-01-01", "2024-01-01"]},
            {**base, "resize": True},
            {**base, "include_endpoint": True},
        )
        for changed in changes:
            self.assertNotEqual(expected, timeline_request_fingerprint(**changed))


class TimelineExecutionTests(unittest.TestCase):
    def execute(self, count: int, runner: FakeRunner, *, endpoint: bool = False) -> dict:
        _, items = observations(count)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return execute_infrastructure_timeline(
            parent_directory=Path(temporary.name) / "timeline",
            question="Compare building and road changes over time",
            fingerprint="fingerprint",
            interpretation=interpretation(),
            observations=items,
            resize_requested=False,
            include_endpoint=endpoint,
            comparison_runner=runner,
        )

    def test_interval_answers_use_only_that_intervals_evidence(self) -> None:
        result = self.execute(3, FakeRunner())
        self.assertEqual("success", result["status"])
        self.assertIn("houses were added", result["comparisons"][0]["answer"])
        self.assertIn(
            "did not identify building or road changes",
            result["comparisons"][1]["answer"],
        )
        self.assertEqual(1, result["comparisons"][1]["answer"].count("road changes"))
        self.assertNotEqual(
            result["comparisons"][0]["answer"],
            result["comparisons"][1]["answer"],
        )

    def test_partial_failure_preserves_and_continues_other_comparisons(self) -> None:
        result = self.execute(4, FakeRunner(fail_calls={2}))
        self.assertEqual("partial", result["status"])
        self.assertEqual(
            ["success", "failed", "success"],
            [item["status"] for item in result["comparisons"]],
        )
        self.assertTrue(Path(result["comparisons"][0]["execution_record"]).is_file())
        saved = json.loads(
            Path(result["outputs"]["timeline_record"]).read_text(encoding="utf-8")
        )
        self.assertEqual("partial", saved["status"])

    def test_optional_endpoint_is_separate_from_consecutive_comparisons(self) -> None:
        runner = FakeRunner()
        result = self.execute(4, runner, endpoint=True)
        self.assertEqual(4, runner.calls)
        self.assertEqual(
            ["consecutive", "consecutive", "consecutive", "endpoint"],
            [item["kind"] for item in result["comparisons"]],
        )
        self.assertIn("not a sum", Path(result["outputs"]["timeline_answer"]).read_text())

    def test_parent_record_preserves_original_and_prepared_hashes(self) -> None:
        result = self.execute(2, FakeRunner())
        for observation in result["observations"]:
            self.assertEqual(64, len(observation["original_upload"]["sha256"]))
            self.assertEqual(64, len(observation["prepared_file"]["sha256"]))
            self.assertTrue(observation["original_upload"]["filename"].endswith(".png"))
            self.assertTrue(result["comparisons"][0]["input_identity_verified"])


if __name__ == "__main__":
    unittest.main()
