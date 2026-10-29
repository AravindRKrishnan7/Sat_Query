from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.presentation import (  # noqa: E402
    PresentationIdentityError,
    comparison_input_paths,
    compose_before_after_reveal,
)


class ComparisonPresentationTests(unittest.TestCase):
    def test_slider_uses_exact_verified_interval_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = root / "before.png"
            after = root / "after.png"
            Image.new("RGB", (4, 2), (10, 20, 30)).save(before)
            Image.new("RGB", (4, 2), (200, 210, 220)).save(after)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            timeline = {
                "observations": [
                    {
                        "id": "a",
                        "prepared_file": {"path": str(before), "sha256": digest(before)},
                    },
                    {
                        "id": "b",
                        "prepared_file": {"path": str(after), "sha256": digest(after)},
                    },
                ],
                "comparisons": [
                    {
                        "id": "comparison-01",
                        "before_observation": "a",
                        "after_observation": "b",
                    }
                ],
            }
            resolved_before, resolved_after = comparison_input_paths(
                timeline, "comparison-01"
            )
            reveal = compose_before_after_reveal(
                resolved_before, resolved_after, before_percent=50
            )
            self.assertEqual((10, 20, 30), reveal.getpixel((0, 0)))
            self.assertEqual((10, 20, 30), reveal.getpixel((1, 0)))
            self.assertEqual((200, 210, 220), reveal.getpixel((2, 0)))
            self.assertEqual((200, 210, 220), reveal.getpixel((3, 0)))

    def test_slider_rejects_changed_or_unknown_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.new("RGB", (2, 2), (0, 0, 0)).save(path)
            timeline = {
                "observations": [
                    {
                        "id": "a",
                        "prepared_file": {"path": str(path), "sha256": "wrong"},
                    }
                ],
                "comparisons": [
                    {
                        "id": "comparison-01",
                        "before_observation": "a",
                        "after_observation": "a",
                    }
                ],
            }
            with self.assertRaisesRegex(PresentationIdentityError, "hash mismatch"):
                comparison_input_paths(timeline, "comparison-01")
            with self.assertRaisesRegex(PresentationIdentityError, "unknown comparison"):
                comparison_input_paths(timeline, "missing")


if __name__ == "__main__":
    unittest.main()
