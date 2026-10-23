from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.change_agent import InputValidationError, colorize_mask, save_outputs, validate_pair


class InputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def save(self, name: str, mode: str = "RGB", size: tuple[int, int] = (256, 256)) -> Path:
        path = self.root / name
        Image.new(mode, size).save(path)
        return path

    def test_accepts_matching_rgb_pair(self) -> None:
        before = self.save("before.png")
        after = self.save("after.png")
        images = validate_pair(before, after)
        self.assertEqual([image.size for image in images], [(256, 256), (256, 256)])

    def test_rejects_wrong_dimensions(self) -> None:
        before = self.save("before.png", size=(512, 256))
        after = self.save("after.png")
        with self.assertRaisesRegex(InputValidationError, "exactly 256x256"):
            validate_pair(before, after)

    def test_rejects_non_rgb(self) -> None:
        before = self.save("before.png", mode="L")
        after = self.save("after.png")
        with self.assertRaisesRegex(InputValidationError, "three-channel RGB"):
            validate_pair(before, after)

    def test_rejects_unreadable_file(self) -> None:
        before = self.root / "before.png"
        before.write_text("not an image", encoding="utf-8")
        after = self.save("after.png")
        with self.assertRaisesRegex(InputValidationError, "unreadable"):
            validate_pair(before, after)


class OutputContractTests(unittest.TestCase):
    def test_color_mask_preserves_expected_dimensions(self) -> None:
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[0, 0] = 1
        mask[0, 1] = 2
        color = np.asarray(colorize_mask(mask))
        self.assertEqual(color.shape, (256, 256, 3))
        self.assertEqual(tuple(color[0, 0]), (255, 255, 0))
        self.assertEqual(tuple(color[0, 1]), (255, 0, 0))

    def test_rejects_unknown_mask_class(self) -> None:
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[0, 0] = 3
        with self.assertRaisesRegex(ValueError, "unsupported labels"):
            colorize_mask(mask)

    def test_saved_output_contract(self) -> None:
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[32:64, 48:80] = 2
        before = Image.new("RGB", (256, 256), "white")
        after = Image.new("RGB", (256, 256), "gray")
        with tempfile.TemporaryDirectory() as directory:
            paths = save_outputs(Path(directory), before, after, mask, "test caption")
            expected = {
                "indexed_mask",
                "color_mask_with_legend",
                "overlay",
                "comparison",
                "caption",
                "execution_record",
            }
            self.assertEqual(set(paths), expected)
            with Image.open(paths["indexed_mask"]) as indexed:
                self.assertEqual(indexed.mode, "L")
                self.assertEqual(indexed.size, (256, 256))
                self.assertEqual(set(np.unique(np.asarray(indexed)).tolist()), {0, 2})
            with Image.open(paths["color_mask_with_legend"]) as color:
                self.assertEqual((color.mode, color.size), ("RGB", (256, 326)))
            with Image.open(paths["overlay"]) as overlay:
                self.assertEqual((overlay.mode, overlay.size), ("RGB", (256, 256)))
            with Image.open(paths["comparison"]) as comparison:
                self.assertEqual((comparison.mode, comparison.size), ("RGB", (1024, 350)))
            caption = Path(paths["caption"]).read_text(encoding="utf-8")
            self.assertIn("not derived from the predicted mask", caption)
            self.assertIn("test caption", caption)
            self.assertFalse(Path(paths["execution_record"]).exists())


if __name__ == "__main__":
    unittest.main()
