from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.geolens import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    GeoLensInputError,
    GeoLensResultError,
    create_unique_run_directory,
    load_execution_record,
    prepare_uploaded_image,
    save_prepared_pair,
    validate_result_record,
)


def image_bytes(size: tuple[int, int], mode: str = "RGB", file_format: str = "PNG") -> bytes:
    color = (20, 80, 140) if mode == "RGB" else (20, 80, 140, 255)
    image = Image.new(mode, size, color)
    output = io.BytesIO()
    image.save(output, format=file_format)
    return output.getvalue()


class UploadPreparationTests(unittest.TestCase):
    def test_accepts_exact_rgb_png_without_modification(self) -> None:
        prepared = prepare_uploaded_image(image_bytes((256, 256)), "before", resize=False)
        self.assertEqual(prepared.image.size, (256, 256))
        self.assertEqual(prepared.image.mode, "RGB")
        self.assertEqual(prepared.original_format, "PNG")
        self.assertFalse(prepared.resized)

    def test_resize_is_explicit_and_recorded(self) -> None:
        with self.assertRaisesRegex(GeoLensInputError, "requires exactly 256x256"):
            prepare_uploaded_image(image_bytes((512, 384), file_format="JPEG"), "after", resize=False)
        prepared = prepare_uploaded_image(
            image_bytes((512, 384), file_format="JPEG"), "after", resize=True
        )
        self.assertEqual(prepared.original_size, (512, 384))
        self.assertEqual(prepared.image.size, (256, 256))
        self.assertTrue(prepared.resized)

    def test_rejects_non_rgb_and_oversized_upload(self) -> None:
        with self.assertRaisesRegex(GeoLensInputError, "three-channel RGB"):
            prepare_uploaded_image(image_bytes((256, 256), mode="RGBA"), "before", resize=False)
        with self.assertRaisesRegex(GeoLensInputError, "exceeds the 10 MB"):
            prepare_uploaded_image(b"x" * (MAX_UPLOAD_BYTES + 1), "before", resize=False)

    def test_internal_paths_are_unique_and_do_not_use_upload_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = create_unique_run_directory(root, "run")
            second = create_unique_run_directory(root, "run")
            self.assertNotEqual(first, second)
            prepared = prepare_uploaded_image(image_bytes((256, 256)), "before", resize=False)
            before_path, after_path = save_prepared_pair(prepared, prepared, first)
            self.assertNotEqual(before_path.name, after_path.name)
            self.assertRegex(before_path.name, r"^before-[0-9a-f]{32}\.png$")
            self.assertRegex(after_path.name, r"^after-[0-9a-f]{32}\.png$")
            with Image.open(before_path) as image:
                self.assertEqual((image.mode, image.size), ("RGB", (256, 256)))


class ResultHandlingTests(unittest.TestCase):
    def test_validates_and_loads_successful_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "comparison.png"
            output.write_bytes(b"png fixture")
            record = {"status": "success", "outputs": {"comparison": str(output)}}
            record_path = root / "run.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(
                validate_result_record(record, ("comparison",))["comparison"], output.resolve()
            )
            self.assertEqual(load_execution_record(record_path), record)

    def test_rejects_missing_or_unsuccessful_results(self) -> None:
        with self.assertRaisesRegex(GeoLensResultError, "successful"):
            validate_result_record({"status": "failed"}, ("comparison",))
        with self.assertRaisesRegex(GeoLensResultError, "does not exist"):
            validate_result_record(
                {"status": "success", "outputs": {"comparison": "/missing/result.png"}},
                ("comparison",),
            )


if __name__ == "__main__":
    unittest.main()
