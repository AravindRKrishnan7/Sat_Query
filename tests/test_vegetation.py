from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import rasterio
    from rasterio.transform import Affine
except ModuleNotFoundError:  # A clear skipped suite is preferable before dependencies are installed.
    rasterio = None

from satquery.vegetation import (  # noqa: E402
    FLOAT_NODATA,
    VegetationValidationError,
    calculate_ndvi,
    classify_delta,
    run,
)


class CalculationTests(unittest.TestCase):
    def test_known_ndvi(self) -> None:
        red = np.asarray([[0.2]], dtype=np.float32)
        nir = np.asarray([[0.6]], dtype=np.float32)
        _, _, ndvi, valid = calculate_ndvi(red, nir, 1.0, 0.0, 1.0, 0.0, np.ones((1, 1), bool), 1e-6)
        self.assertTrue(valid[0, 0])
        self.assertAlmostEqual(ndvi[0, 0], 0.5)

    def test_scale_and_offset_are_applied(self) -> None:
        red = np.asarray([[3000]], dtype=np.uint16)
        nir = np.asarray([[7000]], dtype=np.uint16)
        scaled_red, scaled_nir, ndvi, _ = calculate_ndvi(
            red, nir, 0.0001, -0.1, 0.0001, -0.1, np.ones((1, 1), bool), 1e-6
        )
        self.assertAlmostEqual(scaled_red[0, 0], 0.2)
        self.assertAlmostEqual(scaled_nir[0, 0], 0.6)
        self.assertAlmostEqual(ndvi[0, 0], 0.5)

    def test_offset_applied_provider_values_are_not_offset_twice(self) -> None:
        red = np.asarray([[900]], dtype=np.uint16)
        nir = np.asarray([[1100]], dtype=np.uint16)
        _, _, correct, correct_valid = calculate_ndvi(
            red, nir, 0.0001, 0.0, 0.0001, 0.0, np.ones((1, 1), bool), 1e-6
        )
        _, _, doubled, doubled_valid = calculate_ndvi(
            red, nir, 0.0001, -0.1, 0.0001, -0.1, np.ones((1, 1), bool), 1e-6
        )
        self.assertTrue(correct_valid[0, 0])
        self.assertAlmostEqual(correct[0, 0], 0.1)
        self.assertFalse(doubled_valid[0, 0])
        self.assertTrue(np.isnan(doubled[0, 0]))

    def test_integer_inputs_convert_before_subtraction(self) -> None:
        red = np.asarray([[6000]], dtype=np.uint16)
        nir = np.asarray([[2000]], dtype=np.uint16)
        _, _, ndvi, _ = calculate_ndvi(red, nir, 0.0001, 0.0, 0.0001, 0.0, np.ones((1, 1), bool), 1e-6)
        self.assertAlmostEqual(ndvi[0, 0], -0.5)

    def test_zero_and_near_zero_denominators_are_invalid(self) -> None:
        red = np.asarray([[1.0, 1.0]], dtype=np.float64)
        nir = np.asarray([[-1.0, -1.0 + 1e-8]], dtype=np.float64)
        _, _, ndvi, valid = calculate_ndvi(red, nir, 1.0, 0.0, 1.0, 0.0, np.ones((1, 2), bool), 1e-6)
        self.assertEqual(valid.tolist(), [[False, False]])
        self.assertTrue(np.isnan(ndvi).all())

    def test_threshold_boundaries_are_stable(self) -> None:
        delta = np.asarray([[-0.1001, -0.1, 0.0, 0.1, 0.1001, np.nan]])
        self.assertEqual(
            classify_delta(delta, 0.1),
            {"increased": 1, "decreased": 1, "approximately_stable": 3},
        )


@unittest.skipIf(rasterio is None, "Rasterio is not installed")
class RasterWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.transform = Affine(10, 0, 500000, 0, -10, 2000000)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_raster(
        self,
        name: str,
        values: np.ndarray,
        *,
        transform=None,
        nodata: int | float = 0,
        band_id: str | None = None,
    ) -> Path:
        path = self.root / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            dtype=values.dtype,
            crs="EPSG:32643",
            transform=transform or self.transform,
            nodata=nodata,
        ) as dataset:
            dataset.write(values, 1)
            if band_id:
                dataset.set_band_description(1, band_id)
                dataset.update_tags(1, BAND_ID=band_id)
        return path

    def make_config(self, after_transform=None) -> Path:
        red = np.full((2, 3), 2000, dtype=np.uint16)
        before_nir = np.asarray([[6000, 6000, 6000], [6000, 0, 6000]], dtype=np.uint16)
        after_nir = np.asarray([[8000, 4000, 6000], [6000, 6000, 6000]], dtype=np.uint16)
        scl = np.asarray([[4, 4, 4], [4, 4, 8]], dtype=np.uint8)
        files = {
            "before_blue": self.write_raster(
                "before-blue.tif", np.full((2, 3), 1800, dtype=np.uint16), band_id="B02"
            ),
            "before_green": self.write_raster(
                "before-green.tif", np.full((2, 3), 1900, dtype=np.uint16), band_id="B03"
            ),
            "before_red": self.write_raster("before-red.tif", red, band_id="B04"),
            "before_nir": self.write_raster("before-nir.tif", before_nir, band_id="B08"),
            "before_scl": self.write_raster("before-scl.tif", scl),
            "after_blue": self.write_raster(
                "after-blue.tif",
                np.full((2, 3), 1700, dtype=np.uint16),
                transform=after_transform,
                band_id="B02",
            ),
            "after_green": self.write_raster(
                "after-green.tif",
                np.full((2, 3), 1850, dtype=np.uint16),
                transform=after_transform,
                band_id="B03",
            ),
            "after_red": self.write_raster("after-red.tif", red, transform=after_transform, band_id="B04"),
            "after_nir": self.write_raster("after-nir.tif", after_nir, transform=after_transform, band_id="B08"),
            "after_scl": self.write_raster("after-scl.tif", np.full((2, 3), 4, dtype=np.uint8), transform=after_transform),
        }

        def observation(role: str, when: str) -> dict:
            return {
                "date": when,
                "red": {"path": str(files[f"{role}_red"]), "band": "B04", "unit": "surface_reflectance", "scale": 0.0001, "offset": 0, "nodata": 0, "scale_offset_source": "synthetic test fixture"},
                "nir": {"path": str(files[f"{role}_nir"]), "band": "B08", "unit": "surface_reflectance", "scale": 0.0001, "offset": 0, "nodata": 0, "scale_offset_source": "synthetic test fixture"},
                "true_color": {
                    "green": {
                        "path": str(files[f"{role}_green"]),
                        "band": "B03",
                        "unit": "surface_reflectance",
                        "scale": 0.0001,
                        "offset": 0,
                        "nodata": 0,
                        "scale_offset_source": "synthetic test fixture",
                    },
                    "blue": {
                        "path": str(files[f"{role}_blue"]),
                        "band": "B02",
                        "unit": "surface_reflectance",
                        "scale": 0.0001,
                        "offset": 0,
                        "nodata": 0,
                        "scale_offset_source": "synthetic test fixture",
                    },
                },
                "quality": {
                    "path": str(files[f"{role}_scl"]),
                    "type": "SCL",
                    "valid_classes": [4, 5, 6],
                    "resampling": "nearest",
                    "rule": "synthetic test: retain SCL 4, 5 and 6",
                },
                "source": {"fixture": "synthetic software test; not real-world evidence"},
            }

        config = {
            "schema_version": 1,
            "product": "Sentinel-2 Level-2A",
            "before": observation("before", "2023-01-01"),
            "after": observation("after", "2024-01-01"),
            "analysis": {"change_threshold": 0.09, "denominator_epsilon": 1e-6, "minimum_common_valid_pixels": 1, "interpretation_scl_classes": [4]},
            "output_root": str(self.root / "outputs"),
        }
        path = self.root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_nodata_quality_mask_common_statistics_and_outputs(self) -> None:
        record = run(self.make_config())
        self.assertEqual(record["validity"]["total_pixels"], 6)
        self.assertEqual(record["validity"]["before_quality_mask_valid_pixels"], 5)
        self.assertEqual(record["validity"]["before_numerically_valid_pixels"], 5)
        self.assertEqual(record["validity"]["before_valid_pixels"], 4)
        self.assertEqual(record["validity"]["after_valid_pixels"], 6)
        self.assertEqual(record["validity"]["common_valid_pixels"], 4)
        self.assertEqual(record["validity"]["common_interpretation_suitable_pixels"], 4)
        self.assertEqual(record["inputs"]["before"]["numerical_diagnostics"]["counts"]["near_zero_denominator"], 0)
        self.assertAlmostEqual(record["results"]["before_ndvi"]["mean"], 0.5)
        self.assertAlmostEqual(record["results"]["after_ndvi"]["mean"], (0.6 + 1 / 3 + 0.5 + 0.5) / 4)
        self.assertEqual(record["results"]["change_categories"]["increased"]["pixels"], 1)
        self.assertEqual(record["results"]["change_categories"]["decreased"]["pixels"], 1)
        self.assertEqual(record["results"]["change_categories"]["approximately_stable"]["pixels"], 2)
        self.assertAlmostEqual(record["area"]["pixel_area_square_metres"], 100.0)
        self.assertAlmostEqual(record["area"]["areas_square_metres"]["common_valid"], 400.0)

        for output in record["outputs"].values():
            self.assertTrue(Path(output).is_file(), output)
        with rasterio.open(record["outputs"]["before_ndvi"]) as dataset:
            self.assertEqual(dataset.crs.to_string(), "EPSG:32643")
            self.assertEqual(dataset.transform, self.transform)
            self.assertEqual(dataset.nodata, FLOAT_NODATA)
            self.assertEqual(dataset.dtypes, ("float32",))
            values = dataset.read(1)
            self.assertEqual(values[1, 1], FLOAT_NODATA)
            self.assertEqual(values[1, 2], FLOAT_NODATA)
        with rasterio.open(record["outputs"]["common_valid_mask"]) as dataset:
            self.assertEqual(dataset.nodata, 0)
            self.assertEqual(dataset.read(1).tolist(), [[1, 1, 1], [1, 0, 0]])
        with rasterio.open(record["outputs"]["before_ndvi_raw"]) as dataset:
            self.assertNotEqual(dataset.read(1)[1, 2], FLOAT_NODATA)
        self.assertEqual(record["inputs"]["before"]["true_color"]["band_order"], ["B04", "B03", "B02"])
        self.assertEqual(record["inputs"]["before"]["true_color"]["valid_pixels"], 6)
        self.assertTrue(record["visualization"]["true_color"]["applied_identically_to_both_dates"])
        self.assertEqual(record["visualization"]["true_color"]["reflectance_display_range"], [0.0, 0.3])
        with Image.open(record["outputs"]["comparison_png"]) as image:
            self.assertEqual(image.size, (768, 734))

    def test_mismatched_comparison_grid_is_rejected(self) -> None:
        shifted = Affine(10, 0, 500010, 0, -10, 2000000)
        with self.assertRaisesRegex(VegetationValidationError, "before and after analysis grids do not match"):
            run(self.make_config(after_transform=shifted))

    def test_quality_mask_is_nearest_resampled(self) -> None:
        config_path = self.make_config()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        coarse_transform = Affine(20, 0, 500000, 0, -20, 2000000)
        coarse = self.write_raster("coarse-scl.tif", np.asarray([[4, 8]], dtype=np.uint8), transform=coarse_transform)
        config["before"]["quality"]["path"] = str(coarse)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        record = run(config_path)
        operation = record["inputs"]["before"]["quality"]["resampling"]["operation"]
        self.assertIn("nearest-neighbour", operation)

    def test_mismatched_true_color_grid_is_rejected(self) -> None:
        config_path = self.make_config()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        shifted = Affine(10, 0, 500010, 0, -10, 2000000)
        green = self.write_raster(
            "shifted-before-green.tif",
            np.full((2, 3), 1900, dtype=np.uint16),
            transform=shifted,
            band_id="B03",
        )
        config["before"]["true_color"]["green"]["path"] = str(green)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(VegetationValidationError, "true-colour B03 grid"):
            run(config_path)

    def test_rejects_configured_double_offset_against_explicit_raster_metadata(self) -> None:
        config_path = self.make_config()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        red_path = Path(config["before"]["red"]["path"])
        with rasterio.open(red_path, "r+") as dataset:
            dataset.scales = (0.0001,)
            dataset.offsets = (0.0,)
        config["before"]["red"]["offset"] = -0.1
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(VegetationValidationError, "configured scale/offset.*disagree"):
            run(config_path)


if __name__ == "__main__":
    unittest.main()
