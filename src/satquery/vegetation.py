"""Validated Sentinel-2 Level-2A NDVI comparison workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


FLOAT_NODATA = -9999.0
SCL_CLASSES = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area_or_cast_shadow",
    3: "cloud_shadow",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified_or_low_probability_cloud",
    8: "medium_probability_cloud",
    9: "high_probability_cloud",
    10: "thin_cirrus",
    11: "snow_or_ice",
}


class VegetationValidationError(ValueError):
    """Raised when an input does not satisfy the comparison contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    crs: Any
    transform: Any
    bounds: Any

    def record(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "bounds": [self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top],
        }


@dataclass
class Observation:
    role: str
    acquisition_date: str
    red: np.ndarray
    nir: np.ndarray
    raw_ndvi: np.ndarray
    valid: np.ndarray
    quality_valid: np.ndarray
    numerical_valid: np.ndarray
    interpretation_valid: np.ndarray
    ndvi: np.ndarray
    scl: np.ndarray
    grid: Grid
    true_color: np.ndarray | None
    true_color_valid: np.ndarray | None
    resampling_record: dict[str, Any]
    warnings: list[str]


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VegetationValidationError(f"{label} must be a JSON object")
    return value


def _require_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise VegetationValidationError(f"{label} is missing required fields: {', '.join(missing)}")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VegetationValidationError(f"{label} must be a finite number")
    return float(value)


def _parse_date(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise VegetationValidationError(f"{label} must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise VegetationValidationError(f"{label} must be an ISO date (YYYY-MM-DD): {value!r}") from exc


def _resolve_path(value: Any, base_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise VegetationValidationError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VegetationValidationError(f"configuration does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise VegetationValidationError(f"configuration is unreadable or invalid JSON: {path}: {exc}") from exc
    config = dict(_require_mapping(raw, "configuration"))
    _require_keys(config, {"schema_version", "product", "before", "after", "analysis", "output_root"}, "configuration")
    if config["schema_version"] != 1:
        raise VegetationValidationError("configuration schema_version must be 1")
    if config["product"] != "Sentinel-2 Level-2A":
        raise VegetationValidationError("product must be exactly 'Sentinel-2 Level-2A'")
    return config, path.resolve().parent


def _same_grid(left: Grid, right: Grid) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and left.crs == right.crs
        and tuple(left.transform) == tuple(right.transform)
    )


def _open_single_band(
    path: Path,
    role: str,
    identity: str,
    expected_nodata: Any,
    expected_scale: float,
    expected_offset: float,
) -> tuple[np.ndarray, np.ndarray, Grid, dict[str, Any]]:
    import rasterio

    if not path.is_file():
        raise VegetationValidationError(f"{role} {identity} raster does not exist or is not a file: {path}")
    try:
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise VegetationValidationError(
                    f"{role} {identity} must be a single-channel raster; got {dataset.count} bands: {path}"
                )
            dtype = np.dtype(dataset.dtypes[0])
            if dtype.kind not in "iuf":
                raise VegetationValidationError(f"{role} {identity} must be numeric; got {dtype}: {path}")
            if dataset.crs is None:
                raise VegetationValidationError(f"{role} {identity} has no CRS: {path}")
            if dataset.transform.is_identity:
                raise VegetationValidationError(f"{role} {identity} has an identity/ungeoreferenced transform: {path}")
            configured_nodata = None if expected_nodata is None else _number(expected_nodata, f"{role}.{identity}.nodata")
            if dataset.nodata is not None and configured_nodata is not None and not math.isclose(
                float(dataset.nodata), configured_nodata, rel_tol=0.0, abs_tol=1e-12
            ):
                raise VegetationValidationError(
                    f"{role} {identity} configured nodata {configured_nodata} disagrees with raster nodata {dataset.nodata}: {path}"
                )
            nodata = configured_nodata if configured_nodata is not None else dataset.nodata
            if nodata is None:
                raise VegetationValidationError(f"{role} {identity} nodata is unknown; declare it in the configuration")
            raster_scale = float(dataset.scales[0])
            raster_offset = float(dataset.offsets[0])
            has_explicit_raster_scaling = not (
                math.isclose(raster_scale, 1.0, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(raster_offset, 0.0, rel_tol=0.0, abs_tol=1e-12)
            )
            if has_explicit_raster_scaling and (
                not math.isclose(raster_scale, expected_scale, rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(raster_offset, expected_offset, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise VegetationValidationError(
                    f"{role} {identity} configured scale/offset ({expected_scale}, {expected_offset}) disagree "
                    f"with raster metadata ({raster_scale}, {raster_offset}): {path}"
                )
            description = dataset.descriptions[0]
            tag_identity = dataset.tags(1).get("BAND_ID") or dataset.tags().get("BAND_ID")
            for declared in (description, tag_identity):
                if declared and identity not in declared.upper():
                    raise VegetationValidationError(
                        f"{role} raster metadata identifies {declared!r}, not configured {identity}: {path}"
                    )
            masked = dataset.read(1, masked=True)
            raw = np.asarray(masked.data)
            invalid = np.ma.getmaskarray(masked) | ~np.isfinite(raw)
            if np.isnan(nodata):
                invalid |= np.isnan(raw)
            else:
                invalid |= raw == nodata
            grid = Grid(dataset.width, dataset.height, dataset.crs, dataset.transform, dataset.bounds)
            metadata = {
                "path": str(path),
                "sha256": _sha256(path),
                "dtype": str(dtype),
                "raster_nodata": dataset.nodata,
                "configured_nodata": expected_nodata,
                "effective_nodata": nodata,
                "raster_scale": raster_scale,
                "raster_offset": raster_offset,
                "raster_unit": dataset.units[0],
                "raster_scaling_metadata_explicit": has_explicit_raster_scaling,
                "description": description,
                "band_id_tag": tag_identity,
            }
            return raw, invalid, grid, metadata
    except VegetationValidationError:
        raise
    except (OSError, rasterio.errors.RasterioError) as exc:
        raise VegetationValidationError(f"{role} {identity} raster is unreadable: {path}: {exc}") from exc


def _read_quality(
    path: Path,
    role: str,
    target: Grid,
    valid_classes: Sequence[int],
    resampling: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import rasterio
    from rasterio.warp import Resampling, reproject

    if not path.is_file():
        raise VegetationValidationError(f"{role} SCL raster does not exist or is not a file: {path}")
    try:
        with rasterio.open(path) as dataset:
            if dataset.count != 1 or np.dtype(dataset.dtypes[0]).kind not in "iu":
                raise VegetationValidationError(f"{role} SCL must be a single-channel integer categorical raster: {path}")
            if dataset.crs is None or dataset.transform.is_identity:
                raise VegetationValidationError(f"{role} SCL must have a CRS and georeferenced transform: {path}")
            source_grid = Grid(dataset.width, dataset.height, dataset.crs, dataset.transform, dataset.bounds)
            source = dataset.read(1)
            observed = set(np.unique(source).astype(int).tolist())
            unsupported = sorted(observed - set(SCL_CLASSES))
            if unsupported:
                raise VegetationValidationError(f"{role} SCL contains labels outside 0..11: {unsupported}: {path}")
            if _same_grid(source_grid, target):
                labels = source
                operation = "none; already on analysis grid"
            else:
                if resampling != "nearest":
                    raise VegetationValidationError(
                        f"{role} SCL grid differs from the analysis grid; quality.resampling must be 'nearest'"
                    )
                if dataset.crs != target.crs:
                    raise VegetationValidationError(
                        f"{role} SCL CRS differs from the analysis CRS; first version only resamples within one CRS"
                    )
                tolerance = max(abs(target.transform.a), abs(target.transform.e)) * 0.01
                if (
                    dataset.bounds.left > target.bounds.left + tolerance
                    or dataset.bounds.bottom > target.bounds.bottom + tolerance
                    or dataset.bounds.right < target.bounds.right - tolerance
                    or dataset.bounds.top < target.bounds.top - tolerance
                ):
                    raise VegetationValidationError(f"{role} SCL does not fully cover the analysis grid: {path}")
                labels = np.zeros((target.height, target.width), dtype=source.dtype)
                reproject(
                    source=source,
                    destination=labels,
                    src_transform=dataset.transform,
                    src_crs=dataset.crs,
                    dst_transform=target.transform,
                    dst_crs=target.crs,
                    resampling=Resampling.nearest,
                    src_nodata=dataset.nodata,
                    dst_nodata=0,
                )
                operation = "rasterio.warp.reproject with categorical nearest-neighbour resampling"
            return labels, np.isin(labels, np.asarray(valid_classes)), {
                "path": str(path),
                "sha256": _sha256(path),
                "dtype": dataset.dtypes[0],
                "raster_nodata": dataset.nodata,
                "nodata_interpretation": "SCL class 0 (NO_DATA) is always excluded",
                "source_grid": source_grid.record(),
                "target_grid": target.record(),
                "operation": operation,
                "observed_classes": sorted(observed),
            }
    except VegetationValidationError:
        raise
    except (OSError, rasterio.errors.RasterioError) as exc:
        raise VegetationValidationError(f"{role} SCL raster is unreadable: {path}: {exc}") from exc


def calculate_ndvi(
    red_stored: np.ndarray,
    nir_stored: np.ndarray,
    red_scale: float,
    red_offset: float,
    nir_scale: float,
    nir_offset: float,
    valid: np.ndarray,
    denominator_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale stored values as float64 and calculate NDVI without clipping."""
    red = red_stored.astype(np.float64) * red_scale + red_offset
    nir = nir_stored.astype(np.float64) * nir_scale + nir_offset
    denominator = nir + red
    final_valid = valid & np.isfinite(red) & np.isfinite(nir) & (np.abs(denominator) > denominator_epsilon)
    ndvi = np.full(red.shape, np.nan, dtype=np.float64)
    np.divide(nir - red, denominator, out=ndvi, where=final_valid)
    final_valid &= np.isfinite(ndvi)
    ndvi[~final_valid] = np.nan
    return red, nir, ndvi, final_valid


def _distribution(values: np.ndarray) -> dict[str, Any]:
    percentiles = (0, 1, 5, 25, 50, 75, 95, 99, 100)
    if values.size == 0:
        return {"count": 0, "percentiles": {str(value): None for value in percentiles}}
    calculated = np.percentile(values.astype(np.float64), percentiles)
    return {
        "count": int(values.size),
        "minimum": float(calculated[0]),
        "maximum": float(calculated[-1]),
        "percentiles": {str(key): float(value) for key, value in zip(percentiles, calculated)},
    }


def _observation(
    config: Mapping[str, Any],
    role: str,
    base_dir: Path,
    epsilon: float,
    interpretation_classes: Sequence[int],
) -> tuple[Observation, dict[str, Any]]:
    item = _require_mapping(config.get(role), role)
    _require_keys(item, {"date", "red", "nir", "quality", "source"}, role)
    acquisition_date = _parse_date(item["date"], f"{role}.date")
    source = _require_mapping(item["source"], f"{role}.source")
    if not source:
        raise VegetationValidationError(f"{role}.source must record provenance")

    band_results: dict[str, tuple[np.ndarray, np.ndarray, Grid, dict[str, Any]]] = {}
    band_records: dict[str, Any] = {}
    scales: dict[str, tuple[float, float]] = {}
    for key, expected_identity in (("red", "B04"), ("nir", "B08")):
        spec = _require_mapping(item[key], f"{role}.{key}")
        _require_keys(
            spec,
            {"path", "band", "unit", "scale", "offset", "nodata", "scale_offset_source"},
            f"{role}.{key}",
        )
        if spec["band"] != expected_identity:
            raise VegetationValidationError(f"{role}.{key}.band must be exactly {expected_identity}")
        if spec["unit"] != "surface_reflectance":
            raise VegetationValidationError(f"{role}.{key}.unit must be exactly 'surface_reflectance'")
        scale = _number(spec["scale"], f"{role}.{key}.scale")
        offset = _number(spec["offset"], f"{role}.{key}.offset")
        if scale == 0:
            raise VegetationValidationError(f"{role}.{key}.scale must be non-zero")
        if not isinstance(spec["scale_offset_source"], str) or not spec["scale_offset_source"].strip():
            raise VegetationValidationError(f"{role}.{key}.scale_offset_source must document the metadata source")
        path = _resolve_path(spec["path"], base_dir, f"{role}.{key}.path")
        result = _open_single_band(path, role, expected_identity, spec["nodata"], scale, offset)
        band_results[key] = result
        band_records[key] = {
            **result[3],
            "identity": expected_identity,
            "unit_after_scaling": spec["unit"],
            "scale": scale,
            "offset": offset,
            "scale_offset_source": spec["scale_offset_source"],
        }
        scales[key] = (scale, offset)

    red_raw, red_invalid, red_grid, _ = band_results["red"]
    nir_raw, nir_invalid, nir_grid, _ = band_results["nir"]
    if not _same_grid(red_grid, nir_grid):
        raise VegetationValidationError(f"{role} B04 and B08 grids do not match; align them before analysis")

    true_color: np.ndarray | None = None
    true_color_valid: np.ndarray | None = None
    true_color_record: dict[str, Any] | None = None
    true_color_config = item.get("true_color")
    if true_color_config is not None:
        display = _require_mapping(true_color_config, f"{role}.true_color")
        _require_keys(display, {"green", "blue"}, f"{role}.true_color")
        display_red = red_raw.astype(np.float64) * scales["red"][0] + scales["red"][1]
        display_bands: dict[str, np.ndarray] = {"red": display_red}
        display_invalid: dict[str, np.ndarray] = {"red": red_invalid | ~np.isfinite(display_red)}
        display_records: dict[str, Any] = {
            "red": {
                "reference": f"{role}.red",
                "identity": "B04",
                "note": "The already validated NDVI red reflectance is reused without changing the analysis.",
            }
        }
        for key, identity in (("green", "B03"), ("blue", "B02")):
            spec = _require_mapping(display[key], f"{role}.true_color.{key}")
            _require_keys(
                spec,
                {"path", "band", "unit", "scale", "offset", "nodata", "scale_offset_source"},
                f"{role}.true_color.{key}",
            )
            if spec["band"] != identity:
                raise VegetationValidationError(f"{role}.true_color.{key}.band must be exactly {identity}")
            if spec["unit"] != "surface_reflectance":
                raise VegetationValidationError(
                    f"{role}.true_color.{key}.unit must be exactly 'surface_reflectance'"
                )
            scale = _number(spec["scale"], f"{role}.true_color.{key}.scale")
            offset = _number(spec["offset"], f"{role}.true_color.{key}.offset")
            if scale == 0:
                raise VegetationValidationError(f"{role}.true_color.{key}.scale must be non-zero")
            if not isinstance(spec["scale_offset_source"], str) or not spec["scale_offset_source"].strip():
                raise VegetationValidationError(
                    f"{role}.true_color.{key}.scale_offset_source must document the metadata source"
                )
            path = _resolve_path(spec["path"], base_dir, f"{role}.true_color.{key}.path")
            raw, invalid, grid, metadata = _open_single_band(
                path, role, identity, spec["nodata"], scale, offset
            )
            if not _same_grid(red_grid, grid):
                raise VegetationValidationError(
                    f"{role} true-colour {identity} grid does not match the B04/B08 analysis grid"
                )
            reflectance = raw.astype(np.float64) * scale + offset
            display_bands[key] = reflectance
            display_invalid[key] = invalid | ~np.isfinite(reflectance)
            display_records[key] = {
                **metadata,
                "identity": identity,
                "unit_after_scaling": spec["unit"],
                "scale": scale,
                "offset": offset,
                "scale_offset_source": spec["scale_offset_source"],
            }
        true_color = np.stack(
            (display_bands["red"], display_bands["green"], display_bands["blue"]), axis=-1
        )
        true_color_valid = ~(
            display_invalid["red"] | display_invalid["green"] | display_invalid["blue"]
        )
        true_color_record = {
            "purpose": "Display only; these bands and display adjustments do not enter NDVI, masks, or statistics.",
            "band_order": ["B04", "B03", "B02"],
            "bands": display_records,
            "valid_pixels": int(true_color_valid.sum()),
            "grid": red_grid.record(),
        }

    quality = _require_mapping(item["quality"], f"{role}.quality")
    _require_keys(quality, {"path", "type", "valid_classes", "resampling", "rule"}, f"{role}.quality")
    if quality["type"] != "SCL":
        raise VegetationValidationError(f"{role}.quality.type must be 'SCL'")
    classes = quality["valid_classes"]
    if (
        not isinstance(classes, list)
        or not classes
        or any(isinstance(value, bool) or not isinstance(value, int) for value in classes)
        or not set(classes).issubset(SCL_CLASSES)
    ):
        raise VegetationValidationError(f"{role}.quality.valid_classes must be a non-empty list drawn from 0..11")
    if not set(classes).issubset({4, 5, 6}):
        raise VegetationValidationError(
            f"{role}.quality.valid_classes includes an unsuitable Sentinel-2 SCL class; use only a subset of 4,5,6"
        )
    if not isinstance(quality["rule"], str) or not quality["rule"].strip():
        raise VegetationValidationError(f"{role}.quality.rule must document the mask interpretation")
    quality_path = _resolve_path(quality["path"], base_dir, f"{role}.quality.path")
    scl, quality_valid, resampling_record = _read_quality(
        quality_path, role, red_grid, classes, str(quality["resampling"])
    )
    band_data_valid = ~red_invalid & ~nir_invalid
    red, nir, raw_ndvi, numerical_valid = calculate_ndvi(
        red_raw,
        nir_raw,
        scales["red"][0],
        scales["red"][1],
        scales["nir"][0],
        scales["nir"][1],
        band_data_valid,
        epsilon,
    )
    denominator = red + nir
    near_zero = band_data_valid & np.isfinite(denominator) & (np.abs(denominator) <= epsilon)
    valid = quality_valid & numerical_valid
    interpretation_valid = valid & np.isin(scl, np.asarray(interpretation_classes))
    ndvi = np.where(valid, raw_ndvi, np.nan)
    outside = numerical_valid & ((raw_ndvi < -1.0) | (raw_ndvi > 1.0))
    warnings: list[str] = []
    if outside.any():
        warnings.append(f"{role} contains NDVI outside [-1, 1]; values were retained, not clipped")
    by_scl: dict[str, Any] = {}
    for label in sorted(np.unique(scl).astype(int).tolist()):
        selected = scl == label
        by_scl[str(label)] = {
            "meaning": SCL_CLASSES[label],
            "pixels": int(selected.sum()),
            "quality_mask_valid": int(np.count_nonzero(selected & quality_valid)),
            "band_data_valid": int(np.count_nonzero(selected & band_data_valid)),
            "numerically_valid_ndvi": int(np.count_nonzero(selected & numerical_valid)),
            "analysis_valid": int(np.count_nonzero(selected & valid)),
            "interpretation_suitable": int(np.count_nonzero(selected & interpretation_valid)),
            "red_negative_reflectance": int(np.count_nonzero(selected & band_data_valid & (red < 0))),
            "red_zero_reflectance": int(np.count_nonzero(selected & band_data_valid & (red == 0))),
            "nir_negative_reflectance": int(np.count_nonzero(selected & band_data_valid & (nir < 0))),
            "nir_zero_reflectance": int(np.count_nonzero(selected & band_data_valid & (nir == 0))),
            "near_zero_denominator": int(np.count_nonzero(selected & near_zero)),
            "ndvi_outside_minus1_plus1": int(np.count_nonzero(selected & outside)),
        }
    diagnostics = {
        "stored_values": {
            "red": _distribution(red_raw[~red_invalid]),
            "nir": _distribution(nir_raw[~nir_invalid]),
        },
        "reflectance": {
            "red": {
                **_distribution(red[~red_invalid & np.isfinite(red)]),
                "negative_count": int(np.count_nonzero(~red_invalid & np.isfinite(red) & (red < 0))),
                "zero_count": int(np.count_nonzero(~red_invalid & np.isfinite(red) & (red == 0))),
            },
            "nir": {
                **_distribution(nir[~nir_invalid & np.isfinite(nir)]),
                "negative_count": int(np.count_nonzero(~nir_invalid & np.isfinite(nir) & (nir < 0))),
                "zero_count": int(np.count_nonzero(~nir_invalid & np.isfinite(nir) & (nir == 0))),
            },
        },
        "counts": {
            "total_pixels": int(scl.size),
            "quality_mask_valid": int(quality_valid.sum()),
            "band_data_valid": int(band_data_valid.sum()),
            "near_zero_denominator": int(near_zero.sum()),
            "numerically_valid_ndvi": int(numerical_valid.sum()),
            "ndvi_outside_minus1_plus1": int(outside.sum()),
            "analysis_valid": int(valid.sum()),
            "interpretation_suitable": int(interpretation_valid.sum()),
        },
        "by_scl_class": by_scl,
    }
    record = {
        "date": acquisition_date,
        "source": dict(source),
        "bands": band_records,
        "quality": {
            "type": "Sentinel-2 Level-2A Scene Classification Layer (SCL)",
            "valid_classes": classes,
            "valid_class_meanings": {str(value): SCL_CLASSES[value] for value in classes},
            "excluded_classes": [value for value in SCL_CLASSES if value not in classes],
            "rule": quality["rule"],
            "resampling": resampling_record,
        },
        "true_color": true_color_record,
        "numerical_diagnostics": diagnostics,
        "valid_pixels_for_date": int(valid.sum()),
    }
    return Observation(
        role,
        acquisition_date,
        red,
        nir,
        raw_ndvi,
        valid,
        quality_valid,
        numerical_valid,
        interpretation_valid,
        ndvi,
        scl,
        red_grid,
        true_color,
        true_color_valid,
        resampling_record,
        warnings,
    ), record


def classify_delta(delta: np.ndarray, threshold: float) -> dict[str, int]:
    finite = np.isfinite(delta)
    return {
        "increased": int(np.count_nonzero(finite & (delta > threshold))),
        "decreased": int(np.count_nonzero(finite & (delta < -threshold))),
        "approximately_stable": int(np.count_nonzero(finite & (delta >= -threshold) & (delta <= threshold))),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values))}


def _area_record(grid: Grid, counts: Mapping[str, int]) -> tuple[dict[str, Any] | None, str | None]:
    if not grid.crs.is_projected:
        return None, "Ground area omitted because the analysis CRS is geographic, not projected."
    try:
        unit_name, metres_per_unit = grid.crs.linear_units_factor
        metres_per_unit = float(metres_per_unit)
    except (AttributeError, TypeError, ValueError):
        return None, "Ground area omitted because projected CRS linear units could not be converted to metres."
    determinant = abs(grid.transform.a * grid.transform.e - grid.transform.b * grid.transform.d)
    if not math.isfinite(determinant) or determinant <= 0 or not math.isfinite(metres_per_unit):
        return None, "Ground area omitted because pixel geometry or CRS linear units are invalid."
    pixel_square_metres = determinant * metres_per_unit**2
    return {
        "basis": "absolute affine determinant converted from projected CRS linear units",
        "linear_unit": unit_name,
        "metres_per_linear_unit": metres_per_unit,
        "pixel_area_square_metres": pixel_square_metres,
        "areas_square_metres": {name: value * pixel_square_metres for name, value in counts.items()},
    }, None


def _write_geotiff(path: Path, values: np.ndarray, grid: Grid, nodata: float, dtype: str) -> None:
    import rasterio

    output = np.where(np.isfinite(values), values, nodata).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=1,
        dtype=dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
        predictor=3 if np.dtype(dtype).kind == "f" else 1,
    ) as dataset:
        dataset.write(output, 1)


def _interpolate_colors(values: np.ndarray, stops: Sequence[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    flat = values.ravel()
    result = np.empty((flat.size, 3), dtype=np.uint8)
    positions = np.asarray([item[0] for item in stops], dtype=float)
    colors = np.asarray([item[1] for item in stops], dtype=float)
    for channel in range(3):
        result[:, channel] = np.interp(flat, positions, colors[:, channel]).astype(np.uint8)
    return result.reshape(values.shape + (3,))


def _colorize(values: np.ndarray, low: float, high: float, palette: Sequence[tuple[float, tuple[int, int, int]]]) -> Image.Image:
    valid = np.isfinite(values)
    normalized = np.zeros(values.shape, dtype=float)
    normalized[valid] = np.clip((values[valid] - low) / (high - low), 0.0, 1.0)
    rgb = _interpolate_colors(normalized, palette)
    rgb[~valid] = (190, 190, 190)
    return Image.fromarray(rgb, mode="RGB")


def _true_color_image(
    reflectance: np.ndarray,
    valid: np.ndarray,
    low: float,
    high: float,
    gamma: float,
) -> tuple[Image.Image, dict[str, int]]:
    normalized = np.clip((reflectance - low) / (high - low), 0.0, 1.0)
    displayed = np.power(normalized, 1.0 / gamma)
    rgb = np.rint(displayed * 255.0).astype(np.uint8)
    rgb[~valid] = (190, 190, 190)
    return Image.fromarray(rgb, mode="RGB"), {
        "below_range_values": int(np.count_nonzero(valid[..., None] & (reflectance < low))),
        "above_range_values": int(np.count_nonzero(valid[..., None] & (reflectance > high))),
        "excluded_pixels": int(np.count_nonzero(~valid)),
    }


def _draw_colorbar(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    low: float,
    high: float,
    palette: Sequence[tuple[float, tuple[int, int, int]]],
) -> None:
    gradient = np.linspace(0.0, 1.0, width, dtype=float)[None, :]
    bar = np.repeat(_interpolate_colors(gradient, palette), 14, axis=0)
    canvas.paste(Image.fromarray(bar, mode="RGB"), (x, y))
    draw.rectangle((x, y, x + width - 1, y + 13), outline="black")
    draw.text((x, y + 16), f"{low:.2f}", fill="black")
    draw.text((x + width - 35, y + 16), f"{high:.2f}", fill="black")


def _comparison_png(
    path: Path,
    before: Observation,
    after: Observation,
    delta: np.ndarray,
    interpretation: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    ndvi_palette = ((0.0, (120, 68, 35)), (0.5, (245, 231, 145)), (1.0, (0, 95, 45)))
    delta_palette = ((0.0, (178, 24, 43)), (0.5, (255, 255, 255)), (1.0, (33, 102, 172)))
    delta_limit = 1.0
    panels = (
        (
            f"Before NDVI: {before.acquisition_date}",
            _colorize(before.ndvi, -1.0, 1.0, ndvi_palette),
            -1.0,
            1.0,
            ndvi_palette,
        ),
        (
            f"After NDVI: {after.acquisition_date}",
            _colorize(after.ndvi, -1.0, 1.0, ndvi_palette),
            -1.0,
            1.0,
            ndvi_palette,
        ),
        (
            "NDVI difference (after - before)",
            _colorize(delta, -delta_limit, delta_limit, delta_palette),
            -delta_limit,
            delta_limit,
            delta_palette,
        ),
    )
    panel_width = max(256, before.ndvi.shape[1])
    panel_height = max(256, before.ndvi.shape[0])
    has_true_color = before.true_color is not None and after.true_color is not None
    if has_true_color != (before.true_color_valid is not None and after.true_color_valid is not None):
        raise VegetationValidationError("true-colour arrays and validity masks are incomplete")
    top_height = panel_height + 38 if has_true_color else 0
    footer_y = top_height + panel_height + 72
    canvas = Image.new("RGB", (panel_width * 3, footer_y + 112), "white")
    draw = ImageDraw.Draw(canvas)
    true_color_record: dict[str, Any] | None = None
    if has_true_color:
        reflectance_low = 0.0
        reflectance_high = 0.3
        gamma = 2.2
        before_rgb, before_counts = _true_color_image(
            before.true_color, before.true_color_valid, reflectance_low, reflectance_high, gamma
        )
        after_rgb, after_counts = _true_color_image(
            after.true_color, after.true_color_valid, reflectance_low, reflectance_high, gamma
        )
        for x, title, image in (
            (panel_width // 2, f"True colour before: {before.acquisition_date}", before_rgb),
            (panel_width * 3 // 2, f"True colour after: {after.acquisition_date}", after_rgb),
        ):
            draw.text((x + 6, 4), title, fill="black")
            canvas.paste(image.resize((panel_width, panel_height), Image.Resampling.NEAREST), (x, 20))
        draw.text(
            (panel_width // 2 + 6, panel_height + 22),
            "Same display settings: reflectance 0.00-0.30, gamma 2.2 (display only)",
            fill="black",
        )
        true_color_record = {
            "band_order": ["B04", "B03", "B02"],
            "reflectance_display_range": [reflectance_low, reflectance_high],
            "gamma": gamma,
            "applied_identically_to_both_dates": True,
            "resampling": "nearest-neighbour for PNG enlargement only; source rasters remain on the analysis grid",
            "before_display_counts": before_counts,
            "after_display_counts": after_counts,
            "note": (
                "Reflectance scaling is validated on input. Range clipping and gamma are presentation-only "
                "and do not affect NDVI, masks, GeoTIFFs, or statistics."
            ),
        }
    for index, (title, image, low, high, palette) in enumerate(panels):
        resized = image.resize((panel_width, panel_height), Image.Resampling.NEAREST)
        x = index * panel_width
        y = top_height + 20
        canvas.paste(resized, (x, y))
        draw.text((x + 6, top_height + 4), title, fill="black")
        _draw_colorbar(
            canvas, draw, x + 8, top_height + panel_height + 26, panel_width - 16, low, high, palette
        )
    categories = interpretation["change_categories"]
    scl_classes = interpretation["scl_classes"]
    population_label = "Vegetation-only" if scl_classes == [4] else "Interpretation-selected"
    scl_label = ", ".join(str(value) for value in scl_classes)
    draw.text((8, footer_y), "NDVI: Normalised Difference Vegetation Index", fill="black")
    draw.text(
        (8, footer_y + 14),
        "Difference: blue = NDVI increased; red = NDVI decreased; white = little difference; grey = excluded pixels.",
        fill="black",
    )
    draw.text(
        (8, footer_y + 32),
        f"{population_label} statistics - SCL {scl_label} on BOTH dates ({interpretation['pixels']:,} pixels):",
        fill="black",
    )
    draw.text(
        (8, footer_y + 46),
        f"Mean NDVI {interpretation['before_ndvi']['mean']:.3f} -> {interpretation['after_ndvi']['mean']:.3f}; "
        f"median {interpretation['before_ndvi']['median']:.3f} -> {interpretation['after_ndvi']['median']:.3f}; "
        f"mean difference {interpretation['delta_ndvi']['mean']:+.3f}.",
        fill="black",
    )
    draw.text(
        (8, footer_y + 60),
        f"At +/-{threshold:.2f}: {categories['increased']['fraction']:.1%} increased, "
        f"{categories['decreased']['fraction']:.1%} decreased, "
        f"{categories['approximately_stable']['fraction']:.1%} little difference.",
        fill="black",
    )
    draw.text(
        (8, footer_y + 78),
        "NDVI differences are not vegetation gain/loss or proof of health change.",
        fill="black",
    )
    canvas.save(path)
    before_saturated = int(
        np.count_nonzero(np.isfinite(before.ndvi) & ((before.ndvi < -1.0) | (before.ndvi > 1.0)))
    )
    after_saturated = int(
        np.count_nonzero(np.isfinite(after.ndvi) & ((after.ndvi < -1.0) | (after.ndvi > 1.0)))
    )
    delta_saturated = int(
        np.count_nonzero(np.isfinite(delta) & ((delta < -delta_limit) | (delta > delta_limit)))
    )
    return {
        "ndvi_display_range": [-1.0, 1.0],
        "delta_display_range": [-delta_limit, delta_limit],
        "invalid_color_rgb": [190, 190, 190],
        "true_color": true_color_record,
        "display_saturated_pixel_counts": {
            "before_ndvi": before_saturated,
            "after_ndvi": after_saturated,
            "delta_ndvi": delta_saturated,
        },
        "note": (
            "Only display colors saturate at range endpoints; numerical raster values and statistics are "
            "never clipped or changed."
        ),
    }


def _new_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}-{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _summary_text(record: Mapping[str, Any]) -> str:
    results = record["results"]
    interpretation = record["interpretation_results"]
    validity = record["validity"]
    categories = interpretation["change_categories"]
    interpretation_label = (
        "Vegetation-only" if record["interpretation"]["scl_classes"] == [4] else "Interpretation-selected"
    )
    area_line = (
        f"Common-valid area: {record['area']['areas_square_metres']['common_valid']:.2f} square metres; "
        f"interpretation-suitable area: "
        f"{record['area']['areas_square_metres']['common_interpretation_suitable']:.2f} square metres."
        if record.get("area")
        else record["area_omission_reason"]
    )
    return (
        "SatQuery Sentinel-2 Level-2A NDVI comparison\n"
        f"Dates: {record['dates']['before']} to {record['dates']['after']}\n"
        f"Common valid pixels: {validity['common_valid_pixels']} of {validity['total_pixels']} "
        f"({validity['valid_coverage_percent']:.2f}%)\n"
        f"{interpretation_label} pixels (SCL {record['interpretation']['scl_classes']} on BOTH dates): "
        f"{validity['common_interpretation_suitable_pixels']}\n"
        f"{interpretation_label} before NDVI: mean {interpretation['before_ndvi']['mean']:.6f}, "
        f"median {interpretation['before_ndvi']['median']:.6f}\n"
        f"{interpretation_label} after NDVI: mean {interpretation['after_ndvi']['mean']:.6f}, "
        f"median {interpretation['after_ndvi']['median']:.6f}\n"
        f"{interpretation_label} delta NDVI (after - before): mean {interpretation['delta_ndvi']['mean']:.6f}, "
        f"median {interpretation['delta_ndvi']['median']:.6f}\n"
        f"Threshold: +/-{record['threshold']:.6f} (exploratory and configurable)\n"
        f"NDVI increase: {categories['increased']['pixels']} pixels "
        f"({categories['increased']['fraction']:.2%}); NDVI decrease: {categories['decreased']['pixels']} pixels "
        f"({categories['decreased']['fraction']:.2%}); approximately stable: "
        f"{categories['approximately_stable']['pixels']} pixels "
        f"({categories['approximately_stable']['fraction']:.2%})\n"
        f"All quality-valid classes {record['quality_mask']['valid_scl_classes']} over their common area: "
        f"before mean/median {results['before_ndvi']['mean']:.6f}/{results['before_ndvi']['median']:.6f}, "
        f"after {results['after_ndvi']['mean']:.6f}/{results['after_ndvi']['median']:.6f}, "
        f"delta {results['delta_ndvi']['mean']:.6f}/{results['delta_ndvi']['median']:.6f}.\n"
        f"{area_line}\n"
        "Interpretation: NDVI differences alone do not establish deforestation, crop stress, vegetation gain/loss, "
        "health improvement, damage, or the cause of change. Seasonal comparability, atmosphere, illumination, "
        "view geometry, residual cloud/shadow contamination, land cover, and observation conditions must be considered.\n"
        f"Warnings: {' | '.join(record['warnings'])}\n"
    )


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    config, base_dir = load_config(config_path.resolve())
    analysis = _require_mapping(config["analysis"], "analysis")
    _require_keys(
        analysis,
        {"change_threshold", "denominator_epsilon", "minimum_common_valid_pixels", "interpretation_scl_classes"},
        "analysis",
    )
    threshold = _number(analysis["change_threshold"], "analysis.change_threshold")
    epsilon = _number(analysis["denominator_epsilon"], "analysis.denominator_epsilon")
    minimum = analysis["minimum_common_valid_pixels"]
    interpretation_classes = analysis["interpretation_scl_classes"]
    if threshold < 0 or epsilon < 0:
        raise VegetationValidationError("change_threshold and denominator_epsilon must be non-negative")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise VegetationValidationError("analysis.minimum_common_valid_pixels must be a positive integer")
    if (
        not isinstance(interpretation_classes, list)
        or not interpretation_classes
        or any(isinstance(value, bool) or not isinstance(value, int) for value in interpretation_classes)
        or not set(interpretation_classes).issubset({4, 5, 6})
    ):
        raise VegetationValidationError(
            "analysis.interpretation_scl_classes must be a non-empty subset of SCL classes 4, 5, 6"
        )

    before, before_record = _observation(config, "before", base_dir, epsilon, interpretation_classes)
    after, after_record = _observation(config, "after", base_dir, epsilon, interpretation_classes)
    if before_record["quality"]["valid_classes"] != after_record["quality"]["valid_classes"]:
        raise VegetationValidationError("before and after quality.valid_classes must match")
    if not set(interpretation_classes).issubset(before_record["quality"]["valid_classes"]):
        raise VegetationValidationError(
            "analysis.interpretation_scl_classes must be a subset of both dates' quality.valid_classes"
        )
    if not _same_grid(before.grid, after.grid):
        raise VegetationValidationError("before and after analysis grids do not match; align them before analysis")
    if (before.true_color is None) != (after.true_color is None):
        raise VegetationValidationError("true-colour inputs must be supplied for both dates or neither date")
    if before.acquisition_date >= after.acquisition_date:
        raise VegetationValidationError("before.date must be earlier than after.date")

    common = before.valid & after.valid
    common_count = int(common.sum())
    if common_count < minimum:
        raise VegetationValidationError(
            f"only {common_count} common valid pixels remain; minimum required is {minimum}"
        )
    delta = np.full(before.ndvi.shape, np.nan, dtype=np.float64)
    delta[common] = after.ndvi[common] - before.ndvi[common]
    category_counts = classify_delta(delta, threshold)
    interpretation_common = before.interpretation_valid & after.interpretation_valid
    interpretation_count = int(interpretation_common.sum())
    if interpretation_count == 0:
        raise VegetationValidationError("no common interpretation-suitable pixels remain")
    interpretation_delta = np.full(before.ndvi.shape, np.nan, dtype=np.float64)
    interpretation_delta[interpretation_common] = (
        after.ndvi[interpretation_common] - before.ndvi[interpretation_common]
    )
    interpretation_category_counts = classify_delta(interpretation_delta, threshold)
    fractions = {
        name: {"pixels": count, "fraction": count / common_count}
        for name, count in category_counts.items()
    }
    interpretation_fractions = {
        name: {"pixels": count, "fraction": count / interpretation_count}
        for name, count in interpretation_category_counts.items()
    }
    interpretation_results = {
        "population": "common interpretation-suitable pixels",
        "pixels": interpretation_count,
        "scl_classes": list(interpretation_classes),
        "before_ndvi": _stats(before.ndvi[interpretation_common]),
        "after_ndvi": _stats(after.ndvi[interpretation_common]),
        "delta_ndvi": _stats(interpretation_delta[interpretation_common]),
        "change_categories": interpretation_fractions,
    }
    total = int(common.size)
    area_counts = {
        "common_valid": common_count,
        "common_interpretation_suitable": interpretation_count,
        **{f"all_valid_{name}": count for name, count in category_counts.items()},
        **{f"interpretation_{name}": count for name, count in interpretation_category_counts.items()},
    }
    area, area_warning = _area_record(before.grid, area_counts)
    warnings = [
        "The SCL is a scene-classification mask, not a perfect cloud/shadow detector; residual contamination and classification errors can remain.",
        "The configured change threshold is exploratory, not a universal vegetation-change threshold.",
        f"Interpretation statistics use only SCL classes {interpretation_classes} on both dates; broader quality-valid results are reported separately.",
        "NDVI differences alone do not establish deforestation, crop stress, vegetation gain/loss, health improvement, damage, or causation.",
        *before.warnings,
        *after.warnings,
    ]
    if area_warning:
        warnings.append(area_warning)

    root = output_override.resolve() if output_override else _resolve_path(config["output_root"], base_dir, "output_root")
    run_dir = _new_run_directory(root)
    paths = {
        "before_ndvi": run_dir / "before_ndvi.tif",
        "after_ndvi": run_dir / "after_ndvi.tif",
        "delta_ndvi": run_dir / "delta_ndvi.tif",
        "common_valid_mask": run_dir / "common_valid_mask.tif",
        "common_interpretation_mask": run_dir / "common_interpretation_mask.tif",
        "before_ndvi_raw": run_dir / "before_ndvi_raw.tif",
        "after_ndvi_raw": run_dir / "after_ndvi_raw.tif",
        "comparison_png": run_dir / "comparison.png",
        "execution_record": run_dir / "run.json",
        "summary": run_dir / "summary.txt",
    }
    _write_geotiff(paths["before_ndvi"], before.ndvi, before.grid, FLOAT_NODATA, "float32")
    _write_geotiff(paths["after_ndvi"], after.ndvi, before.grid, FLOAT_NODATA, "float32")
    _write_geotiff(paths["delta_ndvi"], delta, before.grid, FLOAT_NODATA, "float32")
    _write_geotiff(paths["common_valid_mask"], common.astype(np.uint8), before.grid, 0, "uint8")
    _write_geotiff(
        paths["common_interpretation_mask"], interpretation_common.astype(np.uint8), before.grid, 0, "uint8"
    )
    _write_geotiff(paths["before_ndvi_raw"], before.raw_ndvi, before.grid, FLOAT_NODATA, "float32")
    _write_geotiff(paths["after_ndvi_raw"], after.raw_ndvi, before.grid, FLOAT_NODATA, "float32")
    visualization = _comparison_png(
        paths["comparison_png"], before, after, delta, interpretation_results, threshold
    )
    saturated = visualization["display_saturated_pixel_counts"]
    if any(saturated.values()):
        warnings.append(
            "Comparison PNG colors saturate outside the recorded display ranges for "
            f"before={saturated['before_ndvi']}, after={saturated['after_ndvi']}, and delta={saturated['delta_ndvi']} pixels; "
            "GeoTIFF values and statistics were retained without clipping."
        )

    elapsed = time.perf_counter() - started
    record: dict[str, Any] = {
        "schema_version": 2,
        "analysis": "Sentinel-2 Level-2A NDVI comparison",
        "status": "success",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": str(config_path.resolve()),
        "product": config["product"],
        "dates": {"before": before.acquisition_date, "after": after.acquisition_date},
        "inputs": {"before": before_record, "after": after_record},
        "reflectance_formula": "reflectance = stored_value * scale + offset",
        "ndvi_formula": "(NIR - Red) / (NIR + Red)",
        "delta_formula": "after_NDVI - before_NDVI",
        "grid": before.grid.record(),
        "threshold": threshold,
        "threshold_interpretation": {
            "increased": "delta_NDVI > threshold",
            "decreased": "delta_NDVI < -threshold",
            "approximately_stable": "-threshold <= delta_NDVI <= threshold",
            "note": "Exploratory and configurable; there is no universal correct threshold.",
        },
        "denominator_epsilon": epsilon,
        "quality_mask": {
            "valid_scl_classes": config["before"]["quality"]["valid_classes"],
            "purpose": "Exclude unsuitable observation conditions; not a vegetation-only land-cover selection.",
        },
        "interpretation": {
            "scl_classes": interpretation_classes,
            "class_meanings": {str(value): SCL_CLASSES[value] for value in interpretation_classes},
            "pixel_rule": "A pixel must be numerically valid, quality-mask-valid, and in an interpretation class on both dates.",
        },
        "validity": {
            "total_pixels": total,
            "before_quality_mask_valid_pixels": int(before.quality_valid.sum()),
            "after_quality_mask_valid_pixels": int(after.quality_valid.sum()),
            "before_numerically_valid_pixels": int(before.numerical_valid.sum()),
            "after_numerically_valid_pixels": int(after.numerical_valid.sum()),
            "before_valid_pixels": int(before.valid.sum()),
            "after_valid_pixels": int(after.valid.sum()),
            "common_valid_pixels": common_count,
            "valid_coverage_percent": common_count / total * 100.0,
            "before_interpretation_suitable_pixels": int(before.interpretation_valid.sum()),
            "after_interpretation_suitable_pixels": int(after.interpretation_valid.sum()),
            "common_interpretation_suitable_pixels": interpretation_count,
            "interpretation_coverage_percent": interpretation_count / total * 100.0,
            "basis": "Analysis-valid is the intersection of numerical validity and SCL quality-mask validity; interpretation-suitable additionally requires a configured SCL land-cover class.",
        },
        "results": {
            "population": "common analysis-valid pixels across all quality-valid SCL classes",
            "before_ndvi": _stats(before.ndvi[common]),
            "after_ndvi": _stats(after.ndvi[common]),
            "delta_ndvi": _stats(delta[common]),
            "change_categories": fractions,
        },
        "interpretation_results": interpretation_results,
        "area": area,
        "area_omission_reason": area_warning,
        "visualization": visualization,
        "runtime": {
            "seconds": elapsed,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
        "warnings": warnings,
    }
    paths["summary"].write_text(_summary_text(record), encoding="utf-8")
    paths["execution_record"].write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare NDVI for two prepared Sentinel-2 Level-2A observations.")
    parser.add_argument("--config", required=True, type=Path, help="validated JSON input contract")
    parser.add_argument("--output-dir", type=Path, help="override the configuration's output_root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = run(args.config, args.output_dir)
    except VegetationValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": record["status"],
        "dates": record["dates"],
        "common_valid_pixels": record["validity"]["common_valid_pixels"],
        "results": record["results"],
        "interpretation_results": record["interpretation_results"],
        "output_dir": str(Path(record["outputs"]["execution_record"]).parent),
    }, indent=2))
    return 0
