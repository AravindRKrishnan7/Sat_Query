"""Bounded historical Sentinel-2 NDVI analysis for confirmed rectangular AOIs."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from satquery.geolens import create_unique_run_directory
from satquery.vegetation import (
    FLOAT_NODATA,
    calculate_ndvi,
    classify_delta,
    render_ndvi,
    render_ndvi_difference,
    render_true_color,
)


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
HISTORICAL_COLLECTION = "sentinel-2-c1-l2a"
DEMO_BOUNDS = (76.34, 9.48, 76.36, 9.50)
DEFAULT_AREA_LIMIT_SQUARE_KM = 25.0
DEFAULT_RESOLUTION_METRES = 10.0
DEFAULT_CANDIDATE_LIMIT = 5
DEFAULT_MINIMUM_YEAR_COVERAGE = 0.80
DEFAULT_MINIMUM_COMMON_COVERAGE = 0.50
DEFAULT_THRESHOLD = 0.10
SUPPORTED_FIRST_YEAR = 2017
MAX_ANNUAL_OBSERVATIONS = 5
VALID_SCL_CLASSES = (4, 5, 6)
REFLECTANCE_ASSETS = ("blue", "green", "red", "nir")
ALL_ASSETS = (*REFLECTANCE_ASSETS, "scl")


class HistoricalValidationError(ValueError):
    """Raised when a historical request or AOI is invalid."""


class HistoricalNetworkError(RuntimeError):
    """Raised when required remote metadata/data is unavailable and uncached."""


@dataclass(frozen=True)
class ConfirmedAOI:
    west: float
    south: float
    east: float
    north: float
    area_square_km: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    @property
    def geometry(self) -> dict[str, Any]:
        return {
            "type": "Polygon",
            "coordinates": [[
                [self.west, self.south],
                [self.east, self.south],
                [self.east, self.north],
                [self.west, self.north],
                [self.west, self.south],
            ]],
        }

    @property
    def cache_key(self) -> str:
        payload = json.dumps([round(value, 8) for value in self.bounds], separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def record(self) -> dict[str, Any]:
        return {
            "bounds_wgs84": list(self.bounds),
            "geometry_wgs84": self.geometry,
            "coordinate_precision_decimal_degrees": 8,
            "area_square_km": self.area_square_km,
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class SeasonalPolicy:
    name: str = "January-March demo window"
    start_month: int = 1
    start_day: int = 1
    end_month: int = 3
    end_day: int = 31

    def interval(self, year: int) -> tuple[str, str]:
        start = date(year, self.start_month, self.start_day)
        end = date(year, self.end_month, self.end_day)
        return start.isoformat(), end.isoformat()

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_month_day": f"{self.start_month:02d}-{self.start_day:02d}",
            "end_month_day": f"{self.end_month:02d}-{self.end_day:02d}",
            "note": "Configurable demo policy; not claimed to be universally optimal.",
        }


@dataclass(frozen=True)
class QualityRequirements:
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    minimum_year_valid_coverage: float = DEFAULT_MINIMUM_YEAR_COVERAGE
    minimum_common_valid_coverage: float = DEFAULT_MINIMUM_COMMON_COVERAGE
    valid_scl_classes: tuple[int, ...] = VALID_SCL_CLASSES
    denominator_epsilon: float = 1e-6
    change_threshold: float = DEFAULT_THRESHOLD

    def record(self) -> dict[str, Any]:
        return {
            "candidate_limit": self.candidate_limit,
            "minimum_year_valid_coverage": self.minimum_year_valid_coverage,
            "minimum_common_valid_coverage": self.minimum_common_valid_coverage,
            "valid_scl_classes": list(self.valid_scl_classes),
            "denominator_epsilon": self.denominator_epsilon,
            "change_threshold": self.change_threshold,
            "candidate_ranking": (
                "ascending catalogue eo:cloud_cover; select the first shortlisted candidate "
                "meeting the in-AOI valid-coverage requirement"
            ),
        }


@dataclass(frozen=True)
class HistoricalRequest:
    aoi: ConfirmedAOI
    start_year: int
    end_year: int
    analysis: str
    seasonal_policy: SeasonalPolicy
    quality: QualityRequirements

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(range(self.start_year, self.end_year + 1))

    def record(self) -> dict[str, Any]:
        return {
            "aoi": self.aoi.record(),
            "start_year": self.start_year,
            "end_year": self.end_year,
            "years": list(self.years),
            "annual_observations_requested": len(self.years),
            "adjacent_comparisons_requested": max(0, len(self.years) - 1),
            "analysis": self.analysis,
            "seasonal_policy": self.seasonal_policy.record(),
            "data_quality_requirements": self.quality.record(),
        }


@dataclass(frozen=True)
class AnalysisGrid:
    width: int
    height: int
    crs: Any
    transform: Any
    bounds: tuple[float, float, float, float]
    resolution_metres: float
    aoi_mask: np.ndarray

    def record(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "bounds": list(self.bounds),
            "resolution_metres": self.resolution_metres,
            "aoi_pixels": int(self.aoi_mask.sum()),
            "continuous_resampling": "bilinear",
            "categorical_resampling": "nearest-neighbour",
        }


@dataclass
class AssetData:
    values: np.ndarray
    valid: np.ndarray
    cache_status: str
    record: dict[str, Any]


@dataclass
class AnnualData:
    year: int
    date: str
    item_id: str
    ndvi: np.ndarray
    valid: np.ndarray
    true_color: np.ndarray
    true_color_valid: np.ndarray
    record: dict[str, Any]


def approximate_rectangle_area_square_km(bounds: Sequence[float]) -> float:
    if len(bounds) != 4:
        raise HistoricalValidationError("AOI bounds must contain west, south, east and north")
    west, south, east, north = (float(value) for value in bounds)
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise HistoricalValidationError("AOI bounds must be finite")
    if west >= east or south >= north:
        raise HistoricalValidationError("AOI bounds must have positive width and height")
    if west < -180 or east > 180 or south < -90 or north > 90:
        raise HistoricalValidationError("AOI bounds fall outside valid longitude/latitude ranges")
    radius_km = 6371.0088
    mid_latitude = math.radians((south + north) / 2.0)
    width_km = radius_km * math.cos(mid_latitude) * math.radians(east - west)
    height_km = radius_km * math.radians(north - south)
    return abs(width_km * height_km)


def confirm_aoi(
    bounds: Sequence[float],
    area_limit_square_km: float = DEFAULT_AREA_LIMIT_SQUARE_KM,
) -> ConfirmedAOI:
    if not math.isfinite(area_limit_square_km) or area_limit_square_km <= 0:
        raise HistoricalValidationError("AOI area limit must be positive")
    canonical_bounds = tuple(round(float(value), 8) for value in bounds)
    area = approximate_rectangle_area_square_km(canonical_bounds)
    if area > area_limit_square_km:
        raise HistoricalValidationError(
            f"selected area is approximately {area:.2f} km²; the demo limit is "
            f"{area_limit_square_km:.2f} km²"
        )
    west, south, east, north = canonical_bounds
    return ConfirmedAOI(west, south, east, north, area)


def bounds_from_geojson(geometry: Mapping[str, Any]) -> tuple[float, float, float, float]:
    if geometry.get("type") != "Polygon":
        raise HistoricalValidationError("draw a rectangle; other geometry types are unsupported")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates or not isinstance(coordinates[0], list):
        raise HistoricalValidationError("drawn rectangle geometry is malformed")
    ring = coordinates[0]
    try:
        longitudes = [float(point[0]) for point in ring]
        latitudes = [float(point[1]) for point in ring]
    except (TypeError, ValueError, IndexError) as exc:
        raise HistoricalValidationError("drawn rectangle coordinates are malformed") from exc
    bounds = (min(longitudes), min(latitudes), max(longitudes), max(latitudes))
    approximate_rectangle_area_square_km(bounds)
    return bounds


def update_aoi_state(state: MutableMapping[str, Any], bounds: Sequence[float] | None) -> bool:
    """Store a draft AOI and invalidate confirmation/results when it changes."""
    canonical = None if bounds is None else tuple(round(float(value), 8) for value in bounds)
    previous = state.get("historical_draft_bounds")
    if previous == canonical:
        return False
    state["historical_draft_bounds"] = canonical
    state.pop("historical_confirmed_aoi", None)
    state.pop("historical_request", None)
    state.pop("historical_result", None)
    state.pop("assistant_plan", None)
    state.pop("assistant_execution", None)
    state.pop("historical_answer", None)
    state.pop("historical_chat", None)
    return True


def build_request(
    aoi: ConfirmedAOI | None,
    start_year: int,
    end_year: int,
    analysis: str = "vegetation_ndvi",
    seasonal_policy: SeasonalPolicy | None = None,
    quality: QualityRequirements | None = None,
    current_year: int | None = None,
) -> HistoricalRequest:
    if aoi is None:
        raise HistoricalValidationError("confirm a valid AOI before creating an analysis request")
    if analysis != "vegetation_ndvi":
        if analysis in {"construction", "roads", "construction_and_roads"}:
            raise HistoricalValidationError(
                "Historical construction/road analysis is unsupported for the Sentinel-2 timeline. "
                "Supply suitable aligned imagery to the existing construction workflow instead."
            )
        raise HistoricalValidationError(f"unsupported historical analysis: {analysis}")
    if (
        isinstance(start_year, bool)
        or isinstance(end_year, bool)
        or not isinstance(start_year, int)
        or not isinstance(end_year, int)
    ):
        raise HistoricalValidationError("years must be integers")
    last_supported = current_year or datetime.now(timezone.utc).year
    if start_year > end_year:
        raise HistoricalValidationError("start year must not be later than end year")
    if start_year < SUPPORTED_FIRST_YEAR or end_year > last_supported:
        raise HistoricalValidationError(
            f"supported years are {SUPPORTED_FIRST_YEAR} through {last_supported}"
        )
    count = end_year - start_year + 1
    if count > MAX_ANNUAL_OBSERVATIONS:
        raise HistoricalValidationError(
            f"request contains {count} annual observations; the demo limit is {MAX_ANNUAL_OBSERVATIONS}"
        )
    if count < 2:
        raise HistoricalValidationError("historical comparison requires at least two annual observations")
    policy = seasonal_policy or SeasonalPolicy()
    requirements = quality or QualityRequirements()
    if requirements.candidate_limit < 1 or requirements.candidate_limit > 20:
        raise HistoricalValidationError("candidate limit must be between 1 and 20")
    for label, value in (
        ("minimum yearly coverage", requirements.minimum_year_valid_coverage),
        ("minimum common coverage", requirements.minimum_common_valid_coverage),
    ):
        if not math.isfinite(value) or not 0 < value <= 1:
            raise HistoricalValidationError(f"{label} must be greater than 0 and at most 1")
    if (
        not math.isfinite(requirements.denominator_epsilon)
        or requirements.denominator_epsilon <= 0
    ):
        raise HistoricalValidationError("denominator epsilon must be positive and finite")
    if (
        not math.isfinite(requirements.change_threshold)
        or not 0 <= requirements.change_threshold <= 2
    ):
        raise HistoricalValidationError("NDVI change threshold must be between 0 and 2")
    if not requirements.valid_scl_classes or any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 11
        for value in requirements.valid_scl_classes
    ):
        raise HistoricalValidationError("valid SCL classes must be integer labels from 1 through 11")
    return HistoricalRequest(aoi, start_year, end_year, analysis, policy, requirements)


def build_analysis_grid(aoi: ConfirmedAOI, resolution_metres: float = 10.0) -> AnalysisGrid:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform, transform_bounds

    if not math.isfinite(resolution_metres) or resolution_metres <= 0:
        raise HistoricalValidationError("analysis resolution must be positive")
    centre_longitude = (aoi.west + aoi.east) / 2.0
    centre_latitude = (aoi.south + aoi.north) / 2.0
    zone = int((centre_longitude + 180.0) // 6.0) + 1
    epsg = (32600 if centre_latitude >= 0 else 32700) + zone
    crs = rasterio.crs.CRS.from_epsg(epsg)
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", crs, *aoi.bounds, densify_pts=21
    )
    left = math.floor(left / resolution_metres) * resolution_metres
    bottom = math.floor(bottom / resolution_metres) * resolution_metres
    right = math.ceil(right / resolution_metres) * resolution_metres
    top = math.ceil(top / resolution_metres) * resolution_metres
    width = int(round((right - left) / resolution_metres))
    height = int(round((top - bottom) / resolution_metres))
    if width < 1 or height < 1 or width * height > 400_000:
        raise HistoricalValidationError(
            f"analysis grid would contain {width * height:,} pixels; reduce the AOI"
        )
    affine = from_origin(left, top, resolution_metres, resolution_metres)
    rows, columns = np.indices((height, width), dtype=np.float64)
    xs = affine.c + (columns + 0.5) * affine.a
    ys = affine.f + (rows + 0.5) * affine.e
    longitudes, latitudes = transform(
        crs, "EPSG:4326", xs.reshape(-1).tolist(), ys.reshape(-1).tolist()
    )
    longitude_array = np.asarray(longitudes).reshape(height, width)
    latitude_array = np.asarray(latitudes).reshape(height, width)
    aoi_mask = (
        (longitude_array >= aoi.west)
        & (longitude_array <= aoi.east)
        & (latitude_array >= aoi.south)
        & (latitude_array <= aoi.north)
    )
    if not aoi_mask.any():
        raise HistoricalValidationError("confirmed AOI contains no analysis-grid pixel centres")
    return AnalysisGrid(width, height, crs, affine, (left, bottom, right, top), resolution_metres, aoi_mask)


def resolve_reflectance_scaling(
    collection: str,
    item_properties: Mapping[str, Any],
    asset: Mapping[str, Any],
    raster_scale: float,
    raster_offset: float,
) -> tuple[float, float, str]:
    bands = asset.get("raster:bands")
    band = bands[0] if isinstance(bands, list) and bands and isinstance(bands[0], Mapping) else {}
    if "scale" not in band or "offset" not in band:
        raise HistoricalValidationError(
            "reflectance asset lacks explicit item-level raster:bands scale/offset metadata"
        )
    try:
        declared_scale = float(band["scale"])
        declared_offset = float(band["offset"])
    except (TypeError, ValueError) as exc:
        raise HistoricalValidationError(
            "reflectance asset has non-numeric item-level scale/offset metadata"
        ) from exc
    if not all(
        math.isfinite(value)
        for value in (declared_scale, declared_offset, raster_scale, raster_offset)
    ):
        raise HistoricalValidationError("reflectance scale/offset metadata must be finite")
    if collection == "sentinel-2-c1-l2a":
        if not (
            math.isclose(raster_scale, declared_scale, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(raster_offset, declared_offset, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise HistoricalValidationError(
                "Collection-1 item asset scaling disagrees with the remote COG header: "
                f"item=({declared_scale}, {declared_offset}), raster=({raster_scale}, {raster_offset})"
            )
        return (
            declared_scale,
            declared_offset,
            "sentinel-2-c1-l2a item raster:bands, independently matched to remote COG header",
        )
    if collection == "sentinel-2-l2a":
        applied = item_properties.get("earthsearch:boa_offset_applied")
        if applied is True:
            return (
                declared_scale,
                0.0,
                "legacy Earth Search earthsearch:boa_offset_applied=true; contradictory item offset ignored",
            )
        if applied is False:
            return (
                declared_scale,
                declared_offset,
                "legacy Earth Search earthsearch:boa_offset_applied=false; item asset offset retained",
            )
        raise HistoricalValidationError(
            "legacy Earth Search item lacks earthsearch:boa_offset_applied; scaling representation is ambiguous"
        )
    raise HistoricalValidationError(f"unsupported reflectance collection: {collection}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _post_json(url: str, payload: Mapping[str, Any], timeout_seconds: float = 30.0) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(3):
        request = Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json", "User-Agent": "GeoLens-local-research/0.1"},
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - fixed provider URL.
                value = json.load(response)
            if not isinstance(value, dict):
                raise HistoricalNetworkError("Earth Search returned a non-object JSON response")
            return value
        except (OSError, HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    raise HistoricalNetworkError(f"Earth Search request failed after 3 attempts: {last_error}")


def search_candidates(
    request: HistoricalRequest,
    year: int,
    cache_root: Path,
    transport: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    start, end = request.seasonal_policy.interval(year)
    cache_path = cache_root / request.aoi.cache_key / "search" / f"{year}.json"
    if cache_path.is_file() and not refresh:
        value = json.loads(cache_path.read_text(encoding="utf-8"))
        return list(value["features"]), "cache"
    payload = {
        "collections": [HISTORICAL_COLLECTION],
        "bbox": list(request.aoi.bounds),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 100,
    }
    sender = transport or _post_json
    try:
        value = sender(f"{EARTH_SEARCH_URL}/search", payload)
        features = value.get("features")
        if not isinstance(features, list):
            raise HistoricalNetworkError("Earth Search response has no feature list")
        cache_value = {
            "provider": "Element 84 Earth Search",
            "collection": HISTORICAL_COLLECTION,
            "request": payload,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "features": features,
        }
        _atomic_json(cache_path, cache_value)
        return list(features), "retrieved"
    except Exception as exc:
        if cache_path.is_file():
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(value["features"]), "cache_after_network_error"
        if isinstance(exc, HistoricalNetworkError):
            raise
        raise HistoricalNetworkError(f"Earth Search search failed: {exc}") from exc


def _asset_cache_paths(cache_root: Path, aoi_key: str, item_id: str, asset_key: str) -> tuple[Path, Path]:
    safe_item = "".join(character if character.isalnum() or character in "-_" else "_" for character in item_id)
    directory = cache_root / aoi_key / "assets" / safe_item
    return directory / f"{asset_key}.tif", directory / f"{asset_key}.json"


def _write_cached_raster(path: Path, values: np.ndarray, grid: AnalysisGrid, nodata: float, dtype: str) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.tif")
    output = np.where(np.isfinite(values), values, nodata).astype(dtype)
    with rasterio.open(
        temporary,
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
    temporary.replace(path)


def _read_cached_asset(path: Path, metadata_path: Path, asset_key: str) -> AssetData:
    import rasterio

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with rasterio.open(path) as dataset:
        values = dataset.read(1).astype(np.float64 if asset_key != "scl" else np.uint8)
        if asset_key == "scl":
            valid = values != 0
        else:
            values[values == dataset.nodata] = np.nan
            valid = np.isfinite(values)
    return AssetData(values, valid, "cache", metadata)


def _read_remote_asset(
    item: Mapping[str, Any],
    asset_key: str,
    grid: AnalysisGrid,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    assets = item.get("assets")
    if not isinstance(assets, Mapping) or asset_key not in assets:
        raise HistoricalNetworkError(f"item {item.get('id')} has no {asset_key!r} asset")
    asset = assets[asset_key]
    href = asset.get("href")
    if not isinstance(href, str) or not href.startswith("https://"):
        raise HistoricalNetworkError(f"item {item.get('id')} has an unusable {asset_key!r} href")
    environment = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_CONNECTTIMEOUT": "10",
        "GDAL_HTTP_TIMEOUT": "30",
        "GDAL_HTTP_MAX_RETRY": "2",
    }
    try:
        with rasterio.Env(**environment), rasterio.open(href) as source:
            if source.count != 1 or source.crs is None:
                raise HistoricalNetworkError(f"unexpected raster contract: {href}")
            source_bounds = transform_bounds(grid.crs, source.crs, *grid.bounds, densify_pts=21)
            raw_window = from_bounds(*source_bounds, transform=source.transform)
            left = max(0, math.floor(raw_window.col_off) - 1)
            top = max(0, math.floor(raw_window.row_off) - 1)
            right = min(source.width, math.ceil(raw_window.col_off + raw_window.width) + 1)
            bottom = min(source.height, math.ceil(raw_window.row_off + raw_window.height) + 1)
            if right <= left or bottom <= top:
                raise HistoricalNetworkError(f"asset does not overlap the confirmed AOI: {href}")
            window = Window(left, top, right - left, bottom - top)
            source_values = source.read(1, window=window)
            source_transform = source.window_transform(window)
            if asset_key == "scl":
                if np.dtype(source.dtypes[0]).kind not in "iu":
                    raise HistoricalNetworkError(f"SCL asset is not integer categorical data: {href}")
                destination = np.zeros((grid.height, grid.width), dtype=np.uint8)
                reproject(
                    source=source_values,
                    destination=destination,
                    src_transform=source_transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=grid.transform,
                    dst_crs=grid.crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                valid = grid.aoi_mask & (destination != 0)
                record = {
                    "item_id": item.get("id"),
                    "asset": asset_key,
                    "href": href,
                    "source_dtype": source.dtypes[0],
                    "source_nodata": source.nodata,
                    "source_crs": source.crs.to_string(),
                    "source_window": [window.col_off, window.row_off, window.width, window.height],
                    "resampling": "nearest-neighbour",
                    "representation": "Sentinel-2 Scene Classification Layer labels",
                }
                return destination, valid, record

            raster_scale = float(source.scales[0])
            raster_offset = float(source.offsets[0])
            collection = item.get("collection")
            if collection != HISTORICAL_COLLECTION:
                raise HistoricalValidationError(
                    f"unexpected STAC collection for selected item: {collection!r}"
                )
            scale, offset, scaling_source = resolve_reflectance_scaling(
                collection,
                item.get("properties", {}),
                asset,
                raster_scale,
                raster_offset,
            )
            destination_raw = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
            reproject(
                source=source_values,
                destination=destination_raw,
                src_transform=source_transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            valid = grid.aoi_mask & np.isfinite(destination_raw) & (destination_raw != 0)
            # The AOI cache stores float32 reflectance, so return the same representation on
            # a first retrieval and a cache hit to keep repeat executions deterministic.
            reflectance = (
                destination_raw.astype(np.float32) * np.float32(scale) + np.float32(offset)
            ).astype(np.float32)
            reflectance[~valid] = np.nan
            record = {
                "item_id": item.get("id"),
                "asset": asset_key,
                "href": href,
                "source_dtype": source.dtypes[0],
                "source_nodata": source.nodata,
                "source_crs": source.crs.to_string(),
                "source_window": [window.col_off, window.row_off, window.width, window.height],
                "raster_scale": raster_scale,
                "raster_offset": raster_offset,
                "applied_scale": scale,
                "applied_offset": offset,
                "scale_offset_source": scaling_source,
                "resampling": "bilinear",
                "representation": "physical surface reflectance after recorded affine scaling",
                "negative_reflectance_count": int(np.count_nonzero(valid & (reflectance < 0))),
                "zero_reflectance_count": int(np.count_nonzero(valid & (reflectance == 0))),
            }
            return reflectance, valid, record
    except HistoricalValidationError:
        raise
    except (OSError, rasterio.errors.RasterioError) as exc:
        raise HistoricalNetworkError(f"remote AOI window read failed for {href}: {exc}") from exc


def load_asset(
    item: Mapping[str, Any],
    asset_key: str,
    grid: AnalysisGrid,
    aoi_key: str,
    cache_root: Path,
    reader: Callable[[Mapping[str, Any], str, AnalysisGrid], tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None = None,
    refresh: bool = False,
) -> AssetData:
    if asset_key not in ALL_ASSETS:
        raise HistoricalValidationError(f"unsupported asset key: {asset_key}")
    item_id = str(item.get("id") or "")
    if not item_id:
        raise HistoricalValidationError("STAC item has no ID")
    raster_path, metadata_path = _asset_cache_paths(cache_root, aoi_key, item_id, asset_key)
    if raster_path.is_file() and metadata_path.is_file() and not refresh:
        return _read_cached_asset(raster_path, metadata_path, asset_key)
    remote_reader = reader or _read_remote_asset
    try:
        values, valid, record = remote_reader(item, asset_key, grid)
        record = {
            **record,
            "collection": HISTORICAL_COLLECTION,
            "target_grid": grid.record(),
            "cache_path": str(raster_path.resolve()),
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if asset_key == "scl":
            _write_cached_raster(raster_path, values, grid, 0, "uint8")
        else:
            _write_cached_raster(raster_path, values, grid, FLOAT_NODATA, "float32")
        _atomic_json(metadata_path, record)
        return AssetData(values, valid, "retrieved", record)
    except HistoricalValidationError:
        raise
    except HistoricalNetworkError:
        if raster_path.is_file() and metadata_path.is_file():
            cached = _read_cached_asset(raster_path, metadata_path, asset_key)
            cached.cache_status = "cache_after_network_error"
            return cached
        raise
    except Exception as exc:
        raise HistoricalNetworkError(f"asset retrieval failed for {item_id}/{asset_key}: {exc}") from exc


def _catalogue_cloud(item: Mapping[str, Any]) -> float:
    value = item.get("properties", {}).get("eo:cloud_cover")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf
    return number if math.isfinite(number) else math.inf


def select_annual_item(
    request: HistoricalRequest,
    year: int,
    grid: AnalysisGrid,
    cache_root: Path,
    transport: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    asset_reader: Callable[[Mapping[str, Any], str, AnalysisGrid], tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any] | None, AssetData | None, dict[str, Any]]:
    items, search_status = search_candidates(request, year, cache_root, transport=transport)
    ordered = sorted(items, key=lambda item: (_catalogue_cloud(item), str(item.get("id", ""))))
    shortlisted = ordered[: request.quality.candidate_limit]
    attempts: list[dict[str, Any]] = []
    total_aoi = int(grid.aoi_mask.sum())
    for item in shortlisted:
        attempt = {
            "item_id": item.get("id"),
            "datetime": item.get("properties", {}).get("datetime"),
            "catalogue_cloud_cover_percent": item.get("properties", {}).get("eo:cloud_cover"),
        }
        try:
            scl = load_asset(
                item, "scl", grid, request.aoi.cache_key, cache_root, reader=asset_reader
            )
            quality_valid = grid.aoi_mask & np.isin(
                scl.values, np.asarray(request.quality.valid_scl_classes)
            )
            coverage = int(quality_valid.sum()) / total_aoi
            attempt.update({
                "in_aoi_valid_coverage": coverage,
                "scl_cache_status": scl.cache_status,
                "accepted": coverage >= request.quality.minimum_year_valid_coverage,
            })
            attempts.append(attempt)
            if attempt["accepted"]:
                return item, scl, {
                    "search_cache_status": search_status,
                    "catalogue_candidates": len(items),
                    "shortlist_limit": request.quality.candidate_limit,
                    "selection_attempts": attempts,
                    "selected_item_id": item.get("id"),
                }
        except (HistoricalNetworkError, HistoricalValidationError) as exc:
            attempt.update({"accepted": False, "error": str(exc)})
            attempts.append(attempt)
    return None, None, {
        "search_cache_status": search_status,
        "catalogue_candidates": len(items),
        "shortlist_limit": request.quality.candidate_limit,
        "selection_attempts": attempts,
        "selected_item_id": None,
        "reason": (
            "No shortlisted candidate met the minimum in-AOI valid coverage, or required SCL "
            "windows were unavailable. The seasonal window was not changed."
        ),
    }


def _write_analysis_raster(path: Path, values: np.ndarray, grid: AnalysisGrid, nodata: float, dtype: str) -> None:
    _write_cached_raster(path, values, grid, nodata, dtype)


def prepare_annual_data(
    request: HistoricalRequest,
    year: int,
    item: Mapping[str, Any],
    selected_scl: AssetData,
    selection_record: Mapping[str, Any],
    grid: AnalysisGrid,
    cache_root: Path,
    run_directory: Path,
    asset_reader: Callable[[Mapping[str, Any], str, AnalysisGrid], tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None = None,
) -> AnnualData:
    assets: dict[str, AssetData] = {"scl": selected_scl}
    for key in REFLECTANCE_ASSETS:
        assets[key] = load_asset(
            item, key, grid, request.aoi.cache_key, cache_root, reader=asset_reader
        )
    band_valid = grid.aoi_mask.copy()
    for key in REFLECTANCE_ASSETS:
        band_valid &= assets[key].valid
    quality_valid = grid.aoi_mask & np.isin(
        assets["scl"].values, np.asarray(request.quality.valid_scl_classes)
    )
    red = assets["red"].values
    nir = assets["nir"].values
    _, _, raw_ndvi, numerical_valid = calculate_ndvi(
        red,
        nir,
        1.0,
        0.0,
        1.0,
        0.0,
        band_valid,
        request.quality.denominator_epsilon,
    )
    valid = numerical_valid & quality_valid
    ndvi = np.where(valid, raw_ndvi, np.nan)
    denominator = red + nir
    near_zero = band_valid & np.isfinite(denominator) & (
        np.abs(denominator) <= request.quality.denominator_epsilon
    )
    outside = numerical_valid & ((raw_ndvi < -1.0) | (raw_ndvi > 1.0))
    true_color = np.stack(
        (assets["red"].values, assets["green"].values, assets["blue"].values), axis=-1
    )
    true_color_valid = band_valid & np.all(np.isfinite(true_color), axis=-1)
    item_properties = item.get("properties", {})
    acquisition_datetime = str(item_properties.get("datetime") or "")
    acquisition_date = acquisition_datetime[:10]
    stem = f"{year}-{acquisition_date}"
    annual_directory = run_directory / "annual"
    annual_directory.mkdir(parents=True, exist_ok=True)
    ndvi_path = annual_directory / f"{stem}-ndvi.tif"
    valid_path = annual_directory / f"{stem}-valid-mask.tif"
    ndvi_png = annual_directory / f"{stem}-ndvi.png"
    true_color_png = annual_directory / f"{stem}-true-color.png"
    _write_analysis_raster(ndvi_path, ndvi, grid, FLOAT_NODATA, "float32")
    _write_analysis_raster(valid_path, valid.astype(np.uint8), grid, 0, "uint8")
    render_ndvi(ndvi).save(ndvi_png)
    true_color_image, display_counts = render_true_color(true_color, true_color_valid)
    true_color_image.save(true_color_png)
    total_aoi = int(grid.aoi_mask.sum())
    valid_count = int(valid.sum())
    values = ndvi[valid]
    record = {
        "year": year,
        "status": "available",
        "item_id": item.get("id"),
        "acquisition_datetime": acquisition_datetime,
        "acquisition_date": acquisition_date,
        "catalogue_cloud_cover_percent": item_properties.get("eo:cloud_cover"),
        "processing_baseline": item_properties.get("s2:processing_baseline"),
        "platform": item_properties.get("platform"),
        "tile": item_properties.get("grid:code") or item_properties.get("s2:mgrs_tile"),
        "selection": dict(selection_record),
        "valid_pixels": valid_count,
        "aoi_pixels": total_aoi,
        "valid_coverage": valid_count / total_aoi,
        "per_year_population": "quality-mask-valid and numerically valid pixels for this observation",
        "per_year_ndvi": {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        },
        "numerical_diagnostics": {
            "band_data_valid_pixels": int(band_valid.sum()),
            "quality_mask_valid_pixels": int(quality_valid.sum()),
            "numerically_valid_ndvi_pixels": int(numerical_valid.sum()),
            "near_zero_denominator_pixels": int(near_zero.sum()),
            "ndvi_outside_minus1_plus1_pixels": int(outside.sum()),
            "note": "NDVI is retained without clipping; unstable denominators are excluded.",
        },
        "scaling_and_assets": {key: assets[key].record for key in ALL_ASSETS},
        "cache_status": {key: assets[key].cache_status for key in ALL_ASSETS},
        "display": {
            "true_color_band_order": ["red/B04", "green/B03", "blue/B02"],
            "reflectance_range": [0.0, 0.3],
            "gamma": 2.2,
            "display_counts": display_counts,
            "note": "Display-only; numerical reflectance and NDVI are unchanged.",
        },
        "outputs": {
            "true_color_png": str(true_color_png.resolve()),
            "ndvi_png": str(ndvi_png.resolve()),
            "ndvi_geotiff": str(ndvi_path.resolve()),
            "valid_mask_geotiff": str(valid_path.resolve()),
        },
    }
    return AnnualData(
        year,
        acquisition_date,
        str(item.get("id")),
        ndvi,
        valid,
        true_color,
        true_color_valid,
        record,
    )


def common_trend(
    annual: Sequence[AnnualData],
    aoi_mask: np.ndarray,
    minimum_coverage: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not annual:
        empty = np.zeros(aoi_mask.shape, dtype=bool)
        return empty, {
            "sufficient": False,
            "coverage": 0.0,
            "pixels": 0,
            "reason": "No annual observations are available.",
            "annual_statistics": [],
        }
    common = aoi_mask.copy()
    for observation in annual:
        common &= observation.valid
    count = int(common.sum())
    total = int(aoi_mask.sum())
    coverage = count / total
    sufficient = len(annual) >= 2 and count > 0 and coverage >= minimum_coverage
    statistics: list[dict[str, Any]] = []
    if sufficient:
        for observation in annual:
            values = observation.ndvi[common]
            statistics.append({
                "year": observation.year,
                "acquisition_date": observation.date,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            })
    return common, {
        "sufficient": sufficient,
        "coverage": coverage,
        "pixels": count,
        "aoi_pixels": total,
        "minimum_required_coverage": minimum_coverage,
        "included_years": [observation.year for observation in annual],
        "population": "pixels valid across every available included observation",
        "reason": (
            None
            if sufficient
            else (
                "At least two annual observations are required for a trend."
                if len(annual) < 2
                else "Common-valid coverage is below the configured minimum."
            )
        ),
        "annual_statistics": statistics,
    }


def _difference_record(
    label: str,
    before: AnnualData,
    after: AnnualData,
    grid: AnalysisGrid,
    run_directory: Path,
    threshold: float,
) -> dict[str, Any]:
    valid = before.valid & after.valid
    delta = np.full(before.ndvi.shape, np.nan, dtype=np.float64)
    delta[valid] = after.ndvi[valid] - before.ndvi[valid]
    directory = run_directory / "differences"
    directory.mkdir(parents=True, exist_ok=True)
    path_stem = label.replace(" ", "-")
    tif_path = directory / f"{path_stem}.tif"
    png_path = directory / f"{path_stem}.png"
    _write_analysis_raster(tif_path, delta, grid, FLOAT_NODATA, "float32")
    render_ndvi_difference(delta).save(png_path)
    categories = classify_delta(delta, threshold)
    count = int(valid.sum())
    return {
        "status": "available",
        "label": label,
        "before_year": before.year,
        "after_year": after.year,
        "before_date": before.date,
        "after_date": after.date,
        "valid_pixels": count,
        "aoi_pixels": int(grid.aoi_mask.sum()),
        "valid_coverage": count / int(grid.aoi_mask.sum()),
        "threshold": threshold,
        "categories": {
            name: {"pixels": value, "fraction": value / count if count else None}
            for name, value in categories.items()
        },
        "mean": float(np.mean(delta[valid])) if count else None,
        "median": float(np.median(delta[valid])) if count else None,
        "outputs": {
            "difference_png": str(png_path.resolve()),
            "difference_geotiff": str(tif_path.resolve()),
        },
        "interpretation": "NDVI difference only; not statistical significance, vegetation extent, health, or cause.",
    }


def deterministic_summary(record: Mapping[str, Any]) -> str:
    available = [item for item in record["annual_observations"] if item["status"] == "available"]
    missing = [item["year"] for item in record["annual_observations"] if item["status"] != "available"]
    lines = [
        "GeoLens historical Sentinel-2 NDVI timeline",
        f"Confirmed AOI bounds (WGS84): {record['request']['aoi']['bounds_wgs84']}",
        f"Approximate AOI area: {record['request']['aoi']['area_square_km']:.3f} km²",
        f"Requested years: {record['request']['years']}",
        "Actual acquisitions: "
        + (", ".join(f"{item['year']}: {item['acquisition_date']}" for item in available) or "none"),
        f"Unavailable years: {missing or 'none'}",
    ]
    trend = record["common_valid_trend"]
    lines.append(
        f"All-observation common-valid coverage: {trend['coverage']:.2%} "
        f"({trend['pixels']} of {trend.get('aoi_pixels', 0)} AOI pixels)."
    )
    if trend["sufficient"] and trend["annual_statistics"]:
        first = trend["annual_statistics"][0]
        last = trend["annual_statistics"][-1]
        lines.append(
            "Directly comparable common-pixel mean NDVI: "
            f"{first['year']} {first['mean']:.4f}; {last['year']} {last['mean']:.4f}; "
            f"difference {last['mean'] - first['mean']:+.4f}."
        )
    else:
        lines.append(f"Directly comparable trend withheld: {trend['reason']}")
    lines.append(
        "Threshold categories are NDVI increase/decrease/approximately stable only. "
        "They do not establish vegetation gain/loss, plant health, deforestation, damage, or cause."
    )
    return "\n".join(lines) + "\n"


def run_historical(
    request: HistoricalRequest,
    output_root: Path,
    cache_root: Path,
    transport: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    asset_reader: Callable[[Mapping[str, Any], str, AnalysisGrid], tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    grid = build_analysis_grid(request.aoi)
    run_directory = create_unique_run_directory(output_root, "run")
    annual_data: dict[int, AnnualData] = {}
    annual_records: list[dict[str, Any]] = []
    for year in request.years:
        try:
            item, scl, selection = select_annual_item(
                request,
                year,
                grid,
                cache_root,
                transport=transport,
                asset_reader=asset_reader,
            )
            if item is None or scl is None:
                annual_records.append({
                    "year": year,
                    "status": "unavailable",
                    "reason": selection.get("reason"),
                    "selection": selection,
                })
                continue
            prepared = prepare_annual_data(
                request,
                year,
                item,
                scl,
                selection,
                grid,
                cache_root,
                run_directory,
                asset_reader=asset_reader,
            )
            annual_data[year] = prepared
            annual_records.append(prepared.record)
        except (HistoricalNetworkError, HistoricalValidationError, OSError, RuntimeError) as exc:
            annual_records.append({
                "year": year,
                "status": "unavailable",
                "reason": str(exc),
            })

    available = [annual_data[year] for year in request.years if year in annual_data]
    common_mask, common_record = common_trend(
        available, grid.aoi_mask, request.quality.minimum_common_valid_coverage
    )
    if common_mask.any():
        common_path = run_directory / "common-valid-mask.tif"
        _write_analysis_raster(common_path, common_mask.astype(np.uint8), grid, 0, "uint8")
        common_record["mask_geotiff"] = str(common_path.resolve())

    adjacent: list[dict[str, Any]] = []
    for before_year, after_year in zip(request.years, request.years[1:]):
        if before_year not in annual_data or after_year not in annual_data:
            adjacent.append({
                "status": "unavailable",
                "label": f"{before_year}-to-{after_year}",
                "before_year": before_year,
                "after_year": after_year,
                "reason": "One or both requested annual observations are unavailable; no gap was bridged.",
            })
            continue
        adjacent.append(
            _difference_record(
                f"{before_year}-to-{after_year}",
                annual_data[before_year],
                annual_data[after_year],
                grid,
                run_directory,
                request.quality.change_threshold,
            )
        )
    first_last = None
    if request.start_year in annual_data and request.end_year in annual_data:
        first_last = _difference_record(
            f"{request.start_year}-to-{request.end_year}-first-last",
            annual_data[request.start_year],
            annual_data[request.end_year],
            grid,
            run_directory,
            request.quality.change_threshold,
        )
    else:
        first_last = {
            "status": "unavailable",
            "before_year": request.start_year,
            "after_year": request.end_year,
            "reason": "First-to-last difference requires both requested endpoint observations.",
        }

    retrieved_assets = 0
    cached_assets = 0
    for item in annual_records:
        for status in item.get("cache_status", {}).values():
            if status == "retrieved":
                retrieved_assets += 1
            else:
                cached_assets += 1
    missing_years = [item["year"] for item in annual_records if item["status"] != "available"]
    record: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "Historical Sentinel-2 annual NDVI timeline",
        "status": "success" if available and not missing_years else ("partial" if available else "no_data"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": request.record(),
        "provider": {
            "name": "Element 84 Earth Search",
            "stac_api": EARTH_SEARCH_URL,
            "collection": HISTORICAL_COLLECTION,
            "collection_metadata": f"{EARTH_SEARCH_URL}/collections/{HISTORICAL_COLLECTION}",
            "documentation": "https://github.com/Element84/earth-search",
        },
        "grid": grid.record(),
        "annual_observations": annual_records,
        "available_years": [item.year for item in available],
        "missing_years": missing_years,
        "common_valid_trend": common_record,
        "adjacent_differences": adjacent,
        "first_to_last_difference": first_last,
        "cache": {
            "root": str(cache_root.resolve()),
            "matching_aoi_key": request.aoi.cache_key,
            "selected_asset_windows_retrieved": retrieved_assets,
            "selected_asset_windows_loaded_from_cache": cached_assets,
            "note": "Cache keys include the exact confirmed AOI; another AOI is never used as fallback.",
        },
        "limitations": [
            "January-March is a configurable demo window, not a universally optimal seasonal policy.",
            "Catalogue cloud cover only shortlists candidates; acceptance uses SCL coverage inside the AOI.",
            "SCL is a coarse classification and can contain errors or residual contamination.",
            "NDVI differences do not establish vegetation extent loss, plant health, deforestation, or cause.",
            "Annual observations are single acquisitions and do not represent within-season variability.",
        ],
        "runtime_seconds": time.perf_counter() - started,
        "outputs": {},
    }
    summary_path = run_directory / "summary.txt"
    record_path = run_directory / "run.json"
    summary_path.write_text(deterministic_summary(record), encoding="utf-8")
    record["outputs"] = {
        "summary": str(summary_path.resolve()),
        "execution_record": str(record_path.resolve()),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
