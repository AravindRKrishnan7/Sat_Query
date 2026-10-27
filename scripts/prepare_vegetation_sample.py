#!/usr/bin/env python3
"""Download fixed, small Earth Search COG windows for the documented real example.

This is deliberately not a live catalog search. It reads one 2.56 km square
window from six pinned public assets and writes prepared local GeoTIFFs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402


ROOT = PROJECT_ROOT / "data" / "vegetation" / "earth-search-bengaluru"
BASE_URL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/P/GQ"
SAMPLES = {
    "before": {
        "item": "S2A_43PGQ_20240207_0_L2A",
        "prefix": f"{BASE_URL}/2024/2/S2A_43PGQ_20240207_0_L2A",
    },
    "after": {
        "item": "S2B_43PGQ_20250206_0_L2A",
        "prefix": f"{BASE_URL}/2025/2/S2B_43PGQ_20250206_0_L2A",
    },
}
BANDS = {
    "B04": {"window": Window(7922, 6570, 256, 256), "dtype": "uint16", "resolution": 10},
    "B08": {"window": Window(7922, 6570, 256, 256), "dtype": "uint16", "resolution": 10},
    "SCL": {"window": Window(3961, 3285, 128, 128), "dtype": "uint8", "resolution": 20},
}


def prepare() -> None:
    expected_crs = "EPSG:32643"
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES"):
        for role, sample in SAMPLES.items():
            directory = ROOT / role
            directory.mkdir(parents=True, exist_ok=True)
            for identity, specification in BANDS.items():
                url = f"{sample['prefix']}/{identity}.tif"
                output = directory / f"{identity}.tif"
                with rasterio.open(url) as source:
                    if source.count != 1 or source.crs is None or source.crs.to_string() != expected_crs:
                        raise RuntimeError(f"unexpected source raster contract: {url}")
                    if source.dtypes[0] != specification["dtype"] or source.nodata != 0:
                        raise RuntimeError(f"unexpected source dtype/nodata: {url}")
                    source_scale = float(source.scales[0])
                    source_offset = float(source.offsets[0])
                    if identity in {"B04", "B08"} and (source_scale, source_offset) not in {
                        (1.0, 0.0),
                        (0.0001, -0.1),
                    }:
                        raise RuntimeError(
                            f"unexpected source reflectance scale/offset ({source_scale}, {source_offset}): {url}"
                        )
                    prepared_scale, prepared_offset = (
                        (0.0001, -0.1) if identity in {"B04", "B08"} else (source_scale, source_offset)
                    )
                    resolution = abs(source.transform.a)
                    if resolution != specification["resolution"]:
                        raise RuntimeError(f"unexpected source resolution {resolution}: {url}")
                    window = specification["window"]
                    values = source.read(1, window=window)
                    if values.shape != (int(window.height), int(window.width)):
                        raise RuntimeError(f"short source window read: {url}: {values.shape}")
                    profile = source.profile.copy()
                    profile.update(
                        driver="GTiff",
                        width=values.shape[1],
                        height=values.shape[0],
                        transform=source.window_transform(window),
                        count=1,
                        compress="deflate",
                    )
                temporary = output.with_suffix(".tmp.tif")
                with rasterio.open(temporary, "w", **profile) as target:
                    target.write(values, 1)
                    target.scales = (prepared_scale,)
                    target.offsets = (prepared_offset,)
                    target.set_band_description(1, identity)
                    target.update_tags(
                        1,
                        BAND_ID=identity,
                        SOURCE_URL=url,
                        SOURCE_ITEM=sample["item"],
                        PREPARATION="Fixed native-grid window; no spectral-band resampling",
                        SCALE_OFFSET_SOURCE="Earth Search STAC raster:bands metadata",
                    )
                temporary.replace(output)
                print(f"prepared {output.relative_to(PROJECT_ROOT)} from {url}")


if __name__ == "__main__":
    prepare()
