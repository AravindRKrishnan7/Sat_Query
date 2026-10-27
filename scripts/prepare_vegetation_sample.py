#!/usr/bin/env python3
"""Download fixed, small Earth Search COG windows for the documented real example.

This is deliberately not a live catalog search. It reads one 2.56 km square
window from six pinned public assets and writes prepared local GeoTIFFs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402


ROOT = PROJECT_ROOT / "data" / "vegetation" / "earth-search-bengaluru"
BASE_URL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/P/GQ"
STAC_COLLECTION_URL = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"
SAMPLES = {
    "before": {
        "item": "S2A_43PGQ_20240207_0_L2A",
        "prefix": f"{BASE_URL}/2024/2/S2A_43PGQ_20240207_0_L2A",
        "reflectance_scale": 0.0001,
        "reflectance_offset": 0.0,
        "boa_offset_applied": True,
    },
    "after": {
        "item": "S2B_43PGQ_20250206_0_L2A",
        "prefix": f"{BASE_URL}/2025/2/S2B_43PGQ_20250206_0_L2A",
        "reflectance_scale": 0.0001,
        "reflectance_offset": 0.0,
        "boa_offset_applied": True,
    },
}
BANDS = {
    "B04": {"window": Window(7922, 6570, 256, 256), "dtype": "uint16", "resolution": 10},
    "B08": {"window": Window(7922, 6570, 256, 256), "dtype": "uint16", "resolution": 10},
    "SCL": {"window": Window(3961, 3285, 128, 128), "dtype": "uint8", "resolution": 20},
}


def read_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - URLs are pinned constants.
        return json.load(response)


def prepare() -> None:
    expected_crs = "EPSG:32643"
    collection = read_json(STAC_COLLECTION_URL)
    for asset_key in ("red", "nir"):
        band = collection["item_assets"][asset_key]["raster:bands"][0]
        if (float(band["scale"]), float(band["offset"])) != (0.0001, 0.0):
            raise RuntimeError(f"Earth Search collection scaling changed for {asset_key}: {band}")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES"):
        for role, sample in SAMPLES.items():
            item_url = f"{STAC_COLLECTION_URL}/items/{sample['item']}"
            item = read_json(item_url)
            if item["properties"].get("earthsearch:boa_offset_applied") is not sample["boa_offset_applied"]:
                raise RuntimeError(f"Earth Search BOA-offset representation flag changed: {item_url}")
            directory = ROOT / role
            directory.mkdir(parents=True, exist_ok=True)
            for identity, specification in BANDS.items():
                url = f"{sample['prefix']}/{identity}.tif"
                asset_key = {"B04": "red", "B08": "nir", "SCL": "scl"}[identity]
                if item["assets"][asset_key]["href"] != url:
                    raise RuntimeError(f"pinned STAC asset href changed for {identity}: {item_url}")
                output = directory / f"{identity}.tif"
                with rasterio.open(url) as source:
                    if source.count != 1 or source.crs is None or source.crs.to_string() != expected_crs:
                        raise RuntimeError(f"unexpected source raster contract: {url}")
                    if source.dtypes[0] != specification["dtype"] or source.nodata != 0:
                        raise RuntimeError(f"unexpected source dtype/nodata: {url}")
                    source_scale = float(source.scales[0])
                    source_offset = float(source.offsets[0])
                    if identity in {"B04", "B08"} and (source_scale, source_offset) != (1.0, 0.0):
                        raise RuntimeError(
                            f"pinned source COG header changed from audited neutral scale/offset (1.0, 0.0) "
                            f"to ({source_scale}, {source_offset}): {url}"
                        )
                    prepared_scale, prepared_offset = (
                        (sample["reflectance_scale"], sample["reflectance_offset"])
                        if identity in {"B04", "B08"}
                        else (source_scale, source_offset)
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
                        BOA_OFFSET_APPLIED=str(sample["boa_offset_applied"]).lower(),
                        SCALE_OFFSET_SOURCE=(
                            "Earth Search item earthsearch:boa_offset_applied=true and collection-level "
                            "raster:bands scale=0.0001 offset=0; item-level offset=-0.1 is inconsistent and ignored"
                        ),
                    )
                temporary.replace(output)
                print(f"prepared {output.relative_to(PROJECT_ROOT)} from {url}")


if __name__ == "__main__":
    prepare()
