# Sentinel-2 Level-2A vegetation comparison

## Scope

This workflow compares the Normalised Difference Vegetation Index (NDVI) over
the spatial intersection of valid pixels in two already prepared observations:

```text
reflectance = stored_value * scale + offset
NDVI = (B08_reflectance - B04_reflectance) /
       (B08_reflectance + B04_reflectance)
delta_NDVI = NDVI_after - NDVI_before
```

It does no catalog search, download, registration, or analysis-grid
resampling. It does not use Change-Agent or another model.

Ordinary RGB images are not inputs. RGB lacks B08 near-infrared and usually
lacks quantitative surface-reflectance values, CRS, affine transform, no-data,
and pixel-quality information.

## Product facts and choices

The initial target is Copernicus Sentinel-2 Level-2A surface reflectance.
Official references checked for this implementation:

- [SentiWiki S2 Products](https://sentiwiki.copernicus.eu/web/s2-products)
  documents the Level-2A conversion as
  `(DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE`, says DN 0 remains no-data,
  and explains that the band-specific offset was introduced at processing
  baseline 04.00. Scale and offset therefore remain explicit per band and date
  instead of being guessed from acquisition date.
- [SentiWiki S2 Processing](https://sentiwiki.copernicus.eu/web/s2-processing)
  documents the SCL labels 0–11 and says SCL is produced at 20 m. It also
  describes the cloud, shadow, snow, parallax, dilation, and post-processing
  logic.
- [Copernicus Data Space terms](https://dataspace.copernicus.eu/terms-and-conditions)
  state that access and use of Copernicus Sentinel data is free, full, and open
  under the linked Sentinel Data Legal Notice.
- [Earth Search documentation](https://github.com/Element84/earth-search)
  describes its free, no-guarantee STAC index of public AWS data, asset
  metadata, and HTTP access for non-requester-pays assets. A real sample, when
  prepared, must retain its exact STAC item and asset URLs in the configuration.

The supplied SCL rule accepts 4 (vegetation), 5 (not vegetated), and 6 (water).
It rejects 0 (no-data), 1 (saturated/defective), 2 (dark area or cast shadow,
depending on baseline), 3 (cloud shadow), 7 (unclassified/low-probability
cloud), 8–9 (cloud), 10 (cirrus), and 11 (snow/ice). This conservative mask
keeps non-vegetated and water pixels because the tool compares NDVI over the
whole prepared area; users may narrow the accepted subset to 4, 5, or 6 but
cannot configure known unsuitable SCL classes as valid.

SCL is useful but imperfect. Official documentation notes possible cloud
omissions, bright-target false detections, low-altitude cloud/fog omissions,
shadow errors, and cloud-edge artefacts. Passing the mask does not prove a
pixel is contamination-free.

## Input contract

Start from [`configs/vegetation.example.json`](../configs/vegetation.example.json).
For both `before` and `after`, it requires:

- an ISO acquisition date;
- a single-channel numeric B04 GeoTIFF and a single-channel numeric B08
  GeoTIFF;
- explicit `surface_reflectance` unit, `scale`, `offset`, and `nodata` for each
  band;
- an integer Sentinel-2 Level-2A SCL raster, a documented rule, accepted class
  list, and `nearest` resampling policy;
- source provenance such as provider, product/STAC item ID, asset URLs, and
  usage terms.

Paths are resolved relative to the configuration. B04 and B08 must have
identical dimensions, CRS, and affine transform within a date. Both dates must
also share that exact grid. The workflow never silently registers or resizes
spectral data.

SCL may differ from the analysis grid because the native mask is 20 m while
B04/B08 are 10 m. It must use the same CRS and fully cover the target grid; the
only supported operation is Rasterio geospatial reprojection with nearest
neighbour. The execution record captures source and target grids and whether
resampling occurred.

If reading an original SAFE product at processing baseline 04.00 or newer,
typical metadata may give `QUANTIFICATION_VALUE=10000` and
`BOA_ADD_OFFSET=-1000`, corresponding to `scale=0.0001`, `offset=-0.1`. Do not
assume those numbers: copy the actual product or STAC raster metadata. Some
derived COG providers may already expose a different stored scale/offset.

The repository also tracks
[`configs/vegetation.real-example.json`](../configs/vegetation.real-example.json),
which pins two nearly same-day-of-year Earth Search items over a 2.56 km square
area of Bengaluru. Prepare its small native-grid windows without credentials:

```sh
.venv/bin/python scripts/prepare_vegetation_sample.py
```

This fixed downloader is not a catalog search. The source rasters and generated
local chips remain ignored by Git.

## Command and outputs

Install the project dependencies in the existing local environment and run:

```sh
.venv/bin/python -m pip install -r requirements.in
.venv/bin/python scripts/run_vegetation_comparison.py \
  --config configs/vegetation.local.json
```

`--output-dir PATH` can override `output_root`. Every run creates a new
`run-YYYYMMDDTHHMMSSZ` directory (with a numeric collision suffix) rather than
overwriting earlier results. It contains:

- `before_ndvi.tif` and `after_ndvi.tif` (float32, nodata `-9999`);
- `delta_ndvi.tif` (after minus before, valid only on the common-valid mask);
- `common_valid_mask.tif` (`1` valid, `0` invalid/no-data);
- `comparison.png` (shared `[-1, 1]` NDVI scale, `[-1, 1]` zero-centred
  delta scale, invalid pixels in grey, and exact display-saturation counts in
  the JSON record);
- `run.json` (inputs, provenance, grid, scale/offset, mask rule, validity,
  statistics, runtime, outputs, warnings);
- `summary.txt` (deterministic interpretation summary).

Near-zero denominators are invalid, not NDVI zero. The code converts integers
to float64 before arithmetic, does not replace invalid data with zero, and does
not clip computed values. The PNG can saturate only its display colours at the
stated range endpoints; GeoTIFF values and statistics remain untouched.

Area is reported only for a projected CRS whose linear units Rasterio can
convert to metres. Pixel area is the absolute affine determinant, so rotated
projected grids are supported. Geographic grids report counts and an explicit
area-omission reason.

## Interpretation and limitations

Summary statistics use the intersection of valid pixels from both dates. With
threshold `t`, increase is `delta > t`, decrease is `delta < -t`, and values on
both boundaries are approximately stable. The example `t=0.1` is exploratory
and configurable; no universally correct threshold exists.

These are NDVI increase/decrease categories. NDVI alone does not establish
deforestation, crop stress, vegetation gain/loss, health improvement, damage,
or why a change occurred. Compare like seasons and consider phenology,
atmosphere, illumination, view geometry, soil moisture, land cover, residual
cloud/shadow, and other observation conditions. A two-date comparison cannot
separate short-lived variation from a sustained trend.

## Verification status

The lightweight tests use tiny synthetic numeric GeoTIFFs strictly as software
fixtures, not evidence of real-world accuracy. They cover known NDVI, explicit
scale/offset, integer-to-float conversion, near-zero denominators, no-data and
SCL exclusion, exact-grid rejection, common-valid statistics, threshold
boundaries, nearest-neighbour SCL placement, and GeoTIFF geospatial/nodata
metadata.

The fixed real example completed end to end using Earth Search items
`S2A_43PGQ_20240207_0_L2A` and `S2B_43PGQ_20250206_0_L2A`. The 65,536-pixel
grid had 64,428 common-valid pixels (98.31%, or 6,442,800 m² in EPSG:32643).
Over that exact intersection, before/after mean NDVI was 0.816823/0.918332,
median NDVI was 0.423209/0.470973, and mean/median delta was
0.101509/0.027036. At the explicitly exploratory ±0.1 threshold, 22,929 pixels
(35.59%) increased, 17,259 (26.79%) decreased, and 24,240 (37.62%) were
approximately stable.

Those real results need an important qualification: documented `-0.1`
reflectance offsets produce negative or near-zero reflectances for many dark
urban pixels, so many retained NDVI values fall outside `[-1, 1]`. The run
warned about this, retained all numerical values, and recorded colour-only
saturation for 23,870 before, 24,327 after, and 11,781 delta pixels. This is
preferable to silently clipping values, but it also makes the means and change
fractions sensitive to low-signal pixels; medians are less sensitive. Any
future reflectance-range or denominator policy must be explicit and justified,
not tuned to make the map look better.

The real run verifies execution and output contracts, not ecological accuracy.
No ground truth, field measurements, reference change labels, causal analysis,
or multi-date trend validation was available. A successful synthetic test or
real execution must not be presented as proof of real-world accuracy or a
specific vegetation event.
