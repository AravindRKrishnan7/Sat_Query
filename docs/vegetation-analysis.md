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

Ordinary RGB images are not analysis inputs. RGB lacks B08 near-infrared and
usually lacks quantitative surface-reflectance values, CRS, affine transform,
no-data, and pixel-quality information. Optional native B02/B03 bands can be
supplied with the already required B04 band to produce a contextual true-colour
display; they do not enter the NDVI calculation.

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

Provider representation must take precedence over a generic product formula.
For the legacy Earth Search `sentinel-2-l2a` collection, item-level raster
offset metadata can contradict both the collection metadata and the actual COG
representation. Element 84 issue
[#71](https://github.com/Element84/earth-search/issues/71) records cases where
the item advertises `-0.1` even though conversion already applied the offset.
The pinned real-example items both report
`earthsearch:boa_offset_applied=true`; therefore their COG values use
`reflectance = stored * 0.0001`, not `stored * 0.0001 - 0.1`.

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
- optional single-channel B03 and B02 GeoTIFFs under `true_color`; if present,
  they must be supplied for both dates, while B04 is reused as red;
- explicit `surface_reflectance` unit, `scale`, `offset`, `nodata`, and a
  textual `scale_offset_source` for each band;
- an integer Sentinel-2 Level-2A SCL raster, a documented rule, accepted class
  list, and `nearest` resampling policy;
- source provenance such as provider, product/STAC item ID, asset URLs, and
  usage terms.

Paths are resolved relative to the configuration. B02, B03, B04, and B08 must
have identical dimensions, CRS, and affine transform within a date when the
true-colour inputs are configured. Both dates must also share that exact grid.
Each band has its own explicit scale, offset, nodata, and metadata source. The
workflow never silently registers or resizes spectral data.

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
local chips remain ignored by Git. For each pinned acquisition, it checks the
exact STAC URLs and verifies that B02/B03/B04/B08 remain `uint16`, nodata 0,
and internally neutral scale/offset 1/0, then annotates each local copy with the
audited `0.0001/0.0` conversion. The four spectral bands use the same native
10 m window and analysis grid.

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
- `common_interpretation_mask.tif` (`1` suitable for the configured
  interpretation on both dates);
- `before_ndvi_raw.tif` and `after_ndvi_raw.tif` (numerically valid NDVI before
  SCL quality filtering, retained for audit rather than summary statistics);
- `comparison.png` (top-row true-colour before/after; bottom-row before NDVI,
  after NDVI, and NDVI difference; dates, legend, and a vegetation-only summary
  for pixels classified SCL 4 on both dates);
- `run.json` (inputs, provenance, grid, scale/offset, mask rule, validity,
  statistics, runtime, outputs, warnings);
- `summary.txt` (deterministic interpretation summary).

Both true-colour panels use B04/B03/B02 reflectance with the identical fixed
display range 0.00–0.30 and gamma 2.2. The execution record stores those
settings and counts channel values outside the display range. Those values are
clipped only while making the PNG. RGB adjustments are isolated from the
analysis arrays and never alter numerical outputs.

Near-zero denominators are invalid, not NDVI zero. The code converts integers
to float64 before arithmetic, does not replace invalid data with zero, and does
not clip computed values. The PNG can saturate only its display colours at the
stated range endpoints; GeoTIFF values and statistics remain untouched.

The JSON separately reports SCL quality validity, numerical NDVI validity, and
interpretation suitability. Its per-date diagnostics include stored-value and
reflectance percentiles; negative/zero reflectance, near-zero denominator, and
out-of-range NDVI counts; and the same exclusion counts by SCL class.

Area is reported only for a projected CRS whose linear units Rasterio can
convert to metres. Pixel area is the absolute affine determinant, so rotated
projected grids are supported. Geographic grids report counts and an explicit
area-omission reason.

## Interpretation and limitations

Broad statistics use the intersection of numerically valid, quality-mask-valid
pixels from both dates. The real example's broad population includes SCL 4
(vegetation), 5 (not vegetated), and 6 (water). Vegetation-only interpretation
statistics are reported separately and require SCL 4 on both dates. This rule
is explicit but omits transitions into or out of the SCL vegetation class and
can introduce selection bias. With threshold `t`, increase is `delta > t`,
decrease is `delta < -t`, and values on both boundaries are approximately
stable. The example `t=0.1` is exploratory and configurable; no universally
correct threshold exists.

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

The first real run contained a confirmed double-offset bug. Both pinned items
have `earthsearch:boa_offset_applied=true`, but the preparation/configuration
applied the contradictory item-level `-0.1` again. Prior output directories are
preserved for comparison; `run-20261027T165248Z` is the last affected run.

The scaling-corrected baseline run is `run-20261027T171413Z`. The improved
comparison run is `run-20261027T173607Z`; its three NDVI GeoTIFFs are
byte-for-byte identical to the corrected baseline. All 65,536 pixels per date
have valid B04/B08 data. For both dates there are zero negative reflectances,
zero zero-reflectance values, zero near-zero denominators at `1e-6`, and zero
NDVI values outside `[-1, 1]` across every observed SCL class.

The broad SCL 4/5/6 intersection has 64,456 pixels (98.35%). Before/after
mean NDVI is 0.244767/0.259284, median is 0.167017/0.177172, and mean/median
delta is 0.014516/0.014226. At ±0.1, 6,113 pixels (9.48%) increased, 3,299
(5.12%) decreased, and 55,044 (85.40%) were approximately stable.

The vegetation-only SCL 4 intersection has 7,504 pixels (11.45%, 750,400 m²).
Before/after mean NDVI is 0.632204/0.654030, median is 0.639994/0.670941, and
mean/median delta is 0.021826/0.024096. At ±0.1, 1,013 pixels (13.50%)
increased, 485 (6.46%) decreased, and 6,006 (80.04%) were approximately stable.

Full stored-value percentiles, reflectance ranges, and SCL-class distributions
are in the corrected `run.json` and summarised in
[`PROJECT_STATUS.md`](PROJECT_STATUS.md).

The improved run uses B02/B03/B04 from the same two STAC items, native window,
extent, CRS, transform, and 10 m grid as the NDVI analysis. All 65,536 RGB
pixels are band-valid on both dates. Visual inspection confirmed the intended
two-row layout and portable labels. The 0.00–0.30 display range clips 536
before and 733 after channel values (out of 196,608 per date); those display
counts are recorded and do not affect the analysis.

The real run verifies execution and output contracts, not ecological accuracy.
No ground truth, field measurements, reference change labels, causal analysis,
or multi-date trend validation was available. A successful synthetic test or
real execution must not be presented as proof of real-world accuracy or a
specific vegetation event.
