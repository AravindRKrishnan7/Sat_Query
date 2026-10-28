# Project status

Status verified on 2026-10-28. The working implementation has two independent
paths: Change-Agent RGB inference and deterministic Sentinel-2 Level-2A NDVI
comparison. The NDVI path does not call an AI model.

## Local GeoLens interface

GeoLens is now a local Streamlit interface over the existing two execution
functions. It leaves the `satquery` package and repository name unchanged and
adds no LLM, agent framework, authentication, database, or deployment.

Construction mode:

- accepts PNG/JPEG uploads up to 10 MB under unique internal filenames;
- requires an explicit same-location/already-aligned confirmation;
- accepts the model's native 256 × 256 RGB contract, or optionally creates and
  previews explicit Lanczos-resized copies with a warning that resizing does
  not align imagery or guarantee suitable resolution;
- calls the existing Change-Agent CPU runner only on the Analyse button;
- caches the verified model load in the Streamlit process and retains the
  latest execution record in session state;
- displays the prepared pair, coloured class mask, overlay, raw unedited
  caption, predicted pixel fractions, warnings, downloads, and expandable
  technical metadata.

Vegetation mode has one “Run prepared example” action. It accepts no ordinary
RGB upload and calls the existing corrected real-example runner with a separate
`outputs/geolens/vegetation` root. It displays the existing comparison PNG,
dates, provenance, legends, common-valid coverage, threshold, and statistics
for pixels classified SCL 4 vegetation on both dates. Display stretching and
reflectance/NDVI calculations remain separate.

Every GeoLens execution uses a new directory beneath the ignored
`outputs/geolens/` tree. The app performs no asset download at startup and
reports missing checkpoint, upstream source, prepared raster, upload, and
result-contract errors in user-facing language.

### GeoLens verification

- Streamlit 1.66.1 is installed only in the project `.venv` and pinned in
  `requirements.in`.
- All 24 lightweight tests pass. Six new tests cover bounded PNG/JPEG decoding,
  explicit resizing, RGB rejection, unique run/input paths, and execution-record
  validation.
- Streamlit's integration harness rendered both modes without exceptions and
  ran both workflows against real local assets, not mocks.
- Construction used the current 1254 × 1254 repository sample through the
  explicit resize path and the real verified CPU checkpoint. Its execution at
  `outputs/geolens/construction/run-20261028T125540054743Z-54d42679ec`
  predicted 95.93% background, 0.00% road change, and 4.07% building change.
  The raw caption was `some houses are built on the bareland on the bareland`;
  the repetition is retained.
- Vegetation execution
  `outputs/geolens/vegetation/run-20261028T125434Z` reproduced 98.3521% common
  valid coverage, 7,504 persistent-vegetation pixels, the corrected NDVI
  statistics, and threshold ±0.1.
- The local server started successfully at `http://127.0.0.1:8501`. No browser
  instance was connected to the available browser-control surface, so visible
  browser interaction was not verified; server startup and Streamlit integration
  testing are verified separately.

Remaining interface limitations: spatial alignment is caller-confirmed rather
than measured; construction accuracy remains unmeasured; construction resizing
is generic image resampling, not geospatial preparation; the caption is
unverified; vegetation retains all limitations below; results persist only on
local disk/session state; and there is intentionally no multi-user isolation,
authentication, database, background job queue, or public deployment.

## Vegetation scaling audit

A real scaling bug was confirmed and fixed. The first real run applied
`stored * 0.0001 - 0.1` to Earth Search COG values whose BOA offset had already
been applied during COG conversion. This was a second application of the
Sentinel-2 baseline offset.

Evidence for both pinned acquisitions:

- the exact STAC items report `earthsearch:boa_offset_applied: true`;
- Earth Search's collection-level `item_assets` metadata reports reflectance
  `scale: 0.0001`, `offset: 0`;
- the item-level assets contradict that state by reporting `offset: -0.1`;
- each remote COG is a single-band `uint16` GeoTIFF with nodata 0 and neutral
  internal scale/offset 1/0;
- the downloaded windows are byte-for-byte equal as numeric arrays to the
  corresponding remote COG windows;
- Element 84 issue [#71](https://github.com/Element84/earth-search/issues/71)
  documents that the legacy collection can advertise `-0.1` after the COG
  pixels were already offset-adjusted. Issue
  [#24](https://github.com/Element84/earth-search/issues/24) identifies
  `earthsearch:boa_offset_applied` as the intended representation flag.

The corrected conversion is independently recorded for both acquisitions as:

```text
surface_reflectance = stored_uint16 * 0.0001 + 0.0
```

The preparation script now writes that audited scale/offset into each local
B04/B08 GeoTIFF and rejects a change to the pinned remote COG header. The run
configuration records the provider flag, representation, and why the
contradictory item-level `-0.1` is ignored.

Prior results remain untouched under `outputs/vegetation-real/`, including the
last affected run `run-20261027T165248Z`. Its out-of-range NDVI values arose
from double application of the offset, not valid negative reflectance in these
prepared COG windows.

### Per-acquisition numerical audit

Stored-value percentiles (`uint16`, excluding nodata) are:

| Date | Band | Min | P1 | P5 | P25 | P50 | P75 | P95 | P99 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-02-07 | B04 | 107 | 232 | 360 | 724 | 1155 | 1532 | 2122 | 2712 | 5148 |
| 2024-02-07 | B08 | 162 | 244.35 | 1122 | 1548 | 1821 | 2134 | 2740 | 3325.30 | 5341 |
| 2025-02-06 | B04 | 125 | 260 | 340 | 706 | 1148 | 1520 | 2132 | 2770 | 5036 |
| 2025-02-06 | B08 | 176 | 265 | 1185 | 1577 | 1853 | 2170 | 2785 | 3408 | 5767 |

After applying `0.0001/0.0`, B04/B08 reflectance ranges are
0.0107–0.5148/0.0162–0.5341 on 2024-02-07 and
0.0125–0.5036/0.0176–0.5767 on 2025-02-06. Each band/date has 0 negative and
0 zero reflectances. Each date has 0 near-zero denominators and 0 NDVI values
outside `[-1, 1]`.

The nearest-neighbour SCL distribution is:

| SCL | Meaning | 2024-02-07 | 2025-02-06 |
| ---: | --- | ---: | ---: |
| 2 | cast shadow/dark area | 876 | 872 |
| 4 | vegetation | 8,448 | 10,536 |
| 5 | not vegetated | 55,396 | 53,296 |
| 6 | water | 800 | 816 |
| 7 | unclassified | 16 | 16 |

Within every listed SCL class on both dates, negative/zero reflectance,
near-zero denominator, and out-of-range NDVI counts are all zero. Classes 2
and 7 are numerically valid but quality-mask-invalid; classes 5 and 6 are
quality-mask-valid but excluded from the vegetation-only interpretation.

## Corrected real-data verification

Corrected run: `outputs/vegetation-real/run-20261027T171413Z`

The broad quality-valid population uses SCL 4 (vegetation), 5 (not vegetated),
and 6 (water). It has 64,456 common pixels (98.3521%). Mean/median NDVI changed
from 0.244767/0.167017 to 0.259284/0.177172; mean/median delta is
0.014516/0.014226. At the exploratory ±0.1 threshold, 6,113 pixels (9.48%)
increased, 3,299 (5.12%) decreased, and 55,044 (85.40%) were approximately
stable.

The vegetation-only interpretation requires SCL 4 on both dates. It has 7,504
common pixels (11.4502%, 750,400 m²). Mean/median NDVI changed from
0.632204/0.639994 to 0.654030/0.670941; mean/median delta is
0.021826/0.024096. At ±0.1, 1,013 pixels (13.50%) increased, 485 (6.46%)
decreased, and 6,006 (80.04%) were approximately stable.

For both dates, all 65,536 B04/B08 pairs are non-nodata and numerically valid.
There are zero negative reflectances, zero zero-reflectance values, zero
near-zero denominators at epsilon `1e-6`, and zero NDVI values outside
`[-1, 1]`, in every observed SCL class. The execution record contains stored
and reflectance percentiles plus per-class exclusion counts.

## Validity populations

The JSON and summary distinguish three populations:

1. `quality_mask_valid`: the SCL class is accepted, independent of numerical
   validity.
2. `numerically_valid_ndvi`: B04/B08 are present and finite and the absolute
   denominator exceeds the configured epsilon, independent of SCL.
3. `interpretation_suitable`: the pixel passes both prior tests and is in a
   configured interpretation SCL class. The real example uses only SCL 4.

Raw numerically valid before/after NDVI rasters are saved without SCL masking.
The original quality-masked outputs remain, and a separate common
interpretation mask is added. Nothing is clipped.

## Improved comparison visualisation

Newest verified run: `outputs/vegetation-real/run-20261027T173607Z`

The fixed preparation now also reads B02 and B03 from the same two pinned STAC
items, exact 2.56 km extent, and 10 m analysis grid as B04/B08. For each date,
the script verified the exact STAC asset URL, `uint16`/nodata-0 raster contract,
neutral remote COG header scale/offset 1/0, and the provider-specific audited
reflectance conversion `stored * 0.0001 + 0.0`. All 65,536 RGB pixels are
band-valid on each date.

The new PNG uses B04/B03/B02 for true colour in its top row and retains before
NDVI, after NDVI, and after-minus-before NDVI in its bottom row. Both true-colour
panels use the same display-only reflectance range 0.00–0.30 and gamma 2.2.
The execution record reports 536 before and 733 after channel values above that
display range; this clipping affects PNG colours only. It does not affect the
analysis arrays, masks, GeoTIFFs, or statistics.

The before, after, and delta NDVI GeoTIFF SHA-256 hashes exactly match the
scaling-corrected baseline run. All broad and vegetation-only statistics are
also unchanged. The figure labels the latter as applying only to the 7,504
pixels classified as SCL 4 vegetation on both dates and explicitly avoids
claiming vegetation gain/loss or proven health change. Prior output directories
remain preserved.

## Verification and remaining limitations

- 18 lightweight tests pass, including all existing Change-Agent tests and a
  regression check that rejects an RGB band on a different grid.
- Regression coverage now reproduces the double-offset failure and rejects a
  configured offset that conflicts with explicit raster metadata.
- The improved two-row comparison image was visually inspected; it contains no
  generated or unrelated imagery.
- True colour is a contextual rendering, not calibrated colourimetry. Its fixed
  stretch and gamma are documented display choices.
- No ground truth, field measurements, reference change labels, causal
  analysis, or multi-date trend validation is available.
- SCL is an observation-quality and coarse land-cover classification, not
  ground truth. Requiring SCL 4 on both dates omits transitions into or out of
  that class and can introduce selection bias.
- The ±0.1 delta threshold remains exploratory.
