# Project status

Status verified on 2026-10-28. The working implementation has two independent
analysis paths: Change-Agent RGB inference and deterministic Sentinel-2
Level-2A NDVI processing. GeoLens exposes construction, corrected two-date
vegetation, and bounded historical vegetation workflows. Neither NDVI workflow
uses an AI model for its calculations. A separate local language-model layer
now interprets requests and selects validated result statements without
changing either analysis path.

## Local language-model interface

GeoLens now connects to the existing loopback Ollama service. The environment
contained Ollama 0.32.5 and these local models: `qwen3:8b`, `qwen3.5:4b`,
`mistral:latest`, `qwen2.5:7b`, `llama3.2:3b`, and the
`nomic-embed-text` embedding model. No external API provider variable or Python
client was configured, and no paid service or model download was used.

The tested runtime is `ollama/qwen3:8b` (8.2B, Q4_K_M). It advertises
completion, tool and thinking capabilities. Ollama's local `/api/chat` JSON
schema output was tested directly. `qwen3.5:4b` obeyed the first schema but
duplicated vegetation intent and invented clarification fields, so it was not
accepted as the default.

The initial `qwen3:8b` missing-date test also invented a year range. GeoLens
now requires exact source-text evidence for every asserted analysis and map
reference, and accepts a year only if its exact token occurs in the original
question. The strengthened four-case real compatibility run passed:

- vegetation here from 2021 to 2025 → vegetation only, correct years and
  boundary reference;
- construction and roads → both intents preserved, not substituted with
  vegetation;
- vegetation and construction → both intents preserved for a partial-support
  plan;
- vegetation with no dates → start and end years reported as missing rather
  than filled by the model.

The model cannot receive or generate an AOI coordinate field. The application
still requires an explicitly confirmed map boundary and applies the existing
year, request-size, seasonal, quality, cache and sensor validators. Unsupported
construction/road intent is retained in the plan with the exact requirement
for aligned 256 × 256 RGB inputs in Construction changes mode. The model can
invoke no Python, shell command or arbitrary tool: only the predefined
historical vegetation wrapper can reach `run_historical`, and only after the
user reviews the plan and clicks Run.

Final-answer generation receives a compact copy of actual dates, missing
years, coverage, metrics, pixel populations, limitations and output
references. The model returns only schema-constrained statement codes and all
required fact IDs. Dates and numerical claims are rendered from the tool
record, not model prose. Missing/invented fact IDs or an inconsistent
conclusion triggers a visibly labelled deterministic fallback.

### Real end-to-end language-model verification

Request: `Compare vegetation here from 2021 to 2025.`

Assistant audit record:
`outputs/geolens/assistant/assistant-20261028T163108448718Z-2c6a6a9a8e/assistant.json`

The real model interpreted vegetation only, years 2021–2025, and `here` as the
confirmed boundary. The validated historical tool ran from the explicit
Streamlit Run action using the matching cache. Its result remained honestly
partial: 2022 was missing, while 2021, 2023, 2024 and 2025 were available. The
model returned all six required fact IDs with
`partial_historical_ndvi_comparison` and
`missing_years_limit_comparison`; answer validation passed without fallback.
The displayed answer used the actual dates, 76.83% all-observation
common-valid coverage, the fixed-population annual NDVI means, 84.23%
first/last pair coverage and the documented population distinction.

The audit record contains the original question, provider/model identifier,
exact evidence, validated interpretation, selected tool and arguments,
execution status, result references and final answer metadata. Question, draft
AOI or boundary confirmation changes clear stale plans and answers. Ordinary
Streamlit reruns do not execute the tool again.

### Language-interface verification and limitations

- 42 lightweight tests pass. Nine language-interface tests cover unavailable
  service, missing/invented dates, unconfirmed AOI, unsupported construction
  and roads, combined partial planning/execution, tool failure, missing
  numerical fact references and accepted deterministic fact rendering.
- The real four-case `scripts/check_llm.py` compatibility test passed with
  `qwen3:8b`.
- Streamlit's integration harness completed the real natural-language request,
  plan review, explicit tool execution, maps/charts, non-fallback answer and 14
  downloads without exceptions.
- The integration is deliberately local-Ollama-only. It does not implement a
  cloud-provider adapter or conversational slot filling; an incomplete request
  must be resubmitted with the displayed missing fields.
- Language-model interpretation remains fallible. Evidence and application
  validators limit its authority, and deterministic fallback remains the
  authoritative failure path.

## Historical GeoLens workflow

GeoLens now has a map-based historical vegetation mode while preserving the
construction and corrected two-date modes. It provides:

- an Alappuzha-centred interactive map with rectangle drawing, explicit
  confirmation, displayed WGS84 bounds and approximate area;
- a configurable 25 km² limit and invalid/oversized selection rejection;
- invalidation of confirmation, request and result when the draft AOI changes,
  plus request/result invalidation when year or analysis controls change;
- a typed request containing the confirmed geometry, years, requested
  analysis, fixed January–March seasonal policy, candidate limit, SCL classes,
  yearly/common coverage requirements, denominator epsilon and NDVI threshold;
- two to five annual observations (2021–2025 by default), with the adjacent
  comparison count shown before execution;
- a real local-Ollama chat mode plus manual structured controls. The model
  produces an evidence-backed typed plan but cannot execute before validation
  and an explicit Run action;
- honest rejection of historical construction/road requests. The MCI model is
  never applied to Sentinel-2 timeline data;
- exact-AOI search and raster caches, bounded metadata retries, per-year
  partial results and unique result directories.

The provider was reverified as Element 84 Earth Search collection
`sentinel-2-c1-l2a`. Each year uses the same January 1–March 31 window. Up to
five candidates are ranked by ascending catalogue `eo:cloud_cover`, after
which SCL 4/5/6 coverage is evaluated inside the actual AOI. The first
candidate at or above 80% is accepted; the date window is never silently
changed. Only small blue, green, red, NIR and SCL source windows are read.

All acquisitions are prepared on one 10 m EPSG:32643 grid. Reflectance uses
bilinear and SCL uses nearest-neighbour resampling. For every Collection-1
reflectance asset, GeoLens requires the item `raster:bands` scale/offset to
match the remote COG header before applying it. The real run independently
verified `scale=0.0001, offset=-0.1` for all four reflectance assets on every
accepted date. This preserves the provider-specific legacy correction rather
than assuming the legacy and Collection-1 representations are identical.

### Real Alappuzha verification

Confirmed demo bounds: `[76.34, 9.48, 76.36, 9.50]` (approximately
4.878 km²). The 222 × 223 grid contains 48,571 AOI pixel centres. The
true-colour previews were inspected and show correctly located mixed built-up
land, roads, fields and a small water edge; the area is not an empty or
silently substituted cached tile.

First network execution:
`outputs/geolens/historical/run-20261028T154842968270Z-e767826459`

| Requested year | Selected/evaluated item and date | Accepted AOI coverage | Outcome |
| ---: | --- | ---: | --- |
| 2021 | `S2B_T43PFL_20210207T052558_L2A`, 2021-02-07 | 99.7241% | available |
| 2022 | `S2A_T43PFL_20220319T052227_L2A`, 2022-03-19 | 0.0247% | unavailable |
| 2023 | `S2B_T43PFL_20230108T052433_L2A`, 2023-01-08 | 90.4346% | available |
| 2024 | `S2A_T43PFL_20240118T052357_L2A`, 2024-01-18 | 97.7765% | available |
| 2025 | `S2B_T43PFL_20250117T052047_L2A`, 2025-01-17 | 84.2807% | available |

The only 2022 January–March catalogue candidate had 58.1173% scene cloud and
failed the 80% in-AOI rule, so the missing year remains explicit. The
2021→2022 and 2022→2023 adjacent comparisons are unavailable and no gap is
bridged. The 2023→2024 and 2024→2025 comparisons are available, as is the
2021→2025 endpoint comparison.

The all-observation common-valid population across the four available dates is
37,316 pixels, or 76.8277% of the AOI, above the configured 50% minimum. Mean
NDVI on that fixed population is 0.466366 (2021), 0.487538 (2023), 0.513673
(2024) and 0.482375 (2025). The first/last common-population mean difference is
+0.016009. The separately reported 2021→2025 pair-valid map covers 84.2272%
and has mean difference +0.018086; these values use different documented pixel
populations and are not interchangeable. All accepted observations had zero
near-zero denominators and zero un-clipped NDVI values outside `[-1, 1]`.

The first execution retrieved 20 selected reflectance/SCL AOI windows in
64.3 seconds. A second execution at
`outputs/geolens/historical/run-20261028T155002544406Z-3472d5bdec` loaded all
20 selected windows from the matching exact-AOI cache in 0.20 seconds. The
cached run reproduced the available/missing years, coverage and results.

Annual true-colour and NDVI maps, available adjacent maps, the first-to-last
map, common-valid mask, summary and full result JSON are retained. Display
stretching remains separate from reflectance and NDVI. All descriptions use
“NDVI increase/decrease” and explicitly avoid claims of statistically
significant change, vegetation extent, plant health, deforestation, damage or
cause.

### Historical verification and limitations

- 42 lightweight tests pass: the original construction and two-date
  vegetation regressions, nine historical tests, and nine language-interface
  tests. Historical coverage includes unconfirmed and oversized AOIs,
  stale-state invalidation, invalid dates/request size, unsupported
  construction, scaling representations, cache failure, missing-year
  adjacency, and insufficient common coverage.
- Streamlit's integration harness performed the real cached historical flow
  from demo-boundary loading through explicit confirmation, validation,
  execution and result rendering without exceptions.
- The local server started at `http://127.0.0.1:8501`. No browser session was
  connected to the available browser-control surface, so live visible-browser
  interaction is not claimed.
- Place search was optional and is not implemented; manual map selection and
  the documented demo rectangle work independently.
- The January–March policy is a configurable demonstration choice, not a
  universally optimal season. Annual single acquisitions omit within-season
  variability; SCL is imperfect; NDVI alone cannot establish land-cover
  transition or cause; and no ground truth was used.

## Local GeoLens interface

GeoLens is now a local Streamlit interface over the construction, corrected
two-date vegetation, and historical vegetation execution functions. It leaves
the `satquery` package and repository name unchanged. It adds a constrained
local Ollama interface but no agent framework, authentication, database,
cloud-model dependency, or deployment.

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

- Streamlit 1.66.1, Folium 0.20.0 and streamlit-folium 0.27.4 are installed
  only in the project `.venv` and pinned in `requirements.in`.
- All 42 lightweight tests pass. The six local-interface tests cover bounded
  PNG/JPEG decoding, explicit resizing, RGB rejection, unique run/input paths,
  and execution-record validation; nine historical and nine language tests
  cover the contracts listed above.
- Streamlit's integration harness rendered all three modes without exceptions
  and ran the construction, two-date vegetation and cached historical
  workflows against real local assets, not mocks.
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
