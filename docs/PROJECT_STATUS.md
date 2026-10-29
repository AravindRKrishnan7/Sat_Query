# Project status

Status verified on 2026-10-29. The working implementation has two independent
analysis paths: Change-Agent RGB inference and deterministic Sentinel-2
Level-2A NDVI processing. GeoLens now exposes exactly two user-facing
workflows: **Buildings and infrastructure** and **Vegetation history**. The
corrected prepared two-date vegetation runner remains implemented and tested,
but is no longer a separate main-selector choice. Neither NDVI workflow uses an
AI model for its calculations. A separate local language-model layer interprets
requests and selects validated result statements without changing either
analysis path.

## Result exploration, grounded follow-ups and PDF export

The main GeoLens views now use a scoped navy/teal presentation layer, clearer
step labels and compact result selectors. The two workflow choices, existing
inputs, explicit Run actions, calculations, model predictions and execution
records are unchanged.

Infrastructure timelines show one selected comparison at a time. A native
slider composes a display-only vertical reveal from the exact prepared before
and after files after their paths and SHA-256 values are checked against the
parent and child records. A side-by-side fallback and the saved prediction
overlay are separate toggles. Historical results similarly show one dated
true-colour/NDVI pair and one NDVI difference interval while retaining all
missing-year warnings and evidence downloads.

Both result views now provide follow-up conversation tied to the active
analysis ID. Local `ollama/qwen3:8b` selects a constrained intent; the
application renders the answer from saved evidence. It does not give the model
an execution tool, accept arbitrary prose as a finding, or rerun an analysis.
Different dates require a new reviewed analysis. Counts, causes, vegetation
cover, tree counts and ambiguous intervals remain limited or require
clarification. An unavailable or rejected classifier response uses the same
readable deterministic renderer and records the fallback reason.

Illustrated A4 PDF export is available behind an explicit button. The exporter
imports neither analysis engine and consumes only saved records and visual
artifacts. Infrastructure export rechecks prepared hashes and child input
identity; it preserves user-supplied-date warnings, failed intervals, captions,
class populations and model limitations. Historical export includes actual and
missing observations, true-colour/NDVI panels, difference maps, exact pairwise
populations, scaling source, grid/cache provenance and NDVI limitations. The
four-page infrastructure and six-page vegetation example reports were rendered
through Poppler and every page was visually inspected with no clipping,
overlap or unreadable text.

Verification on 2026-10-29:

- all 101 regression tests pass;
- the real `ollama/qwen3:8b` compatibility command passes all 13 cases;
- real-model follow-ups correctly selected `road_changes` for the saved
  synthetic timeline and `missing_observation` for the cached Alappuzha result,
  with no fallback and `execution_performed: false` in both records;
- Streamlit's integration harness restored the exact three-image synthetic
  timeline from original byte streams and dates, exercised the 35% comparison
  divider, both display toggles, follow-up persistence and PDF export, then
  restored the cached Alappuzha result and exercised missing-year follow-up and
  PDF export. Neither flow created an analysis directory or modified its saved
  execution record;
- the correct checkout's port-8501 process predated the visual changes and was
  restarted successfully at `http://127.0.0.1:8501`;
- no browser instance was connected to the browser-control surface. Visible
  browser interaction is therefore not claimed; server startup and Streamlit
  integration testing are reported separately.

## Two-workflow GeoLens delivery

### Buildings and infrastructure

The uploaded-image workflow now accepts two to four images, requires an ISO
acquisition date for every observation, rejects missing or duplicate dates,
sorts the previews chronologically, and retains the explicit same-extent and
alignment confirmation. Supplying fewer than two or more than four images
cannot start inference. Optional Lanczos preparation remains explicit and is
described only as resizing—not registration or restored spatial detail.

GeoLens runs the unchanged two-image MCI model independently for each
consecutive pair. An optional first-to-last comparison for timelines longer
than two observations is labelled as a separate endpoint run, never a sum of
the intermediate predictions. The verified model is cached and reused across
intervals. Parent execution is incremental: a failed interval records its
error while earlier completed comparisons remain available, and independent
later comparisons continue.

Local `qwen3:8b` extracts a constrained question contract for building change,
road change, general image change, instance-count requests and requested
spatial descriptions. Exact source-text evidence is required for every true
field. The application—not the language model—then renders each interval from
that validated intent, its recorded change categories and its own conservative
caption assessment. Only tightly matched building/road addition, removal or
no-change phrases may be paraphrased. Quantities, locations, dates, causes and
instruction-like caption text are discarded. Category and caption agreement
never establishes spatial consistency. The chronological observations and
overlays remain visible; raw captions, category counts, hashes, timings,
assessment, disagreement state and JSON records are expandable.

The real Streamlit integration-harness run used three aligned observations: the
pinned RSICC start and end samples plus a deliberately synthetic 50% pixel
blend as the midpoint. The dates 2000-01-01 through 2000-01-03 are explicitly
artificial demonstration labels, not image acquisition dates. Local
`qwen3:8b` interpreted the question and the verified CPU checkpoint completed
two consecutive comparisons plus the independent endpoint comparison. The
model loaded once and was reused. The first interval and endpoint descriptions
suggested houses were added; the second interval reported no highlighted
building or road changes and an unclear caption. These are model predictions
on a synthetic sequence, not historical findings.

The parent record preserves original upload filename/bytes/SHA-256, prepared
path/SHA-256 and settings for every observation, plus exact interval inputs,
dates, answers, overlays, errors and child records. The request fingerprint
includes every image byte stream, date, question, resizing choice and endpoint
option, so a change hides the retained timeline without deleting it.

- Timeline record:
  `outputs/geolens/construction-timelines/timeline-20261029T044436573706Z-172e36b1a6/timeline.json`
- Timeline answer:
  `outputs/geolens/construction-timelines/timeline-20261029T044436573706Z-172e36b1a6/timeline-answer.md`

#### No-detection audit for the latest uploaded synthetic pair

The later user run
`outputs/geolens/construction-timelines/timeline-20261029T045617162787Z-b075502bbc/timeline.json`
completed successfully but predicted no building change. Chronological order
followed the unverified user dates exactly: `Suburban Neighbourhood
Expansion.png` was assigned 2022-02-02 and prepared as SHA-256 `4cafdb14…186a`;
`Suburban Homes Beside Open Fields.png` was assigned 2023-02-02 and prepared as
`00bf319a…02e2`. Both 1254 × 1254 uploads were explicitly Lanczos-resized to
256 × 256. The first prepared image visibly contains three houses along the
lower road edge that are absent from the second, so the recorded direction is
removal. The application cannot verify whether those supplied dates represent
the intended synthetic chronology.

The child MCI record used those exact prepared paths and hashes. It returned
65,536 background pixels, zero road-change pixels, zero building-change
pixels, and raw caption `the scene is the same as before`. Standalone CPU MCI
on the exact same ordered files reproduced byte-identical mask, overlay,
comparison and caption artifacts. Therefore preparation and timeline
integration did not create the miss. A reversed-order diagnostic predicted
4,827 building-change pixels and captioned houses being built, demonstrating
direction sensitivity; GeoLens does not reverse the user-supplied chronology
to obtain that positive result.

The primary result is a genuine MCI missed detection for the recorded removal
direction, not evidence that nothing changed. A smaller answer-evidence bug was
fixed: the exact caption phrase `the scene is the same as before` is now
recognised as an explicit model no-change statement rather than `unclear`.
The checked answer still says only that the model did not identify building
changes and retains the model-interpretation qualification.

### Vegetation history

The Alappuzha preset, custom map, validated plan, explicit Run action,
exact-AOI cache and corrected NDVI processing are unchanged. The main answer
now expresses only supported direction and mixed spatial behaviour in ordinary
greenness language. It contains no raw NDVI values, pixel counts, coverage
percentages or threshold notation. It does not use undocumented “slight” or
“large” severity labels. Exact values, populations, coverage, thresholds,
provenance, cache details and records remain downloadable in the evidence
expander.

The real cached Streamlit run again accepted 2021, 2023, 2024 and 2025, kept
2022 unavailable, loaded all 20 selected windows from the exact-AOI cache and
retrieved none. The constrained claim plan passed without fallback. Numerical
request, observation, pairwise, grid and limitation records match the corrected
baseline; the common-valid mask GeoTIFF is byte-for-byte identical. Only unique
output paths and the intentionally revised prose differ.

- Historical record:
  `outputs/geolens/historical/run-20261029T034940021165Z-a5697dcee0/run.json`
- Assistant record:
  `outputs/geolens/assistant/assistant-20261029T034939746378Z-23a8ea0af2/assistant.json`

### Verification

- All 101 regression tests pass, including focused presentation, grounded
  follow-up and PDF-export tests plus the preserved MCI, prepared two-date
  vegetation, historical cache/scaling and language-answer suites.
- The expanded real `ollama/qwen3:8b` compatibility command passes all 13
  structured cases: ten vegetation-history cases and three uploaded-pair cases.
- Streamlit's integration harness completed the real three-observation upload,
  date ordering, alignment confirmation, explicit Run action, three MCI calls,
  interval and parent downloads, and expandable evidence without exceptions.
  No browser was connected to the browser control surface, so visible browser
  interaction is not claimed.

## Infrastructure demo feasibility and typed observations

### Exact 2023–2025 Alappuzha result: blocked by source imagery

The exact request `Compare infrastructure changes between 2023 and 2025.` is
not currently executable as an evidence-backed infrastructure analysis. The
confirmed Alappuzha AOI is retained, but no correctly dated, aligned,
appropriately licensed high-resolution 2023/2025 pair for those exact bounds is
available locally or through the checked open sources.

Local imagery inventory:

- the Alappuzha cache contains Sentinel-2 Collection-1 Level-2A observations
  acquired 2023-01-08 and 2025-01-17 on a 10 m grid. These are valid for the
  existing NDVI workflow, not for resolving or counting individual buildings,
  roads, or trees;
- `data/input/before.png` and `after.png` are ungeoreferenced 1254 × 1254 RGB
  files with no acquisition dates or AOI identity. They cannot establish
  Alappuzha provenance or a 2023/2025 interval;
- the pinned RSICC examples are 256 × 256 LEVIR-derived academic samples, not
  Alappuzha. The upstream preparation notes no explicit redistribution licence,
  and the source files contain neither exact acquisition dates nor geospatial
  metadata. They are not relabelled as 2023/2025.

Authoritative source investigation:

- the public Maxar Open Data bucket and event metadata were queried without
  downloading imagery. It is an event-activation collection under CC BY-NC
  4.0. The only 2023 India activation is **North India Floods**, covering the
  Sikkim/Teesta event; no Alappuzha activation or corresponding 2025 pair is
  listed;
- Planet NICFI tropical basemaps are 4.77 m mosaics under a custom
  non-commercial, forest-monitoring-purpose licence. Access requires a Planet
  account/EULA, they are month mosaics rather than single acquisition dates,
  and the programme stopped adding monthly basemaps after December 2024. They
  therefore provide neither the requested 2025 pair nor suitable evidence for
  validated individual-building counts;
- commercial Maxar/Planet archive imagery may exist, but ordering requires an
  account/plan and potentially paid access. No paid search, order or download
  was made.

No infrastructure finding is reported. In particular there is no supported
building count, net count change, new/removed-building count, road-construction
claim, spatial location, or evidence mask for Alappuzha in 2023–2025.

### Implemented request and observation contract

The local `qwen3:8b` interpretation schema now keeps five intents distinct:
vegetation greenness (the existing NDVI tool), vegetation cover, individual
tree count, building/construction change, and road change. The exact word
`infrastructure` resolves to both building and road intents. `buildings and
roads` is handled identically. Dates and the confirmed AOI still pass through
the deterministic source-evidence, application-state and date validators.

The exact infrastructure question was exercised through Streamlit's real-model
integration harness. It produced:

- analyses `construction` and `roads`;
- requested years 2023 and 2025;
- boundary source `application_state`, label `Alappuzha demo area`, and bounds
  `[76.34, 9.48, 76.36, 9.5]`;
- a blocked plan with no Run button and no model execution;
- two structured `not_run` observations (`building_change` and `road_change`)
  with empty actual dates and evidence references, null count/area/location,
  and the verified imagery/method limitation.

Successful NDVI executions now also expose structured observations with change
type, actual dates, evidence-map references, method, pair-valid population,
threshold/categories and limitations. Count, vegetation-cover area and spatial
location remain null because the current NDVI tool does not compute them.
Combined requests still run eligible NDVI tools and retain separate not-run
observations for unsupported intents. Deterministic summaries explain the
validated plan in plain language; unsupported intent is never rewritten as an
NDVI result.

### Separate annotated demonstration proposal

If a building-count and temporal-matching demonstration is required before a
suitable Alappuzha pair is licensed, use a clearly labelled **SpaceNet 7
benchmark AOI**, for example training AOI
`L15-0760E-0887N_3041_4643_13`. Do not label its monthly 2017–2020 Planet Dove
mosaics as Alappuzha or 2023/2025. SpaceNet 7 provides about 4 m imagery,
monthly building-footprint labels and temporal identities under CC BY-SA 4.0;
its SCOT evaluation covers both object tracking and change detection.

The official SpaceNet 7 baseline/winning code and public weights are the model
candidates, not an integrated GeoLens capability. Before integration they need
a bounded sample download, dependency/security review, M4 16 GB inference
benchmark, evaluation against held-out annotated months, and inspection of
count and temporal-match errors. The official solutions are Docker/GPU-oriented
and the full dataset is large, so neither data nor weights were downloaded in
this task. MCI remains useful only for localisation/captioning on suitable
aligned 0.5 m-style RGB pairs; its class mask components are not building
instances.

Primary references: [Maxar Open Data on AWS](https://registry.opendata.aws/maxar-open-data/),
[Planet NICFI user guide](https://assets.planet.com/docs/NICFI_User_Guide_EN.pdf),
[SpaceNet 7 challenge and licence](https://spacenet.ai/sn7-challenge/), and
[SpaceNet 7 winning solutions](https://github.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions).

## Current GeoLens delivery

### Implemented and verified

- The sidebar now contains exactly **Buildings and infrastructure** and
  **Vegetation history**, with Vegetation history selected by default. The
  prepared two-date vegetation implementation and tests remain intact outside
  the main selector.
- A new historical session activates the Alappuzha demo bounds through
  `confirm_aoi`, displays the 4.878053 km² active area, and keeps the map and
  chat ahead of the fallback controls. An unchanged rerun preserves that
  confirmation. Custom drawing is secondary; a changed or reset boundary
  invalidates the old confirmation, plan, execution, answer and result.
- Plan review and the explicit Run action are unchanged. The manual structured
  request is now in a fallback expander. The readable checked answer remains
  above images and charts; raw plans, pixel populations, provider details and
  machine records remain expandable and downloadable.
- The full lightweight suite passes 87 tests. Three Streamlit interface tests
  verify workflow availability/default selection, one-time demo confirmation,
  unchanged reruns, and reset invalidation. Existing tests continue to cover
  changed-boundary invalidation, combined intent, missing years, unsupported
  tools, cache identity, construction and two-date vegetation.

The real updated Streamlit integration flow asked `Compare vegetation from
2021 to 2025.` using local `qwen3:8b`. It produced a ready plan on the exact
demo bounds, ran only after the Run action, loaded all 20 selected windows from
the matching cache, and returned the established partial result: 2021, 2023,
2024 and 2025 available; 2022 unavailable; 37,316 all-observation common-valid
pixels (76.8277%). The constrained claim plan passed without fallback and the
app rendered 15 downloads with no exception.

- Assistant record:
  `outputs/geolens/assistant/assistant-20261029T034939746378Z-23a8ea0af2/assistant.json`
- Historical record:
  `outputs/geolens/historical/run-20261029T034940021165Z-a5697dcee0/run.json`

### Investigated but unverified

**Vegetation extent.** Dynamic World V1 is the most plausible candidate found:
it is a 10 m, nine-class probability/label product derived per Sentinel-2 L1C
observation, available from 2015 onward under CC BY 4.0. The relevant candidate
vegetation classes are trees, grass, flooded vegetation, crops, and
shrub/scrub. Google recommends thresholding top-class probability rather than
accepting every label. GeoLens has not established or locally validated an
Alappuzha-specific probability threshold, temporal compositing policy, class
aggregation, registration to the accepted L2A observations, or transition
error matrix. Raw class/probability rasters, validity masks and projected area
accounting would all be required before reporting square kilometres.

ESA WorldCover was also assessed. It is a 10 m, 11-class CC BY 4.0 product, but
only reference years 2020 and 2021 are available. ESA warns that the releases
used different algorithm versions, so differences combine actual land-cover
change and algorithm change. It cannot support the requested 2021–2025 annual
workflow. SCL remains a quality mask; no SCL transition is reported as
vegetation-area change.

**Buildings.** Google's Open Buildings 2.5D Temporal product covers India with
annual 2016–2023 building-presence and fractional-count rasters under CC BY 4.0
or ODbL 1.0. It is generated from 32-image Sentinel-2 stacks around 30 June,
not the individual historical acquisitions used here, and has about 4 m
effective resolution. The publisher states that presence confidence is
uncalibrated and that cross-year spatial drift, clouds and tiling can affect
results. Its rasters do not identify individual temporal instances. The model
is available only by request and the published large-scale inference used
TPUs. The static Open Buildings high-resolution source imagery/annotations are
not publicly released, and no suitable freely usable dated high-resolution
Alappuzha image series plus locally evaluated detector was established.

Primary references: [Dynamic World V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1),
[ESA WorldCover data access](https://esa-worldcover.org/en/data-access), and
[Open Buildings 2.5D Temporal](https://sites.research.google/gr/open-buildings/temporal/).
No new imagery or model was downloaded during this assessment.

### Blocked or planned

- Vegetation-covered area, vegetation/nonvegetation transitions and ground-area
  totals remain blocked on product access plus a documented probability,
  compositing, registration and local validation design. They are not inferred
  from NDVI or SCL.
- Historical building counts remain blocked on a suitable dated image/product
  validation for this exact AOI and period. “New” and “removed” buildings would
  additionally require temporal object identities and matching; subtracting
  annual fractional counts would provide only an uncertain net estimate.
- The existing MCI workflow remains the supported construction option only for
  caller-supplied aligned 256 × 256 RGB pairs. It is not a building counter and
  is never run on upsampled Sentinel-2 data.
- No browser was connected to the browser-control surface, so visible browser
  interaction is not claimed. The real Streamlit integration harness and
  loopback server are verified separately.

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
question. The strengthened 13-case real compatibility run passed. Ten cases
cover the historical/map contract, including infrastructure, buildings and
roads, combined vegetation plus infrastructure, vegetation cover and
individual-tree-count intent. Three more cover uploaded-pair building/road,
count and spatial-description questions:

- vegetation here from 2021 to 2025 → vegetation only, correct years and
  boundary reference;
- construction and roads → both intents preserved, not substituted with
  vegetation;
- vegetation and construction → both intents preserved for a partial-support
  plan;
- vegetation with no dates → start and end years reported as missing rather
  than filled by the model;
- Kochi while the Alappuzha preset is active → named-location confirmation
  required rather than silently using either location.

The model cannot receive or generate an AOI coordinate field. The application
still requires an explicitly confirmed map boundary and applies the existing
year, request-size, seasonal, quality, cache and sensor validators. Unsupported
construction/road intent is retained in the plan with the exact requirement
for aligned 256 × 256 RGB inputs in Buildings and infrastructure. The model can
invoke no Python, shell command or arbitrary tool: only the predefined
historical vegetation wrapper can reach `run_historical`, and only after the
user reviews the plan and clicks Run.

Final-answer generation receives a compact copy of requested/available years,
actual dates, missing observations, common-pixel metrics, pair-valid
comparisons, thresholds, limitations, unsupported analyses and tool failures.
The model does not write unrestricted prose. It selects a schema-constrained
answer kind, lead claim and complete claim-ID set; the application renders
evidence-backed prose from the execution record. An invented claim ID, extra
number/date field, wrong conclusion or missing claim triggers a readable
deterministic fallback.

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
The displayed answer used the actual dates and documented population
distinction while leaving raw values, coverage and pixel populations in the
technical evidence section.

The audit record contains the original question, provider/model identifier,
exact evidence, validated interpretation, selected tool and arguments,
execution status, result references and final answer metadata. Question, draft
AOI or boundary confirmation changes clear stale plans and answers. Ordinary
Streamlit reruns do not execute the tool again.

### Language-interface verification and limitations

- 87 lightweight tests pass. Twenty-seven historical language-interface tests cover unavailable
  service, missing/invented dates, unconfirmed AOI, unsupported construction
  and roads, combined partial planning/execution, preset/application-state AOI
  resolution, conflicting places, boundary invalidation, tool failure, missing
  numerical fact references and accepted deterministic fact rendering.
- The real 13-case `scripts/check_llm.py` compatibility test passed with
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

## Confirmed-AOI live integration correction

The user-facing failure was reproduced and traced to the server process rather
than a second checkout. Port 8501 was served by PID 85235 from
`/Users/aravindrkrishnan/Dev/satquery`, the only SatQuery `app.py` found under
the development tree, but that process had started at 21:30 before the
confirmed-AOI integration was written. Its UI and planner matched commit
`9cd1947`: the UI displayed “Confirmed AOI” while the old planner required an
explicit source-text phrase such as “here.” It therefore blocked
`compare the vegetation from 2022 to 2024` with `confirmed_boundary` despite
the session containing a confirmed AOI.

The request path now keeps two explicit records. `raw_model_interpretation`
retains the validated model output and its literal evidence; a separate
`boundary_resolution` resolves the authoritative `ConfirmedAOI` from
Streamlit application state. The latter records source `application_state`,
the active AOI source and label, exact WGS84 bounds, conflict status and final
missing-information list. No boundary evidence is inserted into the raw model
record. The planner receives the same `ConfirmedAOI` object shown by the UI.

The exact request was exercised through Streamlit's interaction harness with
real local `qwen3:8b`. With the Alappuzha preset confirmed, the raw model
reported no explicit boundary reference and no boundary evidence; application
state resolved bounds `[76.34, 9.48, 76.36, 9.5]`; the plan was ready for
years 2022–2024 and displayed the explicit Run action. Separate real
interaction tests kept an unconfirmed AOI blocked by `confirmed_boundary` and
kept Kochi blocked by `named_location_confirmation`. Re-confirming even the
same area now clears any stale blocked plan and chat state.

## Conversational evidence-backed answers

Historical results now lead with a short GeoLens answer that explains NDVI on
first use as a satellite-based indicator related to vegetation greenness,
names the actual acquisition dates, describes only available adjacent
comparisons, explains missing years naturally and ends with one relevant
limitation. The renderer explicitly distinguishes the all-observation
common-pixel population from each pair-valid population. A small mean
difference is shown alongside the local increase/decrease/stable fractions and
is not generalized into a claim that the whole area was stable.

The local `qwen3:8b` model selects only a constrained claim plan. GeoLens—not
the model—renders dates, numbers, intervals and prose from checked structured
results. Validation therefore covers conclusion type, claim identity,
comparison interval and population identity in addition to numerical source
values. It does not claim to validate arbitrary language merely by checking
numbers. Missing observations are not bridged, spatial findings are never
invented, and unsupported construction/road analyses or tool failures receive
separate no-finding statements.

The answer is prominent above the existing imagery and charts. Raw plans,
per-year metrics, pairwise records, provider timings and result JSON are in
expandable evidence sections. `answer.md` is downloadable, while the full
assistant JSON retains the claim plan, model runtime, fallback flag and
rejection reason. Session state prevents regeneration on unrelated reruns.

### Real conversational-answer verification

Assistant audit record:
`outputs/geolens/assistant/assistant-20261029T004414533960Z-517476cfa3/assistant.json`

Answer file:
`outputs/geolens/assistant/assistant-20261029T004414533960Z-517476cfa3/answer.md`

Historical execution record:
`outputs/geolens/historical/run-20261029T004414731762Z-bffd12a90c/run.json`

The real cached Alappuzha Streamlit flow used all 20 matching cached band
windows and retrieved zero. `qwen3:8b` returned the complete constrained claim
plan without fallback. The rendered answer used the actual 2021, 2023, 2024
and 2025 dates; identified 2022 as unavailable; withheld the two adjacent
intervals touching 2022; described the supported 2023→2024 and 2024→2025
pair-valid comparisons; and kept the 2021→2025 endpoint population separate
from the 37,316-pixel all-observation population. Streamlit's interaction
harness rendered the answer, imagery, charts, evidence expanders and 15
downloads without exceptions, including **Download this answer**. No browser
was connected to the browser-control surface, so visible browser interaction
is not claimed.

## Reproducible Alappuzha demo preset

Historical mode is now the default and activates the Alappuzha demo area once
when a new session first enters the app. It validates the tested WGS84 bounds
`[76.34, 9.48, 76.36, 9.50]` through the same `confirm_aoi` contract as a
custom selection, makes it the active confirmed AOI, displays its 4.878053 km²
calculated area and labels the map rectangle “Alappuzha demo area.” It records
`demo_preset` as the boundary source. **Load Alappuzha demo area** remains
available to restore the preset after a custom or reset selection.

Activating the preset clears any plan, execution, answer or historical result
from another AOI and defaults the manual controls to 2021–2025. Years extracted
from a new language request remain independent. Switching to a new draft custom
rectangle clears the preset confirmation and all stale plan/result state.

Confirmed-area context is now deterministic. With the preset active,
`Compare vegetation from 2021 to 2025` creates a runnable plan without the word
“here.” The interpretation record uses boundary source `demo_preset` from
application state and supplies no invented boundary-text evidence. The UI and
plan display “Using the Alappuzha demo area.” An explicitly named different
location is recorded and blocked with `named_location_confirmation` until that
place is resolved and confirmed.

Cache-hit reporting was strengthened without changing retrieval or analysis.
Search cache reads must match the exact bbox, year interval, collection,
candidate limit and observation identities. Raster cache reads must match the
observation ID, band/asset URL, collection, AOI key, target grid, resampling,
dtype, nodata and item scale/offset. A mismatched cache is never used as a
network fallback.

### Real preset verification

Assistant audit record:
`outputs/geolens/assistant/assistant-20261028T165355469985Z-4662ff0ced/assistant.json`

Historical execution record:
`outputs/geolens/historical/run-20261028T165355641047Z-301be5429a/run.json`

The Streamlit harness loaded the preset, confirmed the exact bounds and area,
showed the label, retained manual defaults 2021–2025, interpreted the original
question without “here,” displayed the application-state boundary in the plan
and executed only after the explicit Run action. The result loaded all 20
selected observation windows from validated matching-AOI caches and retrieved
zero. It remained honestly partial based on the actual execution: 2021, 2023,
2024 and 2025 were available and 2022 was unavailable. The supported final
answer passed without fallback.

This is a reproducible local demonstration preset, not a claim of production
readiness or unrestricted geographic support. Custom rectangles remain
available within the documented area, request and provider constraints.

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
- a real local-Ollama chat mode plus manual structured controls in a fallback
  expander. The model
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

- 87 lightweight tests pass: the infrastructure timeline and two-date
  vegetation regressions, eleven historical tests, twenty-seven historical
  language-interface tests, twenty-two uploaded-infrastructure/timeline tests and three
  focused Streamlit layout/state tests. Historical coverage includes unconfirmed and oversized AOIs,
  stale-state invalidation, invalid dates/request size, unsupported
  construction, scaling representations, cache failure, missing-year
  adjacency, and insufficient common coverage.
- Streamlit's integration harness performed the real cached historical flow
  from validated automatic demo activation through plan validation,
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

GeoLens now presents exactly two main workflows while preserving the existing
construction, corrected two-date vegetation and historical execution
functions. It leaves the `satquery` package and repository name unchanged and
adds no agent framework, authentication, database, cloud-model dependency or
deployment.

Buildings and infrastructure:

- requires a natural-language question and both images;
- accepts PNG/JPEG uploads up to 10 MB under unique internal filenames;
- accepts optional user-supplied dates without treating them as image-derived
  metadata;
- requires an explicit same-location/already-aligned confirmation;
- accepts the model's native 256 × 256 RGB contract, or optionally creates and
  previews explicit Lanczos-resized copies with a warning that resizing does
  not align imagery or guarantee suitable resolution;
- calls the existing Change-Agent CPU runner only on the Analyse button;
- caches the verified model load in the Streamlit process and retains the
  latest execution record in session state;
- displays checked prose, prepared images and overlay first, with the class
  mask, raw unedited caption, predicted pixel fractions, downloads and records
  in expandable evidence;
- leads with a conservative ordinary-language caption interpretation only when
  the corresponding change category is present, and surfaces unclear or
  contradictory evidence rather than selecting one output silently.

Vegetation history retains the map/chat and corrected annual NDVI path. Its
main prose omits dense metrics while exact values remain in expandable
evidence. The prepared two-date runner remains implemented, tested and
available through its command-line entry point, but is not a third main choice.

Every GeoLens execution uses a new directory beneath the ignored
`outputs/geolens/` tree. The app performs no asset download at startup and
reports missing checkpoint, upstream source, prepared raster, upload, and
result-contract errors in user-facing language.

### GeoLens verification

- Streamlit 1.66.1, Folium 0.20.0 and streamlit-folium 0.27.4 are installed
  only in the project `.venv` and pinned in `requirements.in`.
- All 87 lightweight tests pass. The local-interface tests cover workflow
  selection/default state, one-time demo confirmation and reset invalidation, plus bounded
  PNG/JPEG decoding, explicit resizing, RGB rejection, unique run/input paths,
  and execution-record validation; eleven historical, twenty-seven historical
  language and twenty-two uploaded-infrastructure/timeline tests cover the contracts listed
  above.
- Streamlit's integration harness rendered both main workflows without
  exceptions and ran real MCI and exact-AOI cached historical analysis, not
  mocks. The prepared two-date analysis remains covered by its existing
  regression and real-output checks.
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
