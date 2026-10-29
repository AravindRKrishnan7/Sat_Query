# SatQuery AI

SatQuery AI is an early satellite change-analysis prototype with exactly two
user-facing GeoLens workflows over two independent analysis paths:

- **Buildings and infrastructure:** consecutive Change-Agent MCI inference on
  a timeline of two to four dated, already aligned RGB images.
- **Vegetation history:** a bounded historical Sentinel-2 annual NDVI timeline
  for a user-confirmed rectangle.

The corrected prepared two-date Sentinel-2 comparison remains implemented and
covered by tests and command-line instructions, but it is no longer a separate
main-selector workflow.

The vegetation calculations remain deterministic numerical geospatial
processing. A separate, constrained local language-model layer may interpret a
request and explain validated tool results; it does not calculate or alter
reflectance, masks, NDVI, retrieval or model predictions.

## Launch the separate React frontend

The `figma-frontend-integration` branch contains a React 19, TypeScript, Vite
and Tailwind frontend under `frontend/`. Its visual layout was imported from a
Figma prototype, but its Figma deployment plugins, embedded repository
instructions, root Git files, simulated timers, fabricated metrics and sample
answers were not imported into the running application. The frontend now uses
the existing Python analysis modules through a thin loopback-only FastAPI
adapter. It does not call Streamlit as an API and does not duplicate MCI, NDVI,
request-validation, answer, follow-up or report calculations.

Prerequisites are the same Python environment, model assets, local Ollama
`qwen3:8b` service and cached/accessible satellite data described below, plus
Node.js 22.12 or later and npm 10 or later. Install dependencies once:

```sh
.venv/bin/python -m pip install -r requirements.in
cd frontend
npm install
cd ..
```

Start the backend from the repository root. Port 8010 is used because port 8000
may already belong to another local application:

```sh
PYTHONPATH=src .venv/bin/python -m uvicorn satquery.api:app \
  --host 127.0.0.1 \
  --port 8010
```

In a second terminal, start the frontend:

```sh
cd frontend
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The API is available only
at `http://127.0.0.1:8010`; CORS accepts only the configured local frontend
origin, which defaults to `http://127.0.0.1:5173`.

The infrastructure path validates two to four bounded PNG/JPEG uploads,
user-supplied unique dates, optional explicit resizing, endpoint policy and the
alignment confirmation before returning a reviewable plan. A separate Run
action submits the existing consecutive-pair MCI workflow to one background
CPU worker. The vegetation path validates the Alappuzha preset (or bounded
numeric custom rectangle), natural-language request and tool plan before its
separate Run action uses the existing exact-AOI historical engine. Both paths
poll real job state, display only registered artifacts for that analysis ID,
retain partial comparison failures, and bind grounded follow-ups and reports
to the completed saved record. Changing any active frontend input clears the
displayed plan and result.

Local plan/job state is intentionally in memory for this prototype. A backend
restart requires a new plan even though prior ignored output directories remain
on disk. There is no authentication, database, multi-user isolation or public
deployment. The React custom-area control is numeric; the preserved Streamlit
fallback remains the interface with interactive rectangle drawing.

## Launch the Streamlit fallback locally

GeoLens also retains the working Streamlit interface around those existing analysis paths.
It supports a loopback-only Ollama request interface but adds no agent
framework, authentication, database, cloud language-model dependency, or
public deployment.

Prerequisites:

- create `.venv` and install `requirements.in` as described below;
- run `./scripts/prepare_change_agent.sh` so the verified checkpoint and pinned
  upstream source are present;
- run `.venv/bin/python scripts/prepare_vegetation_sample.py` once if the fixed
  Sentinel-2 sample is not already present;
- allow outbound HTTPS to Element 84 Earth Search only when running a new
  historical request; no satellite data is downloaded at app startup;
- for natural-language requests, start a local Ollama service with the tested
  `qwen3:8b` model as described below. Manual structured controls remain
  available when Ollama is unavailable.

Launch the app from the repository root with exactly:

```sh
.venv/bin/python -m streamlit run app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). The server binds only to
the loopback interface. It does not download model or satellite assets on
startup.

GeoLens uses a native Streamlit light theme defined in
`.streamlit/config.toml`: an off-white page, white widget/card surfaces, dark
navy text, muted gray supporting text, teal actions and visible gray borders.
The sidebar uses the same light palette. App-level CSS is intentionally limited
to the GeoLens header, step labels and stable outer result containers; it does
not force colours on nested selectboxes, uploaders, radios, forms, tabs,
buttons, downloads or chat controls. This keeps normal, hover, focus and
disabled states under Streamlit's accessible theme system.

**Buildings and infrastructure** requires a natural-language question and two
to four PNG/JPEG observations. Every image requires a user-supplied ISO
acquisition date; GeoLens rejects missing or duplicate dates, sorts the images
chronologically, and previews the order before execution. It runs the existing
two-image model independently for every consecutive pair. With four images,
that means A→B, B→C and C→D. An optional first-to-last comparison is a separate
model run—not a sum of intermediate predictions and not evidence of continuous
growth. Completed intervals remain available if a later comparison fails.

The model still receives only 256 × 256 RGB inputs. Optional resizing is
explicit, produces prepared-copy previews, and does not imply registration,
alignment, suitable ground resolution, or restored detail. Analysis starts
only after the user confirms that all observations depict the same extent and
are already aligned, then clicks **Run timeline analysis**.

The local language model extracts constrained building, road, general-change,
count and location-question intent. GeoLens then renders checked prose from
that intent and the recorded MCI mask classes. It does not let the language
model invent objects or locations. MCI does not detect or temporally match
building instances, so questions such as “How many new buildings?” receive an
explicit unsupported-count explanation; connected mask components are never
reported as buildings. Road-change pixels are not called new roads. The raw
caption remains a separate model-head output and may disagree with the mask.
Each interval preserves the original MCI record plus `answer.md` and an
assistant audit record. A parent `timeline.json` records chronological dates,
original upload filenames and SHA-256 hashes, prepared-file hashes and settings,
interval status, answers, overlays and record references. Changing any image,
date, question, resizing choice or endpoint option invalidates the displayed
timeline without deleting its unique output directory.

**Vegetation history** is the default GeoLens mode. On its first session run it
activates the documented Alappuzha demo area through the same validator used
for custom rectangles, shows the confirmed bounds and area, and keeps the map
and chat as the primary interaction. Custom selection remains available in a
secondary expander; changing or resetting the area invalidates the old plan and
result. The non-LLM structured request is retained in a fallback expander. The
workflow supports two to five annual observations, defaults to 2021–2025, and
rejects historical construction/road requests rather than applying the
fine-resolution MCI model to Sentinel-2. Its chat field uses the local model
only to extract a typed plan and choose from constrained, validated result
statements. The main answer describes the direction of satellite-observed
vegetation greenness in ordinary language without raw index values, pixel
counts or threshold notation. Exact NDVI metrics, populations, coverage,
thresholds and provenance remain under **View evidence and technical
details**. Analysis still starts only after an explicit Run action.

The sidebar contains only **Buildings and infrastructure** and **Vegetation
history**. Their inputs and results are not moved between workflows. The
prepared two-date NDVI runner remains available through the documented command
line path below.

### Explore and export completed results

Completed results use a restrained navy-and-teal presentation layer while the
underlying records and calculations remain unchanged. In **Buildings and
infrastructure**, select one saved interval at a time and move the
before/after divider to compare the exact prepared inputs recorded for that
child execution. The divider is display-only. A side-by-side fallback and the
prediction overlay remain available independently. GeoLens verifies the
prepared paths and SHA-256 values before composing the comparison view.

In **Vegetation history**, choose one dated true-colour/NDVI observation and
one available or unavailable NDVI interval at a time. Missing-year warnings
remain visible, and the fixed difference legend continues to distinguish NDVI
increase, decrease, little difference and excluded pixels.

Both workflows include **Ask about this completed analysis** after a retained
result is active. The local `qwen3:8b` model selects only a constrained
follow-up intent; GeoLens renders the answer deterministically from the saved
record. Questions about unsupported counts, causes, vegetation cover, trees,
ambiguous intervals or different dates cannot create findings. If Ollama is
unavailable or its response is rejected, the same evidence-bound renderer uses
a clearly recorded deterministic fallback. Follow-ups never rerun inference,
retrieval or NDVI, and threads are keyed to the active analysis ID.

The **Download illustrated PDF report** expander creates a local A4 report on
explicit request. Infrastructure reports include the original question,
chronological observations, full input hashes, user-supplied-date warning,
per-interval images, overlays, captions, class populations, failures and model
limitations. Vegetation reports include the confirmed area, actual and missing
observations, true-colour/NDVI previews, difference maps, exact pairwise
statistics, scaling provenance and interpretation limits. Reports are built
only from saved outputs; exporting cannot change or rerun an analysis. The PDF
runtime dependency is included in `requirements.in`.

GeoLens writes every execution to a new ignored directory beneath
`outputs/geolens/`. Infrastructure timelines use
`outputs/geolens/construction-timelines/`; their prepared observations and
comparison artifacts remain within one unique parent directory. Weights,
prepared data, uploads, and generated results remain excluded from Git. The
local upload limit is 10 MB per image.

## Run the historical Alappuzha demonstration

The documented rectangle is WGS84 bounds
`[76.34, 9.48, 76.36, 9.50]`, approximately 4.878 km² and below the
configurable 25 km² limit. The preset labels this rectangle “Alappuzha demo
area,” displays it on the map, records `demo_preset` as its application-state
source, and sets the manual year defaults to 2021–2025. It does not overwrite
years explicitly extracted from a new question. A direct verification uses
the same engine and defaults as the app:

```sh
.venv/bin/python scripts/run_historical_demo.py
```

For each year, GeoLens searches January 1–March 31 in
[Element 84 Earth Search](https://github.com/Element84/earth-search)'s
[`sentinel-2-c1-l2a` collection](https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a).
It ranks by catalogue cloud cover,
inspects at most five candidates, and accepts the first with at least 80% SCL
4/5/6 coverage inside the AOI. It never widens the season to hide a missing
year. Only the AOI windows for blue, green, red, NIR and SCL are read.
Continuous bands use bilinear resampling and SCL uses nearest neighbour on one
10 m UTM grid.

Collection-1 reflectance scaling is verified independently for every selected
asset: item-level `raster:bands` scale/offset must exactly match the remote COG
header before the affine conversion is applied. The verified acquisitions use
`stored * 0.0001 - 0.1`. This differs from the legacy Earth Search assets in
the two-date scaling audit; GeoLens does not reuse the legacy offset decision
blindly. The annual true-colour panels use one fixed display-only 0.00–0.30
reflectance range and gamma 2.2; neither setting changes reflectance or NDVI.

The verified 2021–2025 run was partial: 2021-02-07, 2023-01-08, 2024-01-18
and 2025-01-17 were accepted. The sole January–March 2022 candidate had only
0.0247% accepted in-AOI SCL coverage, so 2022 remained unavailable. The
population valid across all four available observations retained 76.83% of the
AOI. Its mean NDVI was 0.4664 in 2021 and 0.4824 in 2025; this is an NDVI
comparison, not evidence of vegetation gain/loss, plant health, deforestation,
damage or cause. Adjacent comparisons touching 2022 are shown as unavailable
and are not bridged.

Exact-AOI search results and prepared raster windows are cached under the
ignored `data/cache/geolens-historical/` tree. Each execution still creates a
unique ignored directory under `outputs/geolens/historical/`. A verified repeat
loaded all 20 selected band windows from cache and did not retrieve another
AOI's data. Annual true-colour/NDVI PNGs, available difference PNGs, the
deterministic summary and result JSON are downloadable in the app.

Before reporting a cache hit, GeoLens now revalidates the exact search request,
provider collection, observation IDs, asset/band URLs, AOI key, analysis grid,
resampling, cached raster dtype/nodata and reflectance scale/offset. A mismatch
uses the existing bounded retrieval path; if retrieval is unavailable, the
mismatched cache is rejected rather than substituted.

### Investigated historical capabilities

Vegetation-covered area and building counts are not currently supported
GeoLens findings. They remain separate from NDVI and changed-pixel outputs.

- [Dynamic World V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1)
  is a promising 10 m, nine-class, per-Sentinel-2 land-cover probability
  product available from June 2015 onward under CC BY 4.0. Its own guidance
  calls for top-class probability thresholding. GeoLens has not yet selected
  or locally validated a confidence/compositing policy for heterogeneous,
  water-rich Alappuzha, aligned it to the accepted L2A observations, or
  evaluated transition errors. SCL classes are therefore not repurposed as a
  vegetation-extent classifier.
- [ESA WorldCover](https://esa-worldcover.org/en/data-access) is 10 m and CC BY
  4.0, but only the 2020 and 2021 maps are available. ESA also warns that those
  maps use different algorithm versions, so their differences mix real change
  with algorithm change. It cannot provide the requested 2021–2025 timeline.
- [Open Buildings 2.5D Temporal](https://sites.research.google/gr/open-buildings/temporal/)
  provides annual building-presence and fractional-count rasters for India
  from 2016–2023 under CC BY 4.0 or ODbL 1.0. It is derived from stacks of 32
  Sentinel-2 images, not the same individual GeoLens observations; it does not
  identify temporal building instances, its confidence is uncalibrated, and
  the publisher documents spatial drift. The model is available only by
  request and was run on TPUs. No suitable freely usable, dated
  high-resolution image series and evaluated local model for the same
  Alappuzha area was established, so GeoLens does not expose counts or
  new/removed-building claims.

Adding either capability requires a separate, reproducible validation against
local reference data, exact provenance and masks, and projected ground-area or
temporal object-matching evaluation. No large asset or model was downloaded for
this investigation.

## Local language-model interface

The verified provider is local Ollama 0.32.5 with the already-installed
`qwen3:8b` model (8.2B parameters, Q4_K_M). No OpenAI, Azure OpenAI,
Anthropic, Google or other external API credential/provider was configured or
used. GeoLens accepts only a loopback Ollama endpoint and never downloads a
model automatically.

Start and verify the service:

```sh
ollama serve
# Run this only when the model is not already installed:
ollama pull qwen3:8b
.venv/bin/python scripts/check_llm.py
```

The optional configuration variables are:

```sh
export GEOLENS_OLLAMA_URL=http://127.0.0.1:11434
export GEOLENS_OLLAMA_MODEL=qwen3:8b
```

The compatibility command makes thirteen small structured-output calls: ten
vegetation-history cases covering
infrastructure, buildings plus roads, vegetation greenness, vegetation cover,
individual-tree counts, combined requests, omitted dates, and a conflicting
named location, plus three uploaded-pair cases covering building/road intent,
unsupported building counts and requested spatial descriptions. It does not
run satellite retrieval or MCI inference. The app only probes
Ollama's model list on startup; it loads the model after the user submits a
question.

Interpretation uses a JSON schema with separate vegetation-greenness,
vegetation-cover, individual-tree-count, construction/building and road
intents, years, and a reference to the current boundary. “Infrastructure”
resolves to both building and road intent. Every asserted intent and boundary
reference must include an exact source-text quotation.
Extracted years are discarded unless the exact year token occurs in the
question. The result is then passed through the existing AOI, date-range,
five-observation and sensor-suitability validators. The model receives no
coordinate field and cannot create or replace a confirmed AOI.

When an AOI is already confirmed, a question such as `Compare vegetation from
2021 to 2025` uses that AOI deterministically from application state; the word
“here” is not required and no boundary quote is invented. The plan displays
“Using the Alappuzha demo area” for the preset. An explicitly named conflicting
place, such as Kochi while Alappuzha is active, is blocked until that different
location is resolved and confirmed.

The historical plan preserves unsupported construction/road intent and
explains that aligned 256 × 256 RGB before/after inputs belong in **Buildings
and infrastructure**. A combined request can run the supported historical vegetation tool only
after the plan is reviewed and the user clicks **Run interpreted request**.
GeoLens never executes model-generated code, shell commands or arbitrary tool
names.

For the final answer, the model receives a compact result containing requested
and available years, actual dates, missing observations, common-pixel metrics,
pair-valid comparisons, thresholds, unsupported analyses and tool failures.
The model does not write unrestricted prose: it selects a schema-constrained
answer kind, lead claim and complete set of allowed claim identifiers. GeoLens
then renders short conversational paragraphs from checked dates, intervals,
metric identities and pixel populations in the execution record.

This separation prevents an invented number, date, comparison interval or
unsupported conclusion from entering the displayed answer. Validation checks
the claim set and conclusion type as well as the numerical source values; it
does not claim that number checking alone verifies arbitrary prose. If the
model service or claim plan fails validation, GeoLens renders the same readable
evidence-backed explanation deterministically and records the rejection reason
in technical details.

The conversational answer appears before the imagery and charts and is saved
as a downloadable `answer.md`. Maps and satellite previews remain directly
visible. Raw plans, per-year metrics, pairwise records, provider details,
timings and machine-readable JSON are placed in expandable **View evidence and
technical details** sections. Ordinary Streamlit reruns reuse the answer kept
in session state rather than regenerating it.

Each interpreted execution writes an ignored
`outputs/geolens/assistant/assistant-*/assistant.json` audit record containing
the original question, provider/model, evidence-backed interpretation,
validated tool arguments, execution status, result references/errors, final
answer validation metadata and the `answer.md` path. Changing the question or
AOI invalidates the old plan and result.

## Current status

Real CPU inference succeeds on an Apple M4 using Python 3.11.17 and native
arm64 PyTorch 2.14.1. The verified example produced a 256 × 256 indexed mask
containing background (`0`) and building change (`2`) and the caption “some
houses are built along the road.” The caption is a separate model-head output,
not a fact derived from the mask.

MPS inference is not currently available in this process:
`torch.backends.mps.is_built()` is true, but `is_available()` is false. The
CPU path remains the supported prototype path. Accuracy has not been measured
because the selected official example does not include a matching reference
mask.

## Reproduce the experiment

Create a project-local Python 3.11 environment and install only the tested
direct dependencies. Do not install these globally.

```sh
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.in
.venv/bin/python scripts/check_environment.py
```

Fetch the pinned upstream source, required model assets, and the two small
academic-example images, verify their hashes, and apply the tracked
compatibility transformation:

```sh
./scripts/prepare_change_agent.sh
```

The preparation script is idempotent: existing verified artifacts are reused.
It does not download the LEVIR-MCI dataset or an external language model. All
downloaded code, weights, sample data, and generated outputs remain ignored by
Git. The sample repository does not state a dataset licence; use it only for
this academic experiment and do not redistribute it until permission is
clarified.

Run the command that succeeded locally:

```sh
.venv/bin/python scripts/run_change_agent.py \
  --before data/input/before.png \
  --after data/input/after.png \
  --checkpoint weights/MCI_model.pth \
  --output-dir outputs/change-agent-cpu \
  --device cpu
```

The adapter rejects unreadable, non-RGB, or non-256 × 256 inputs. It performs
no resizing or registration. A successful run writes `class_mask.png`,
`color_mask_with_legend.png`, `overlay.png`, `comparison.png`, `caption.txt`,
and `run.json` under the selected output directory.

Run the lightweight tests without downloading or loading model weights:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Compare vegetation index between dates

NDVI needs the Sentinel-2 red (B04) and near-infrared (B08) surface-reflectance
bands. The comparison figure also accepts native B03 and B02 reflectance bands
and combines them with B04 for true colour. An RGB PNG is insufficient as an
analysis input: it has no NIR channel and normally discards the radiometric,
no-data, CRS, transform, and quality-mask metadata needed for a defensible
calculation.

Copy [the example configuration](configs/vegetation.example.json), replace its
paths and provenance placeholders with two prepared, same-grid Level-2A
observations, then run:

```sh
.venv/bin/python scripts/run_vegetation_comparison.py \
  --config configs/vegetation.local.json
```

A fixed credential-free real example can be prepared and run with:

```sh
.venv/bin/python scripts/prepare_vegetation_sample.py
.venv/bin/python scripts/run_vegetation_comparison.py \
  --config configs/vegetation.real-example.json
```

The preparation script performs no live search: it reads the same 256 × 256
native 10 m window from B02, B03, B04, and B08 (plus the corresponding 128 ×
128 native 20 m SCL) for two pinned public Earth Search acquisitions over
Bengaluru. Downloaded rasters remain ignored by Git; exact product IDs, asset
URLs, processing baselines, window, scale/offset, and usage terms are tracked
in the configuration.

Each invocation creates a new timestamped directory beneath `output_root` and
writes quality-masked and raw numerical NDVI GeoTIFFs, a delta GeoTIFF,
common-valid and interpretation masks, a comparison PNG, `run.json`, and
`summary.txt`. The PNG has same-date true-colour before/after images above the
before NDVI, after NDVI, and NDVI-difference panels. B02/B03/B04 are required
to match the analysis grid exactly. Both RGB panels use the same display-only
reflectance range (`0.00–0.30`) and gamma (`2.2`); these adjustments never feed
back into NDVI, masks, GeoTIFFs, or statistics. The workflow rejects mismatched
spectral grids rather than resizing them. A 20 m SCL mask may be placed on the
10 m analysis grid only by the explicitly configured categorical
nearest-neighbour operation.

The command prints its new output directory. Open only its comparison image in
Preview, for example:

```sh
open -a Preview outputs/vegetation-real/run-YYYYMMDDTHHMMSSZ/comparison.png
```

The default example threshold of ±0.1 is an exploratory starting point, not a
universal change threshold. Output classes describe NDVI increase, decrease,
and approximate stability—not vegetation gain/loss, health, damage, or cause.
See the [vegetation comparison guide](docs/vegetation-analysis.md) for the full
input contract, preparation notes, interpretation, limitations, and test
status.

## Documentation

- [v1 prototype scope](docs/v1-scope.md)
- [model feasibility and first inference](docs/model-feasibility.md)
- [Sentinel-2 vegetation comparison](docs/vegetation-analysis.md)
- [Project status and scaling audit](docs/PROJECT_STATUS.md)
