# SatQuery AI

SatQuery AI is an early satellite change-analysis prototype with two separate
capabilities:

- Change-Agent MCI inference on an already aligned RGB image pair.
- Deterministic NDVI comparison of two prepared Sentinel-2 Level-2A
  multispectral observations.

The vegetation workflow is numerical geospatial processing; it does not load
or call an AI model. The existing Change-Agent path remains independent.

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
bands. An RGB PNG is insufficient: it has no NIR channel and normally discards
the radiometric, no-data, CRS, transform, and quality-mask metadata needed for
a defensible calculation.

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

The preparation script performs no live search: it reads a 256 × 256 native
10 m window (and corresponding 128 × 128 native 20 m SCL) from six pinned
public Earth Search COG assets over Bengaluru. Downloaded rasters remain
ignored by Git; exact product IDs, asset URLs, processing baselines, window,
scale/offset, and usage terms are tracked in the configuration.

Each invocation creates a new timestamped directory beneath `output_root` and
writes before/after/delta GeoTIFFs, a common-valid mask, a comparison PNG,
`run.json`, and `summary.txt`. It rejects mismatched analysis grids rather than
resizing them. A 20 m SCL mask may be placed on the 10 m analysis grid only by
the explicitly configured categorical nearest-neighbour operation.

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
