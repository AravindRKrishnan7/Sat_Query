# SatQuery AI

SatQuery AI is an early satellite change-analysis prototype. It accepts two
already aligned optical images of the same place and runs the official
Change-Agent MCI architecture to produce a three-class change mask, visual
outputs, and a model-generated change caption.

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

## Documentation

- [v1 prototype scope](docs/v1-scope.md)
- [model feasibility and first inference](docs/model-feasibility.md)
