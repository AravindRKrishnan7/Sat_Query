# SatQuery AI

SatQuery AI is an early-stage satellite change-analysis project. The first
prototype will take two already aligned optical images of the same place and
produce a semantic change mask, a visual overlay, and—when the selected model
supports it—a short change caption.

## Current status

The repository currently contains the v1 scope, a feasibility review of the
official Change-Agent implementation, and a dependency-free environment check.
No model code, application code, checkpoint, language model, or dataset has
been downloaded or added.

Change-Agent is a plausible research baseline because its MCI checkpoint has a
joint three-class change-detection branch and caption decoder. It is not ready
to run on this Mac unchanged: upstream hard-codes CUDA, has command-line and
resize defects in its prediction path, and relies on an old OpenMMLab stack.
See [the feasibility report](docs/model-feasibility.md) for the evidence and
the proposed first experiment.

## Local environment

The selected isolation boundary is a project-local `.venv` using Python 3.9,
matching Change-Agent's documented Python version. On the investigated machine
it was created from `/usr/bin/python3` (Python 3.9.6). No inference packages are
installed yet.

```sh
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python scripts/check_environment.py
```

[`requirements.in`](requirements.in) is a minimal, evidence-based inventory
for the direct MCI inference graph. It is not a tested lock file; installing it
is deliberately deferred to the real-inference experiment because the legacy
MMCV/MMseg combination and Apple Silicon still need validation.

## Project documents

- [v1 prototype scope](docs/v1-scope.md)
- [Change-Agent feasibility](docs/model-feasibility.md)
