# Change-Agent model feasibility

Verified 2026-10-22 against the official
[Change-Agent repository](https://github.com/Chen-Yang-Liu/Change-Agent) and
[published checkpoint repository](https://huggingface.co/lcybuaa/Change-Agent).
No checkpoint, language model, or dataset was downloaded during this review.

## Decision

Use the multi-level change interpretation (MCI) model as the first research
candidate, but do not treat upstream inference as Mac-compatible. Start with a
small, auditable adapter and CPU inference. Attempt MPS only after CPU produces
a valid output. The joint checkpoint supports both a three-class mask and a
caption, so an external LLM and the `lagent` agent layer are not required for
the v1 outputs.

The main risks are CUDA-hardcoded tensors, legacy MMCV/MMseg dependencies,
broken CLI argument handling, unsafe non-256 resizing, pickle-based checkpoint
loading, and unclear combined licensing terms.

## What was verified locally

- Repository: empty initial branch `main`, no commits, remote
  `https://github.com/AravindRKrishnan7/Sat_Query.git`.
- Hardware: MacBook Pro `Mac16,1`, Apple M4 (10 CPU cores), arm64, 16 GB unified
  memory.
- Operating system: macOS 15.6 (build 24G84).
- Storage at inspection time: 149 GiB available on the project volume.
- Memory at inspection time: `memory_pressure -Q` reported 40% system-wide free.
- Python found: Homebrew Python 3.11.17, framework Python 3.14.6, and Apple
  `/usr/bin/python3` 3.9.6.
- Isolation choice: `.venv` created with Python 3.9.6 to match upstream's stated
  Python 3.9 environment. It currently contains no model packages.
- PyTorch is absent in that environment, so CPU tensor execution and MPS
  availability have not yet been measured through PyTorch.

Hardware capacity is reasonable for a 914 MB checkpoint, but 16 GB unified
memory is not proof that deserialization plus the model and activations will
fit. Peak memory is part of the real test.

## What upstream states and the source actually does

### Entry points

The README documents direct inference as
[`Multi_change/predict.py`](https://github.com/Chen-Yang-Liu/Change-Agent/blob/main/Multi_change/predict.py),
invoked with `--imgA_path`, `--imgB_path`, and `--mask_save_path`. It calls both
`generate_change_caption()` and `change_detection()`. The README command has a
typo: its `--imgB_path` example repeats `{imgA_path}`.

The documented dataset evaluator is `Multi_change/test.py`. It requires the
LEVIR-MCI test split and reference captions, so it is not the entry point for a
single user-supplied pair.

The interactive agent examples (`try_chat.py` and `web_demo.py`) wrap the visual
model in `lagent` and configure either an OpenAI API model or a 7B InternLM model,
plus web search. That orchestration is not needed to run the MCI mask/caption
heads and is out of v1 scope.

The direct script is not runnable as documented without modification:

- Its outer parser accepts the three image arguments, then `Change_Perception`
  calls a second `parse_args()` using a parser that does not accept them. Normal
  documented arguments therefore become “unrecognized arguments.”
- Its defaults for `list_path` and sample images are developer-specific Windows
  paths.
- The result directory option is defined but the direct mask path is separate.

### Required files and sizes

| File | Purpose | Published size |
| --- | --- | ---: |
| [`MCI_model.pth`](https://huggingface.co/lcybuaa/Change-Agent/blob/main/MCI_model.pth) | Encoder, attentive encoder, and caption decoder state dictionaries | 914 MB |
| [`mit_b1.pth`](https://github.com/Chen-Yang-Liu/Change-Agent/blob/main/Multi_change/model/pretrained/mit_b1.pth) | SegFormer MiT-B1 initializer loaded during encoder construction | 52.2 MB |
| [`vocab.json`](https://github.com/Chen-Yang-Liu/Change-Agent/blob/main/Multi_change/data/LEVIR_MCI/vocab.json) | 468-token caption vocabulary, including special tokens | 6.82 KB |

The Hugging Face page publishes SHA-256
`34c6926342c40fdd6d50b43d50257cb46cb6b61763448de9ee551828a64b3eb9`
for `MCI_model.pth`. It is a PyTorch pickle-bearing file, so it must be obtained
only from the official repository and hash-checked before loading.

As written, encoder construction loads `mit_b1.pth` before the MCI encoder state
is loaded. It may be redundant if `encoder_dict` is complete, but removing that
requirement cannot be considered safe until the checkpoint keys are inspected
in the real test. The prediction import graph also needs the upstream `model/`,
`utils_tool/`, and `eval_func/` source trees. Dataset images, split lists, and
caption tokens are not required for direct prediction.

For context only, the separate
[LEVIR-MCI dataset repository](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main)
is 2.77 GB. It was not downloaded and is not needed for the proposed first run.

### Exact preprocessing

The direct prediction code:

1. Reads each file with `imageio.v2.imread`, which yields RGB ordering for
   ordinary RGB PNG/JPEG inputs.
2. Converts to NumPy `float32` without dividing by 255.
3. Transposes HWC to CHW.
4. Per channel, subtracts
   `[0.39073, 0.38623, 0.32989] × 255` and divides by
   `[0.15329, 0.14628, 0.13648] × 255`.
5. Wraps each array as a `torch.FloatTensor` and adds a batch dimension, yielding
   `(1, 3, 256, 256)` for valid first-test inputs.

The training dataset loader uses the same channel statistics. The predictor's
resize branch is defective: it transposes to CHW and then calls OpenCV resize,
which interprets the dimensions as HWC. The first experiment must use exact
256 × 256 inputs. A later adapter should resize in HWC before transposing and
record the interpolation policy.

There is no validation for channel count, bit depth, alignment, image order, or
matching dimensions upstream. SatQuery must add it.

### Mask and caption semantics

The attentive encoder emits three logits per pixel. `argmax` maps them to:

| Value | Meaning | Upstream training label | Upstream rendered color |
| ---: | --- | ---: | --- |
| 0 | background / neither target class | 0 | black |
| 1 | road change | 128 → 1 | yellow (`[0,255,255]` written as OpenCV BGR) |
| 2 | building change | 255 → 2 | red (`[0,0,255]` written as OpenCV BGR) |

Upstream saves only the rendered three-channel mask from `predict.py`; it does
not create an overlay. Its return-value comment incorrectly says the returned
indexed array has shape `(256,256,3)`.

Captioning is part of the same MCI model. The decoder greedily generates up to
41 token positions, removes `<START>`, `<END>`, and `<NULL>`, and maps IDs through
the bundled 468-token vocabulary. Direct prediction needs the MCI checkpoint
and `vocab.json`, but no reference caption, full dataset, OpenAI API, InternLM,
or other language model. Reference captions and Java-backed METEOR are relevant
to evaluation, not one-pair prediction.

### Inference dependencies

The upstream
[`requirement.txt`](https://github.com/Chen-Yang-Liu/Change-Agent/blob/main/Multi_change/requirement.txt)
documents Python 3.9 and pins CUDA 11.8 builds of PyTorch 2.0.1, torchvision
0.15.2, and torchaudio 2.0.2. It also pins NumPy, OpenCV, MMCV, MMEngine,
MMsegmentation, Pillow, pandas, PyArrow, Transformers, OpenAI, OpenDataLab, and
tqdm.

For direct MCI prediction, source inspection verifies these direct third-party
imports: PyTorch, torchvision, NumPy, OpenCV, MMCV, MMsegmentation, `einops`,
`imageio`, scikit-image, and `timm`. The last four are omitted from upstream's
requirements. Conversely, the direct prediction graph does not import
torchaudio, OpenAI, OpenDataLab, pandas, PyArrow, Transformers, or MMEngine;
they are excluded from this project's [`requirements.in`](../requirements.in).
Transitive packages will still be resolved by the installer.

This inventory has not been installed or locked. In particular, MMCV 1.3.1 and
MMsegmentation 0.13.0 are legacy releases, while current MMCV documentation
notes that Apple Silicon requires PyTorch 1.13 or newer and may require a source
build. The default model uses only `mmcv.runner.load_checkpoint` and
`mmseg.utils.get_root_logger` from those frameworks, so replacing those two
imports with local equivalents may be safer than compiling old MMCV; that must
be demonstrated, not assumed.

### CUDA, CPU, and MPS

Mac compatibility is not supplied upstream. CUDA is hard-coded in all three
execution layers:

- `predict.py` sends the encoder, attentive encoder, decoder, and both input
  tensors through `.cuda()`.
- `model_encoder_att.py` creates positional index tensors with `.cuda()`.
- `model_decoder.py` creates target, mask, start-token, sequence, and beam-search
  tensors with `.cuda()`.
- `torch.load()` is called without `map_location`, which can fail on a machine
  without the device on which the checkpoint tensors were saved.

CPU should be structurally possible after device-neutral changes because the
model is composed mainly of standard convolutions, transposed convolutions,
normalization, attention, embeddings, interpolation, and tensor operations.
That is a code-based assessment, not a successful inference result.

PyTorch's official [MPS backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
shows the required `torch.backends.mps.is_built()` and `is_available()` checks.
The M4/macOS combination is eligible in principle, but individual operators,
memory use, old PyTorch behavior, and numerical parity remain untested. The
environment script will report these checks once PyTorch is installed.

## Licences and usage constraints

These are separate artifacts and must not be conflated:

- **Code:** the repository contains an
  [MIT licence file](https://github.com/Chen-Yang-Liu/Change-Agent/blob/main/LICENSE.txt),
  but its README also says the code “can be used for academic purposes only.”
  That extra statement conflicts with the permission language in MIT. Treat
  commercial use as unresolved and ask the authors or counsel before shipping.
- **Checkpoint:** the Hugging Face model repository declares `apache-2.0` in
  its model-card metadata. It supplies no fuller artifact-specific terms beyond
  the link back to the code repository, so the interaction with the repository's
  academic-only statement is unresolved.
- **Dataset:** the LEVIR-MCI Hugging Face repository has empty/missing YAML
  metadata and exposes no licence file or licence identifier. Its page links
  back to Change-Agent, but that is not an explicit dataset grant. Treat the
  dataset licence as unspecified. The earlier LEVIR-CC repository also does not
  display a licence. Do not redistribute or use dataset samples commercially
  until permission is clarified.

This is an engineering inventory, not legal advice.

## Blockers before calling the model feasible

1. Installability of the minimal dependency set on arm64/Python 3.9 is unknown.
2. CPU execution needs device-neutral patches and one complete forward pass.
3. MPS operator coverage, peak unified-memory use, speed, and output parity are
   unknown.
4. Checkpoint contents and key completeness have not been inspected; doing so
   requires the 914 MB download explicitly deferred from this task.
5. The upstream argument parsing and non-256 resize path must be corrected.
6. A real pair must confirm mask dimensions/classes and intelligible caption
   decoding.
7. Code/checkpoint/dataset usage terms need resolution for any non-research use.

## Proposed first inference experiment

Use one user-supplied or separately licensed, already aligned 256 × 256 RGB PNG
pair—ideally one obvious building or road change and no clouds. Do not download
LEVIR-MCI.

1. In the existing Python 3.9 `.venv`, attempt the minimal dependency install
   and record every resolved version. If legacy MMCV fails, replace only its two
   used helpers rather than broadening the dependency stack.
2. Fetch the official source at a recorded commit, the 914 MB MCI checkpoint,
   the 52.2 MB MiT-B1 file, and the 6.82 KB vocabulary. Verify the published MCI
   SHA-256 before deserialization. Do not fetch `lagent`, InternLM, or the dataset.
3. Add a thin local adapter that has one argument parser, validates RGB 256 ×
   256 inputs, uses `torch.load(..., map_location=device)`, derives every new
   tensor's device from existing tensors, and runs under `torch.inference_mode()`.
4. Run CPU first and save the raw `{0,1,2}` mask, fixed-color mask, post-image
   overlay, caption, and run record. Record wall time and peak memory.
5. If CPU succeeds, run the identical pair on MPS. Compare the exact mask (and,
   if different, class-wise pixel counts), caption, time, and memory. Only enable
   `PYTORCH_ENABLE_MPS_FALLBACK=1` if required, and record that fallback.

The next implementation task should begin with dependency installation and
checkpoint download approval, then execute steps 2–4 on one supplied pair.
