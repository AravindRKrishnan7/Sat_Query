# Change-Agent model feasibility

Investigated against the official
[Change-Agent repository](https://github.com/Chen-Yang-Liu/Change-Agent) and
[model repository](https://huggingface.co/lcybuaa/Change-Agent). The first real
experiment completed on 2026-10-23.

## Outcome

The MCI model is feasible as a CPU research baseline on this Apple M4 machine.
A real checkpoint forward pass completed, generated finite segmentation logits,
saved a three-class indexed mask and visual products, and decoded a caption.
This result uses the actual upstream architecture with small, reproducible
device/dependency patches—not image differencing or synthetic predictions.

MPS was attempted after CPU success but was unavailable before model loading:
PyTorch 2.14.1 reports that MPS is built but
`torch.backends.mps.is_available()` is false. No MPS prediction exists, so mask
agreement and caption parity cannot be calculated.

## Verified local environment

- MacBook Pro `Mac16,1`, Apple M4 (10 logical CPU cores), arm64, 16 GiB unified
  memory; macOS 15.6 (build 24G84).
- Project-local `.venv`: CPython 3.11.17. Python 3.9 was not retained merely
  because upstream used it; current PyTorch 2.14.1 provides a native CPython
  3.11/macOS arm64 wheel.
- Installed direct dependencies: PyTorch 2.14.1, NumPy 2.4.6, Pillow 13.0.0,
  and einops 0.8.2. No global packages were installed.
- At the resumed check: 139.2 GiB disk free and 33% system-wide memory free.
- CPU tensor test passed; CUDA is unavailable; MPS is built but unavailable.

## Upstream entry point and compatibility work

Upstream documents `Multi_change/predict.py` with `--imgA_path`, `--imgB_path`,
and `--mask_save_path`. It invokes the joint mask and caption heads. The
dataset evaluator `Multi_change/test.py` requires the LEVIR-MCI split and is not
needed for a single pair. The `lagent`/InternLM/OpenAI interactive layer is also
not required.

The documented predictor cannot run unchanged:

- nested argument parsers reject the documented outer arguments;
- default paths are developer-specific Windows paths;
- CUDA placement is hard-coded in the predictor, positional embeddings, and
  caption decoder;
- its non-256 resize path transposes to CHW before OpenCV resizing;
- it imports legacy MMCV/MMseg/timm helpers for operations available locally;
- checkpoint loads omit `map_location` and restricted loading.

SatQuery therefore uses one parser, validates exact 256 × 256 RGB inputs,
derives temporary tensor placement from existing tensors, replaces the small
legacy helper surface, and uses restricted CPU-first checkpoint loading. The
tracked `scripts/patch_change_agent.py` accepts only the pinned source revision
and known original/already-patched file hashes, then verifies the resulting
hashes. `scripts/prepare_change_agent.sh` reconstructs the ignored checkout and
assets from a fresh SatQuery checkout.

## Required artifacts and identities

| Artifact | Source identity | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Change-Agent source | revision `68cbaa7f388b36e4fc10872f7a2911482d26ae5b` | sparse checkout | pinned by Git |
| `MCI_model.pth` | official Hugging Face model repository | 913,762,855 | `34c6926342c40fdd6d50b43d50257cb46cb6b61763448de9ee551828a64b3eb9` |
| `mit_b1.pth` | Change-Agent revision above | 54,683,828 | `26f8b1a980ad08ddda46f82ba5008b2f6322f11939d59313285c4b477bfe60b1` |
| `vocab.json` | Change-Agent revision above | 6,980 | `c9323565663797aac437e1ad4cd5f0f0a7da8b08d8d438bcfa70708ddcca8fc5` |
| before example | RSICC revision `d1505e514c450c3728782ca723e82761e70bafd3`, `Example/A/train_000016.png` | 114,622 | `8e5d586a4d6af2c658591fd2b9b16142fc9d26100f76ba58069f90de623668f7` |
| after example | same revision, `Example/B/train_000016.png` | 120,084 | `c92e9b37043c9fec9ef9cbfba6c2b4b4d40f3be2546efd314cfe70a9ed00a3fc` |

Only these model assets and example images were obtained. The full dataset and
all external language models remain absent.

Before loading, PyTorch's unsafe-global scanner reported no unsafe globals in
the MCI checkpoint. It was then loaded with
`torch.load(map_location="cpu", weights_only=True)` and validated as exactly
three string-to-tensor state dictionaries. The older MiT-B1 file cannot be
scanned by that newer ZIP-checkpoint scanner, but restricted
`weights_only=True`, CPU-mapped loading succeeded; it was never unrestricted.

## Preprocessing and outputs

The adapter preserves the official normalization. Each uint8 RGB image is
converted to float32 without dividing by 255, then each channel uses:

- mean: `[0.39073, 0.38623, 0.32989] × 255`
- standard deviation: `[0.15329, 0.14628, 0.13648] × 255`

HWC is transposed to CHW and a batch dimension is added. There is no resize,
registration, reprojection, or alignment check.

The mask values are the upstream classes:

| Value | Meaning | Training value | Display color |
| ---: | --- | ---: | --- |
| 0 | background / neither target class | 0 | black |
| 1 | road change | 128 mapped to 1 | yellow |
| 2 | building change | 255 mapped to 2 | red |

Caption decoding is greedy for at most 41 positions, removes `<START>`,
`<END>`, and `<NULL>`, and uses the bundled 468-token vocabulary. It is a
separate learned head. The caption must not be treated as a textual derivation
or validation of the predicted mask.

## CPU experiment results

Command:

```sh
.venv/bin/python scripts/run_change_agent.py \
  --before data/input/before.png \
  --after data/input/after.png \
  --checkpoint weights/MCI_model.pth \
  --output-dir outputs/change-agent-cpu \
  --device cpu
```

The final run succeeded. Logits had shape `(1, 3, 256, 256)`, were finite, and
ranged from -44.996246337890625 to 67.92454528808594. The indexed mask had
shape `(256, 256)`, mode `L`, and only values `{0, 2}`:

| Class | Pixels | Fraction |
| --- | ---: | ---: |
| background (0) | 59,027 | 90.0681% |
| road change (1) | 0 | 0% |
| building change (2) | 6,509 | 9.9319% |

Model-generated caption: **“some houses are built along the road”**. This is an
unverified caption-head claim, not a mask-derived fact.

Visual inspection found that the predicted red regions broadly cover the row
of houses present in the after image. This is a qualitative inspection only.
The example has no matching reference mask, so no IoU, F1, accuracy, or other
quality metric was calculated.

The repeated CPU run produced the same mask distribution, caption, and indexed
mask SHA-256 (`120e179042b35bd84b6f999a8006d145ca16e2add118ceebe47ff7a5112aec18`).

### Parameter counts

Counts use `Parameter.numel()` and de-duplicate by Python object identity both
within each component and across components:

| Active component | Parameter tensors | Parameters |
| --- | ---: | ---: |
| image encoder | 176 | 13,151,424 |
| attentive/change encoder | 750 | 109,686,659 |
| caption decoder | 42 | 7,664,596 |
| **unique total** | **968** | **130,502,679** |

There are zero parameter tensors shared across the three top-level components,
so the component sum equals the unique total. The checkpoint contains 942
additional attentive-encoder keys that upstream ignores with `strict=False`;
there were no missing active keys. Those unused checkpoint entries are not
counted as model parameters.

### Timing and memory

Final-run wall times were 1.123 s model load, 1.117 s inference, 0.018 s output
save, and 3.443 s end-to-end. These are single-run observations, not a
benchmark.

The OS-reported process maximum resident set size after the run was
1,672,019,968 bytes (about 1.56 GiB), up from 202,227,712 bytes before model
loading. This is process peak RSS, not Python allocation tracking and not an
incremental model-only measurement. macOS `memory_pressure` reported 33% free
before and 35% after; that is system-wide state. No MPS allocation measurement
exists because MPS was unavailable.

PyTorch emitted a deprecation warning about mixed key-padding and attention-mask
types in the upstream decoder. It did not stop inference but should be resolved
before treating newer PyTorch versions as validated.

## MPS attempt

The same command with `--output-dir outputs/change-agent-mps --device mps`
exited with status 2 and:

```text
error: MPS was requested but torch.backends.mps.is_available() is false
```

Because failure occurred before model loading, no MPS mask or caption was
created. CPU/MPS pixel agreement and caption differences are therefore not
available; no CPU fallback was silently enabled.

## Dependencies

The patched inference graph directly requires PyTorch, NumPy, Pillow, and
einops. The tested versions are pinned in `requirements.in`. OpenCV, imageio,
scikit-image, torchvision, MMCV, MMsegmentation, timm, torchaudio, lagent,
Transformers, and an external LLM are not imported by this path.

## Licences and unresolved permissions

- **Code:** the Change-Agent repository includes an MIT licence file, while its
  README additionally says the code is for academic purposes only. Those terms
  conflict; commercial use remains unresolved.
- **Checkpoint:** the Hugging Face model card declares Apache-2.0, but also
  points back to the project. Interaction with the repository's academic-only
  statement remains unresolved.
- **Example/dataset:** the RSICC example and LEVIR-MCI repositories expose no
  explicit dataset licence. The two images are retained only in ignored local
  storage for this academic experiment and must not be redistributed without
  permission.

This is an engineering inventory, not legal advice.

## Remaining blockers

1. MPS availability and operator compatibility have not been demonstrated.
2. Accuracy is unmeasured without a licensed matching reference mask and
   verified class mapping.
3. The upstream decoder warning should be corrected and regression-tested for
   future PyTorch releases.
4. Code, checkpoint, and example-image permissions need clarification before
   commercial use or redistribution.
