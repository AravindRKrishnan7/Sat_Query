"""Minimal, device-neutral inference adapter for the official Change-Agent MCI model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError


IMAGE_SIZE = (256, 256)
MEAN_255 = np.asarray([0.39073, 0.38623, 0.32989], dtype=np.float32) * 255.0
STD_255 = np.asarray([0.15329, 0.14628, 0.13648], dtype=np.float32) * 255.0
CLASS_NAMES = {0: "background", 1: "road_change", 2: "building_change"}
CLASS_COLORS_RGB = {
    0: (0, 0, 0),
    1: (255, 255, 0),
    2: (255, 0, 0),
}
EXPECTED_MCI_SHA256 = "34c6926342c40fdd6d50b43d50257cb46cb6b61763448de9ee551828a64b3eb9"
EXPECTED_MIT_B1_SHA256 = "26f8b1a980ad08ddda46f82ba5008b2f6322f11939d59313285c4b477bfe60b1"
EXPECTED_VOCAB_SHA256 = "c9323565663797aac437e1ad4cd5f0f0a7da8b08d8d438bcfa70708ddcca8fc5"
CHANGE_AGENT_REVISION = "68cbaa7f388b36e4fc10872f7a2911482d26ae5b"
MCI_URL = "https://huggingface.co/lcybuaa/Change-Agent/resolve/main/MCI_model.pth"
CHANGE_AGENT_URL = "https://github.com/Chen-Yang-Liu/Change-Agent"
SAMPLE_REVISION = "d1505e514c450c3728782ca723e82761e70bafd3"
SAMPLE_URL = "https://github.com/Chen-Yang-Liu/RSICC/tree/d1505e514c450c3728782ca723e82761e70bafd3/Example"
KNOWN_SAMPLE_HASHES = {
    "before": "8e5d586a4d6af2c658591fd2b9b16142fc9d26100f76ba58069f90de623668f7",
    "after": "c92e9b37043c9fec9ef9cbfba6c2b4b4d40f3be2546efd314cfe70a9ed00a3fc",
}
EXPECTED_PATCHED_SOURCE_HASHES = {
    "Multi_change/model/model_encoder_att.py": "c590af28f9fa580c82082a6fa493eabd9f3458307a84336f9c5fd35bca3e840b",
    "Multi_change/model/model_decoder.py": "a47f9b5be8f00b18038cffc9efac73477c6a68c76de1ce0448571dc09fa61aa2",
    "Multi_change/model/segformer.py": "719877d00ca236c2e70cf2f27fc27c9671b1a5a63ca6051065e716d1eade6158",
}


class InputValidationError(ValueError):
    """Raised when an input is not an aligned 256x256 RGB image."""


class ModelSafetyError(RuntimeError):
    """Raised when a checkpoint cannot be restricted to tensor-only loading."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_image(path: Path, role: str) -> Image.Image:
    if not path.is_file():
        raise InputValidationError(f"{role} image does not exist or is not a file: {path}")
    try:
        with Image.open(path) as opened:
            opened.load()
            if opened.size != IMAGE_SIZE:
                raise InputValidationError(
                    f"{role} image must be exactly 256x256; got {opened.size[0]}x{opened.size[1]}: {path}"
                )
            if opened.mode != "RGB":
                raise InputValidationError(
                    f"{role} image must be 8-bit three-channel RGB; got mode {opened.mode!r}: {path}"
                )
            image = opened.copy()
    except InputValidationError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise InputValidationError(f"{role} image is unreadable: {path}: {exc}") from exc

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != (256, 256, 3):
        raise InputValidationError(
            f"{role} image must decode as uint8 HWC (256, 256, 3); got {array.dtype} {array.shape}: {path}"
        )
    return image


def validate_pair(before_path: Path, after_path: Path) -> tuple[Image.Image, Image.Image]:
    before = validate_image(before_path, "before")
    after = validate_image(after_path, "after")
    if before.size != after.size:
        raise InputValidationError("before and after images must have identical dimensions")
    return before, after


def preprocess(image: Image.Image, torch: Any) -> Any:
    array = np.asarray(image, dtype=np.float32)
    array = (array - MEAN_255.reshape(1, 1, 3)) / STD_255.reshape(1, 1, 3)
    chw = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    if mask.shape != (256, 256):
        raise ValueError(f"class mask must have shape (256, 256); got {mask.shape}")
    values = set(np.unique(mask).tolist())
    if not values.issubset(CLASS_NAMES):
        raise ValueError(f"class mask contains unsupported labels: {sorted(values)}")
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    for label, color in CLASS_COLORS_RGB.items():
        rgb[mask == label] = color
    return Image.fromarray(rgb, mode="RGB")


def add_legend(color_mask: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (256, 326), "white")
    canvas.paste(color_mask, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = 264
    for label in (0, 1, 2):
        draw.rectangle((8, y, 24, y + 16), fill=CLASS_COLORS_RGB[label], outline="gray")
        draw.text((30, y + 2), f"{label}: {CLASS_NAMES[label]}", fill="black")
        y += 20
    return canvas


def make_overlay(after: Image.Image, color_mask: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    blended = Image.blend(after, color_mask, alpha)
    result = np.asarray(after).copy()
    changed = mask != 0
    result[changed] = np.asarray(blended)[changed]
    return Image.fromarray(result, mode="RGB")


def make_comparison(
    before: Image.Image,
    after: Image.Image,
    color_mask: Image.Image,
    overlay: Image.Image,
    caption: str,
) -> Image.Image:
    width = 4 * 256
    canvas = Image.new("RGB", (width, 350), "white")
    draw = ImageDraw.Draw(canvas)
    panels = (("Before", before), ("After", after), ("Predicted mask", color_mask), ("Overlay", overlay))
    for index, (label, image) in enumerate(panels):
        x = index * 256
        draw.text((x + 8, 7), label, fill="black")
        canvas.paste(image, (x, 28))
    draw.rectangle((8, 296, 24, 312), fill=CLASS_COLORS_RGB[1], outline="gray")
    draw.text((30, 298), "1 road change", fill="black")
    draw.rectangle((150, 296, 166, 312), fill=CLASS_COLORS_RGB[2], outline="gray")
    draw.text((172, 298), "2 building change", fill="black")
    caption_text = f"Model-generated caption (not a mask-derived fact): {caption or '[empty]'}"
    draw.text((8, 322), textwrap.shorten(caption_text, width=155, placeholder="..."), fill="black")
    return canvas


def process_peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def system_free_memory_percent() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        output = subprocess.run(
            ["memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"free percentage:\s*(\d+)%", output)
    return int(match.group(1)) if match else None


def mps_memory(torch: Any) -> dict[str, int] | None:
    if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
        return None
    result = {"current_allocated_bytes": int(torch.mps.current_allocated_memory())}
    if hasattr(torch.mps, "driver_allocated_memory"):
        result["driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    if hasattr(torch.mps, "recommended_max_memory"):
        result["recommended_max_bytes"] = int(torch.mps.recommended_max_memory())
    return result


def resolve_device(torch: Any, requested: str) -> Any:
    available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not available:
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    selected = "mps" if requested == "auto" and available else requested
    if selected == "auto":
        selected = "cpu"
    return torch.device(selected)


def verify_tensor_checkpoint(torch: Any, checkpoint_path: Path) -> tuple[Mapping[str, Any], list[str]]:
    scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if scanner is None:
        raise ModelSafetyError("this PyTorch build lacks get_unsafe_globals_in_checkpoint")
    try:
        unsafe_globals = list(scanner(checkpoint_path))
    except Exception as exc:
        raise ModelSafetyError(f"checkpoint safety scan failed: {exc}") from exc
    if unsafe_globals:
        raise ModelSafetyError(f"checkpoint declares unsafe globals: {unsafe_globals}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"encoder_dict", "encoder_trans_dict", "decoder_dict"}
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != required:
        raise ModelSafetyError(f"checkpoint root must contain exactly {sorted(required)}")
    for name in sorted(required):
        state = checkpoint[name]
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
        ):
            raise ModelSafetyError(f"{name} must be a string-to-tensor state dictionary")
    return checkpoint, unsafe_globals


def upstream_revision(source_dir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def verify_source_checkout(source_dir: Path) -> tuple[str, dict[str, str]]:
    revision = upstream_revision(source_dir)
    if revision != CHANGE_AGENT_REVISION:
        raise ModelSafetyError(
            f"Change-Agent source must be revision {CHANGE_AGENT_REVISION}; got {revision or 'unknown'}"
        )
    hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_PATCHED_SOURCE_HASHES.items():
        path = source_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"required patched upstream source is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ModelSafetyError(
                f"patched upstream source hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        hashes[relative] = actual
    return revision, hashes


def parameter_inventory(models: Mapping[str, Any]) -> dict[str, Any]:
    """Count active parameters by component and once across the model graph."""
    global_parameters: dict[int, Any] = {}
    owners: dict[int, list[str]] = {}
    components: dict[str, dict[str, int]] = {}
    for name, model in models.items():
        component_parameters = {id(parameter): parameter for parameter in model.parameters()}
        components[name] = {
            "unique_parameter_tensors": len(component_parameters),
            "parameters": sum(parameter.numel() for parameter in component_parameters.values()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in component_parameters.values() if parameter.requires_grad
            ),
        }
        for identity, parameter in component_parameters.items():
            global_parameters[identity] = parameter
            owners.setdefault(identity, []).append(name)

    shared = [names for names in owners.values() if len(names) > 1]
    return {
        "components": components,
        "component_parameter_sum": sum(item["parameters"] for item in components.values()),
        "total_unique_parameters": sum(parameter.numel() for parameter in global_parameters.values()),
        "total_unique_parameter_tensors": len(global_parameters),
        "cross_component_shared_parameter_tensors": len(shared),
        "counting_method": "torch parameter numel; object identity de-duplicated within and across components",
    }


def load_model(source_dir: Path, checkpoint_path: Path, device: Any, torch: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    multi_change = source_dir / "Multi_change"
    vocab_path = multi_change / "data" / "LEVIR_MCI" / "vocab.json"
    aux_path = multi_change / "model" / "pretrained" / "mit_b1.pth"
    for required_path in (vocab_path, aux_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"required upstream artifact is missing: {required_path}")
    vocab_sha256 = sha256_file(vocab_path)
    aux_sha256 = sha256_file(aux_path)
    if vocab_sha256 != EXPECTED_VOCAB_SHA256:
        raise ModelSafetyError(
            f"vocabulary SHA-256 mismatch: expected {EXPECTED_VOCAB_SHA256}, got {vocab_sha256}"
        )
    if aux_sha256 != EXPECTED_MIT_B1_SHA256:
        raise ModelSafetyError(
            f"MiT-B1 SHA-256 mismatch: expected {EXPECTED_MIT_B1_SHA256}, got {aux_sha256}"
        )

    sys.path.insert(0, str(multi_change))
    try:
        from model.model_decoder import DecoderTransformer
        from model.model_encoder_att import AttentiveEncoder, Encoder
    finally:
        sys.path.pop(0)

    word_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if sorted(word_vocab.values()) != list(range(len(word_vocab))):
        raise RuntimeError("caption vocabulary IDs are not contiguous from zero")

    encoder = Encoder("segformer-mit_b1")
    encoder_trans = AttentiveEncoder(
        train_stage=None,
        n_layers=3,
        feature_size=[16, 16, 512],
        heads=8,
        dropout=0.1,
    )
    decoder = DecoderTransformer(
        encoder_dim=512,
        feature_dim=512,
        vocab_size=len(word_vocab),
        max_lengths=41,
        word_vocab=word_vocab,
        n_head=8,
        n_layers=1,
        dropout=0.1,
    )

    checkpoint, unsafe_globals = verify_tensor_checkpoint(torch, checkpoint_path)
    encoder_result = encoder.load_state_dict(checkpoint["encoder_dict"])
    trans_result = encoder_trans.load_state_dict(checkpoint["encoder_trans_dict"], strict=False)
    decoder_result = decoder.load_state_dict(checkpoint["decoder_dict"])
    if encoder_result.missing_keys or encoder_result.unexpected_keys:
        raise RuntimeError(f"encoder checkpoint mismatch: {encoder_result}")
    if trans_result.missing_keys:
        raise RuntimeError(f"attentive encoder has missing checkpoint keys: {trans_result.missing_keys}")
    if decoder_result.missing_keys or decoder_result.unexpected_keys:
        raise RuntimeError(f"decoder checkpoint mismatch: {decoder_result}")

    del checkpoint
    gc.collect()
    models = (encoder.eval().to(device), encoder_trans.eval().to(device), decoder.eval().to(device))
    metadata = {
        "vocabulary_path": str(vocab_path.resolve()),
        "vocabulary_sha256": vocab_sha256,
        "vocabulary_size": len(word_vocab),
        "mit_b1_path": str(aux_path.resolve()),
        "mit_b1_sha256": aux_sha256,
        "checkpoint_unsafe_globals": unsafe_globals,
        "attentive_encoder_unexpected_checkpoint_keys": len(trans_result.unexpected_keys),
        "parameters": parameter_inventory(
            {"encoder": models[0], "attentive_encoder": models[1], "caption_decoder": models[2]}
        ),
        "word_vocab": word_vocab,
    }
    return *models, metadata


def decode_caption(sequence: Sequence[int], vocabulary: Mapping[str, int]) -> str:
    inverse = {index: token for token, index in vocabulary.items()}
    excluded = {vocabulary["<START>"], vocabulary["<END>"], vocabulary["<NULL>"]}
    tokens = [inverse[index] for index in sequence if index not in excluded]
    return " ".join(tokens).strip()


def save_outputs(
    output_dir: Path,
    before: Image.Image,
    after: Image.Image,
    mask: np.ndarray,
    caption: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "indexed_mask": output_dir / "class_mask.png",
        "color_mask_with_legend": output_dir / "color_mask_with_legend.png",
        "overlay": output_dir / "overlay.png",
        "comparison": output_dir / "comparison.png",
        "caption": output_dir / "caption.txt",
        "execution_record": output_dir / "run.json",
    }
    Image.fromarray(mask.astype(np.uint8), mode="L").save(paths["indexed_mask"])
    color_mask = colorize_mask(mask)
    add_legend(color_mask).save(paths["color_mask_with_legend"])
    overlay = make_overlay(after, color_mask, mask)
    overlay.save(paths["overlay"])
    make_comparison(before, after, color_mask, overlay, caption).save(paths["comparison"])
    paths["caption"].write_text(
        "Model-generated caption (unverified; not derived from the predicted mask):\n"
        f"{caption}\n",
        encoding="utf-8",
    )
    return {name: str(path.resolve()) for name, path in paths.items()}


def run(
    args: argparse.Namespace,
    model_loader: Callable[[Path, Path, Any, Any], tuple[Any, Any, Any, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    import torch

    before_path = args.before.resolve()
    after_path = args.after.resolve()
    checkpoint_path = args.checkpoint.resolve()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    device = resolve_device(torch, args.device)
    system_memory_before = system_free_memory_percent()
    process_peak_before = process_peak_rss_bytes()
    mps_memory_before = mps_memory(torch)

    validate_started = time.perf_counter()
    before, after = validate_pair(before_path, after_path)
    validation_seconds = time.perf_counter() - validate_started

    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_MCI_SHA256:
        raise ModelSafetyError(
            f"checkpoint SHA-256 mismatch: expected {EXPECTED_MCI_SHA256}, got {checkpoint_sha}"
        )
    source_revision, source_hashes = verify_source_checkout(source_dir)
    input_hashes = {"before": sha256_file(before_path), "after": sha256_file(after_path)}
    known_sample = input_hashes == KNOWN_SAMPLE_HASHES
    warnings = [
        "The model-generated caption is unverified and is not a statement derived from the predicted mask.",
        "No reference mask was supplied; accuracy has not been measured.",
    ]
    if known_sample:
        warnings.append(
            "The RSICC example is used for an academic experiment; its repository does not provide an explicit dataset licence."
        )

    load_started = time.perf_counter()
    loader = model_loader or load_model
    encoder, encoder_trans, decoder, cached_model_meta = loader(
        source_dir, checkpoint_path, device, torch
    )
    # A cached loader may return the same metadata object for several runs.
    # Work on a shallow copy because the vocabulary is intentionally excluded
    # from the JSON execution record below.
    model_meta = dict(cached_model_meta)
    load_seconds = time.perf_counter() - load_started
    if model_meta["attentive_encoder_unexpected_checkpoint_keys"]:
        warnings.append(
            "Upstream loads encoder_trans_dict with strict=False; "
            f"{model_meta['attentive_encoder_unexpected_checkpoint_keys']} unused checkpoint keys were ignored and no active keys were missing."
        )

    preprocess_started = time.perf_counter()
    image_a = preprocess(before, torch).to(device)
    image_b = preprocess(after, torch).to(device)
    preprocess_seconds = time.perf_counter() - preprocess_started

    inference_started = time.perf_counter()
    with torch.inference_mode():
        feat_a, feat_b = encoder(image_a, image_b)
        caption_a, caption_b, logits = encoder_trans(feat_a, feat_b)
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("segmentation logits contain NaN or infinity")
        sequence = decoder.sample(caption_a, caption_b, k=1)
        if device.type == "mps":
            torch.mps.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    logits_cpu = logits.detach().to("cpu")
    mask = torch.argmax(logits_cpu, dim=1)[0].numpy().astype(np.uint8)
    values = set(np.unique(mask).tolist())
    if mask.shape != (256, 256) or not values.issubset(CLASS_NAMES):
        raise RuntimeError(f"invalid model mask contract: shape={mask.shape}, values={sorted(values)}")
    caption = decode_caption(sequence, model_meta.pop("word_vocab"))

    save_started = time.perf_counter()
    output_paths = save_outputs(output_dir, before, after, mask, caption)
    save_seconds = time.perf_counter() - save_started

    counts = Counter(mask.reshape(-1).tolist())
    total_pixels = int(mask.size)
    distribution = {
        str(label): {
            "name": CLASS_NAMES[label],
            "pixels": int(counts.get(label, 0)),
            "fraction": counts.get(label, 0) / total_pixels,
        }
        for label in (0, 1, 2)
    }
    finished = time.perf_counter()
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "inputs": {
            "before": {"path": str(before_path), "sha256": input_hashes["before"]},
            "after": {"path": str(after_path), "sha256": input_hashes["after"]},
            "width": 256,
            "height": 256,
            "mode": "RGB",
            "alignment": "caller-supplied; not checked or modified",
            "known_official_example": known_sample,
            "example_source": SAMPLE_URL if known_sample else None,
            "example_revision": SAMPLE_REVISION if known_sample else None,
        },
        "artifacts": {
            "change_agent_source_url": CHANGE_AGENT_URL,
            "change_agent_revision_expected": CHANGE_AGENT_REVISION,
            "change_agent_revision_actual": source_revision,
            "patched_source_sha256": source_hashes,
            "checkpoint": {
                "path": str(checkpoint_path),
                "source_url": MCI_URL,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": checkpoint_sha,
                "expected_sha256": EXPECTED_MCI_SHA256,
                "hash_verified": checkpoint_sha == EXPECTED_MCI_SHA256,
                "loading": "torch.load(map_location='cpu', weights_only=True) after unsafe-global scan",
            },
            **model_meta,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "requested_device": args.device,
            "selected_device": str(device),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "timing_seconds": {
                "validation": validation_seconds,
                "model_load": load_seconds,
                "preprocessing": preprocess_seconds,
                "inference": inference_seconds,
                "output_save": save_seconds,
                "total": finished - started,
            },
            "memory": {
                "process_peak_rss_bytes_before_model": process_peak_before,
                "process_peak_rss_bytes_after_run": process_peak_rss_bytes(),
                "process_metric": "OS-reported maximum resident set size for this process; not Python allocation tracking",
                "system_free_percent_before": system_memory_before,
                "system_free_percent_after": system_free_memory_percent(),
                "system_metric": "macOS memory_pressure system-wide free percentage; not process memory",
                "mps_before": mps_memory_before,
                "mps_after": mps_memory(torch),
                "mps_metric": "current allocator snapshots, not peak model memory",
            },
        },
        "preprocessing": {
            "read": "Pillow; input accepted only when decoded mode is RGB and size is exactly 256x256",
            "resize": None,
            "dtype": "float32",
            "layout": "HWC to CHW, then batch dimension",
            "mean_times_255": MEAN_255.tolist(),
            "std_times_255": STD_255.tolist(),
        },
        "model_outputs": {
            "segmentation_logits_finite": True,
            "segmentation_logits_shape": list(logits_cpu.shape),
            "segmentation_logits_min": float(logits_cpu.min().item()),
            "segmentation_logits_max": float(logits_cpu.max().item()),
            "mask_shape": list(mask.shape),
            "mask_allowed_values": [0, 1, 2],
            "mask_present_values": sorted(values),
            "mask_class_distribution": distribution,
            "caption": {
                "text": caption,
                "source": "MCI checkpoint decoder, greedy decoding",
                "relationship_to_mask": "separate model head; not derived from or verified against the mask",
            },
            "accuracy": {
                "measured": False,
                "reason": "No reference mask was supplied to this inference command.",
            },
        },
        "outputs": output_paths,
        "warnings": warnings,
    }
    Path(output_paths["execution_record"]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run real Change-Agent MCI inference on one aligned RGB pair.")
    parser.add_argument("--before", required=True, type=Path, help="aligned 256x256 RGB before image")
    parser.add_argument("--after", required=True, type=Path, help="aligned 256x256 RGB after image")
    parser.add_argument("--checkpoint", required=True, type=Path, help="official MCI_model.pth path")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for generated outputs")
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "auto"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_root / ".cache" / "upstream" / "change-agent",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = run(args)
    except (FileNotFoundError, InputValidationError, ModelSafetyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": record["status"],
        "device": record["runtime"]["selected_device"],
        "caption": record["model_outputs"]["caption"]["text"],
        "mask_class_distribution": record["model_outputs"]["mask_class_distribution"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))
    return 0
