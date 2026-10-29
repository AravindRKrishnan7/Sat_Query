"""Pure preparation and result helpers for the local GeoLens interface."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_PIXELS = 25_000_000
ALLOWED_UPLOAD_FORMATS = {"JPEG", "PNG"}


class GeoLensInputError(ValueError):
    """Raised when a local UI input cannot safely satisfy an analysis contract."""


class GeoLensResultError(ValueError):
    """Raised when an analysis result does not satisfy the UI result contract."""


@dataclass(frozen=True)
class PreparedImage:
    role: str
    image: Image.Image
    original_size: tuple[int, int]
    original_format: str
    resized: bool
    original_bytes: int
    original_sha256: str


def prepare_uploaded_image(data: bytes, role: str, resize: bool) -> PreparedImage:
    """Decode one bounded PNG/JPEG and optionally make an explicit 256 px copy."""
    if not data:
        raise GeoLensInputError(f"{role} upload is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise GeoLensInputError(
            f"{role} upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB local limit"
        )
    try:
        with Image.open(io.BytesIO(data)) as opened:
            original_format = str(opened.format or "").upper()
            if original_format not in ALLOWED_UPLOAD_FORMATS:
                raise GeoLensInputError(f"{role} must be a PNG or JPEG image")
            width, height = opened.size
            if width < 1 or height < 1 or width * height > MAX_UPLOAD_PIXELS:
                raise GeoLensInputError(
                    f"{role} dimensions are unsafe for local preview: {width}x{height}"
                )
            opened.load()
            if opened.mode != "RGB":
                raise GeoLensInputError(
                    f"{role} must decode as three-channel RGB; got mode {opened.mode!r}"
                )
            image = opened.copy()
    except GeoLensInputError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise GeoLensInputError(f"{role} upload is not a readable PNG/JPEG: {exc}") from exc

    original_size = image.size
    was_resized = original_size != (256, 256)
    if was_resized and not resize:
        raise GeoLensInputError(
            f"{role} is {original_size[0]}x{original_size[1]}; the model requires exactly 256x256"
        )
    if was_resized:
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
    return PreparedImage(
        role,
        image,
        original_size,
        original_format,
        was_resized,
        len(data),
        hashlib.sha256(data).hexdigest(),
    )


def create_unique_run_directory(root: Path, prefix: str) -> Path:
    """Create a collision-resistant run directory without touching older runs."""
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for _ in range(10):
        candidate = root / f"{prefix}-{timestamp}-{uuid.uuid4().hex[:10]}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique GeoLens run directory")


def save_prepared_pair(
    before: PreparedImage,
    after: PreparedImage,
    run_directory: Path,
) -> tuple[Path, Path]:
    """Save prepared model inputs under unique internal names."""
    if before.image.size != (256, 256) or after.image.size != (256, 256):
        raise GeoLensInputError("prepared construction images must both be exactly 256x256")
    inputs = run_directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    before_path = inputs / f"before-{token}.png"
    after_path = inputs / f"after-{token}.png"
    before.image.save(before_path, format="PNG")
    after.image.save(after_path, format="PNG")
    return before_path, after_path


def validate_result_record(
    record: Mapping[str, Any],
    required_outputs: Sequence[str],
) -> dict[str, Path]:
    """Validate the small result surface consumed by the Streamlit app."""
    if record.get("status") != "success":
        raise GeoLensResultError("analysis did not return a successful execution record")
    outputs = record.get("outputs")
    if not isinstance(outputs, Mapping):
        raise GeoLensResultError("execution record has no output mapping")
    resolved: dict[str, Path] = {}
    for name in required_outputs:
        value = outputs.get(name)
        if not isinstance(value, str) or not value:
            raise GeoLensResultError(f"execution record is missing output {name!r}")
        path = Path(value).resolve()
        if not path.is_file():
            raise GeoLensResultError(f"recorded output does not exist: {path}")
        resolved[name] = path
    return resolved


def load_execution_record(path: Path) -> dict[str, Any]:
    """Read a successful local execution record for integration verification."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeoLensResultError(f"execution record is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GeoLensResultError(f"execution record must be a JSON object: {path}")
    if value.get("status") != "success":
        raise GeoLensResultError(f"execution record is not successful: {path}")
    return value
