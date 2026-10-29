"""Display-only image composition with strict saved-input identity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


class PresentationIdentityError(ValueError):
    """Raised when a display asset does not belong to the selected result."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison_input_paths(
    timeline: Mapping[str, Any], comparison_id: str
) -> tuple[Path, Path]:
    """Resolve and verify the exact prepared inputs for one timeline interval."""
    observations = {item["id"]: item for item in timeline["observations"]}
    comparison = next(
        (item for item in timeline["comparisons"] if item["id"] == comparison_id),
        None,
    )
    if comparison is None:
        raise PresentationIdentityError(f"unknown comparison id: {comparison_id}")
    paths: list[Path] = []
    for role in ("before_observation", "after_observation"):
        observation = observations.get(comparison[role])
        if observation is None:
            raise PresentationIdentityError(
                f"comparison references missing observation: {comparison[role]}"
            )
        prepared = observation["prepared_file"]
        path = Path(prepared["path"]).resolve()
        if not path.is_file():
            raise PresentationIdentityError(f"prepared comparison input is missing: {path}")
        if _sha256(path) != prepared["sha256"]:
            raise PresentationIdentityError(
                f"prepared comparison input hash mismatch: {path}"
            )
        paths.append(path)
    return paths[0], paths[1]


def compose_before_after_reveal(
    before_path: Path,
    after_path: Path,
    before_percent: int,
) -> Image.Image:
    """Create a display-only vertical wipe without resampling either source."""
    if not 0 <= before_percent <= 100:
        raise ValueError("before reveal percentage must be between 0 and 100")
    with Image.open(before_path) as before_source, Image.open(after_path) as after_source:
        before = before_source.convert("RGB")
        after = after_source.convert("RGB")
    if before.size != after.size:
        raise PresentationIdentityError(
            f"comparison images have different dimensions: {before.size} and {after.size}"
        )
    split = round(before.width * before_percent / 100)
    revealed = after.copy()
    if split:
        revealed.paste(before.crop((0, 0, split, before.height)), (0, 0))
    return revealed
