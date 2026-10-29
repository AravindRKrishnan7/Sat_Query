"""Validated parent workflow for consecutive uploaded-image comparisons."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from satquery.geolens import PreparedImage
from satquery.infrastructure import InfrastructureInterpretation, write_infrastructure_audit


class InfrastructureTimelineError(ValueError):
    """Raised when uploaded timeline inputs or result identity are invalid."""


@dataclass(frozen=True)
class TimelineObservation:
    """One validated and prepared observation, before chronological sorting."""

    upload_order: int
    filename: str
    acquisition_date: date
    prepared: PreparedImage


ComparisonRunner = Callable[[Path, Path, Path], Mapping[str, Any]]


def validate_timeline_observations(
    filenames: Sequence[str],
    prepared_images: Sequence[PreparedImage],
    date_values: Sequence[str],
) -> tuple[TimelineObservation, ...]:
    """Require two to four uniquely dated observations and sort them by date."""
    count = len(prepared_images)
    if not (2 <= count <= 4):
        raise InfrastructureTimelineError("upload between 2 and 4 images")
    if len(filenames) != count or len(date_values) != count:
        raise InfrastructureTimelineError("every uploaded image must have one acquisition date")

    parsed: list[date] = []
    for index, raw in enumerate(date_values, start=1):
        value = raw.strip()
        if not value:
            raise InfrastructureTimelineError(
                f"acquisition date is required for image {index}"
            )
        try:
            parsed.append(date.fromisoformat(value))
        except ValueError as exc:
            raise InfrastructureTimelineError(
                f"acquisition date for image {index} must use YYYY-MM-DD"
            ) from exc
    if len(set(parsed)) != count:
        raise InfrastructureTimelineError("acquisition dates must be unique")

    observations = tuple(
        TimelineObservation(
            upload_order=index,
            filename=Path(filename).name,
            acquisition_date=acquisition_date,
            prepared=prepared,
        )
        for index, (filename, acquisition_date, prepared) in enumerate(
            zip(filenames, parsed, prepared_images, strict=True),
            start=1,
        )
    )
    return tuple(sorted(observations, key=lambda item: item.acquisition_date))


def timeline_request_fingerprint(
    question: str,
    upload_bytes: Sequence[bytes],
    date_values: Sequence[str],
    resize: bool,
    include_endpoint: bool,
) -> str:
    """Identify every input that controls a retained timeline result."""
    digest = hashlib.sha256()
    metadata = json.dumps(
        {
            "question": question.strip(),
            "dates": [value.strip() for value in date_values],
            "resize": resize,
            "include_endpoint": include_endpoint,
            "observation_count": len(upload_bytes),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    for value in upload_bytes:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_observations(
    parent: Path,
    observations: Sequence[TimelineObservation],
    resize_requested: bool,
) -> tuple[list[dict[str, Any]], list[Path]]:
    observation_directory = parent / "observations"
    observation_directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    paths: list[Path] = []
    for sequence, observation in enumerate(observations, start=1):
        path = observation_directory / (
            f"{sequence:02d}-{observation.acquisition_date.isoformat()}.png"
        )
        observation.prepared.image.save(path, format="PNG")
        prepared_sha256 = _sha256(path)
        paths.append(path.resolve())
        records.append(
            {
                "id": f"observation-{sequence:02d}",
                "chronological_order": sequence,
                "upload_order": observation.upload_order,
                "acquisition_date": observation.acquisition_date.isoformat(),
                "acquisition_date_source": "user supplied; not verified from image metadata",
                "original_upload": {
                    "filename": observation.filename,
                    "bytes": observation.prepared.original_bytes,
                    "sha256": observation.prepared.original_sha256,
                    "format": observation.prepared.original_format,
                    "width": observation.prepared.original_size[0],
                    "height": observation.prepared.original_size[1],
                },
                "preparation": {
                    "resize_requested": resize_requested,
                    "resized": observation.prepared.resized,
                    "resampling": (
                        "Pillow Lanczos to 256x256"
                        if observation.prepared.resized
                        else "none; decoded RGB pixels saved as PNG"
                    ),
                    "width": 256,
                    "height": 256,
                    "mode": "RGB",
                },
                "prepared_file": {
                    "path": str(path.resolve()),
                    "sha256": prepared_sha256,
                },
            }
        )
    return records, paths


def _write_parent_record(record: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _render_timeline_answer(comparisons: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "Using the user-supplied dates, GeoLens compared each consecutive pair independently."
    ]
    for comparison in comparisons:
        before = comparison["before_date"]
        after = comparison["after_date"]
        if comparison["kind"] == "endpoint":
            heading = f"Overall endpoint comparison, {before} to {after}"
            qualification = (
                "This is a separate endpoint prediction, not a sum of the intermediate results."
            )
        else:
            heading = f"{before} to {after}"
            qualification = None
        if comparison["status"] == "success":
            lines.append(f"**{heading}:** {comparison['answer']}")
            if qualification:
                lines.append(qualification)
        else:
            lines.append(
                f"**{heading}:** This comparison could not be completed: {comparison['error']}"
            )
    return "\n\n".join(lines)


def execute_infrastructure_timeline(
    *,
    parent_directory: Path,
    question: str,
    fingerprint: str,
    interpretation: InfrastructureInterpretation,
    observations: Sequence[TimelineObservation],
    resize_requested: bool,
    include_endpoint: bool,
    comparison_runner: ComparisonRunner,
) -> dict[str, Any]:
    """Run consecutive comparisons and persist partial progress after every interval."""
    if not (2 <= len(observations) <= 4):
        raise InfrastructureTimelineError("timeline execution requires 2 to 4 observations")
    ordered = tuple(sorted(observations, key=lambda item: item.acquisition_date))
    if len({item.acquisition_date for item in ordered}) != len(ordered):
        raise InfrastructureTimelineError("timeline execution requires unique dates")

    parent = parent_directory.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    timeline_path = parent / "timeline.json"
    answer_path = parent / "timeline-answer.md"
    observation_records, observation_paths = _save_observations(
        parent, ordered, resize_requested
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "question": question,
        "request_fingerprint": fingerprint,
        "validated_interpretation": interpretation.record(),
        "execution_policy": {
            "consecutive_pairs": True,
            "endpoint_requested": bool(include_endpoint and len(ordered) > 2),
            "endpoint_is_independent": True,
            "model_reused_within_process": True,
        },
        "observations": observation_records,
        "comparisons": [],
        "outputs": {
            "timeline_answer": str(answer_path),
            "timeline_record": str(timeline_path),
        },
    }
    _write_parent_record(record, timeline_path)

    pair_specs = [
        (index, index + 1, "consecutive") for index in range(len(ordered) - 1)
    ]
    if include_endpoint and len(ordered) > 2:
        pair_specs.append((0, len(ordered) - 1, "endpoint"))

    comparison_root = parent / "comparisons"
    comparison_root.mkdir(exist_ok=False)
    for sequence, (before_index, after_index, kind) in enumerate(pair_specs, start=1):
        before_record = observation_records[before_index]
        after_record = observation_records[after_index]
        before_date = before_record["acquisition_date"]
        after_date = after_record["acquisition_date"]
        comparison_directory = comparison_root / (
            f"{sequence:02d}-{kind}-{before_date}-to-{after_date}"
        )
        comparison_directory.mkdir()
        comparison: dict[str, Any] = {
            "id": f"comparison-{sequence:02d}",
            "kind": kind,
            "before_observation": before_record["id"],
            "after_observation": after_record["id"],
            "before_date": before_date,
            "after_date": after_date,
            "status": "running",
            "output_directory": str(comparison_directory),
        }
        record["comparisons"].append(comparison)
        _write_parent_record(record, timeline_path)
        try:
            analysis = dict(
                comparison_runner(
                    observation_paths[before_index],
                    observation_paths[after_index],
                    comparison_directory,
                )
            )
            analysis_before = analysis["inputs"]["before"]
            analysis_after = analysis["inputs"]["after"]
            expected_before = before_record["prepared_file"]
            expected_after = after_record["prepared_file"]
            identity_matches = (
                Path(analysis_before["path"]).resolve()
                == Path(expected_before["path"]).resolve()
                and analysis_before["sha256"] == expected_before["sha256"]
                and Path(analysis_after["path"]).resolve()
                == Path(expected_after["path"]).resolve()
                and analysis_after["sha256"] == expected_after["sha256"]
            )
            if not identity_matches:
                raise InfrastructureTimelineError(
                    "comparison execution record does not match its timeline observations"
                )
            audit = write_infrastructure_audit(
                interpretation,
                analysis,
                before_date,
                after_date,
            )
            comparison.update(
                {
                    "status": "success",
                    "input_identity_verified": True,
                    "answer": audit["answer"]["text"],
                    "execution_record": analysis["outputs"]["execution_record"],
                    "assistant_record": audit["outputs"]["assistant_record"],
                    "answer_file": audit["outputs"]["answer_text"],
                    "overlay": analysis["outputs"]["overlay"],
                    "class_mask": analysis["outputs"]["color_mask_with_legend"],
                    "raw_caption": analysis["outputs"]["caption"],
                }
            )
        except Exception as exc:  # preserve completed comparisons and record the failed interval
            comparison.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc) or "comparison failed without an error message",
                }
            )
        answer = _render_timeline_answer(record["comparisons"])
        answer_path.write_text(answer.rstrip() + "\n", encoding="utf-8")
        successful = sum(item["status"] == "success" for item in record["comparisons"])
        failed = sum(item["status"] == "failed" for item in record["comparisons"])
        record["status"] = (
            "partial"
            if successful and failed
            else ("failed" if failed and not successful else "running")
        )
        _write_parent_record(record, timeline_path)

    failed = any(item["status"] == "failed" for item in record["comparisons"])
    successful = any(item["status"] == "success" for item in record["comparisons"])
    record["status"] = (
        "partial" if failed and successful else ("failed" if failed else "success")
    )
    record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    answer_path.write_text(
        _render_timeline_answer(record["comparisons"]).rstrip() + "\n",
        encoding="utf-8",
    )
    _write_parent_record(record, timeline_path)
    return record
