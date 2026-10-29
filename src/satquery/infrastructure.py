"""Validated language and answer contract for uploaded infrastructure image pairs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from satquery.language import (
    MAX_REQUEST_CHARACTERS,
    LanguageModelValidationError,
    OllamaProvider,
)


INFRASTRUCTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "building_changes_requested": {"type": "boolean"},
        "road_changes_requested": {"type": "boolean"},
        "general_change_requested": {"type": "boolean"},
        "instance_count_requested": {"type": "boolean"},
        "spatial_description_requested": {"type": "boolean"},
        "building_evidence": {"type": ["string", "null"]},
        "road_evidence": {"type": ["string", "null"]},
        "general_change_evidence": {"type": ["string", "null"]},
        "instance_count_evidence": {"type": ["string", "null"]},
        "spatial_description_evidence": {"type": ["string", "null"]},
    },
    "required": [
        "building_changes_requested",
        "road_changes_requested",
        "general_change_requested",
        "instance_count_requested",
        "spatial_description_requested",
        "building_evidence",
        "road_evidence",
        "general_change_evidence",
        "instance_count_evidence",
        "spatial_description_evidence",
    ],
}


@dataclass(frozen=True)
class InfrastructureInterpretation:
    original_request: str
    requested_classes: tuple[str, ...]
    general_change_requested: bool
    instance_count_requested: bool
    spatial_description_requested: bool
    evidence: Mapping[str, str | None]
    provider: str
    model: str
    runtime: Mapping[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "original_request": self.original_request,
            "requested_classes": list(self.requested_classes),
            "general_change_requested": self.general_change_requested,
            "instance_count_requested": self.instance_count_requested,
            "spatial_description_requested": self.spatial_description_requested,
            "evidence": dict(self.evidence),
            "provider": self.provider,
            "model": self.model,
            "runtime": dict(self.runtime),
        }


def missing_image_message(has_before: bool, has_after: bool) -> str | None:
    """Explain the missing half of a required comparison without inventing it."""
    if has_before and not has_after:
        return "Add the after image to compare changes; GeoLens will not invent a later observation."
    if has_after and not has_before:
        return "Add the before image to compare changes; GeoLens will not invent an earlier observation."
    return None


def _validate_payload(value: Mapping[str, Any]) -> None:
    expected = set(INFRASTRUCTURE_SCHEMA["required"])
    if set(value) != expected:
        raise LanguageModelValidationError(
            "infrastructure interpretation has missing or unexpected fields"
        )
    for name in (
        "building_changes_requested",
        "road_changes_requested",
        "general_change_requested",
        "instance_count_requested",
        "spatial_description_requested",
    ):
        if not isinstance(value[name], bool):
            raise LanguageModelValidationError(f"interpretation field {name!r} is not boolean")
    for name in (
        "building_evidence",
        "road_evidence",
        "general_change_evidence",
        "instance_count_evidence",
        "spatial_description_evidence",
    ):
        if value[name] is not None and not isinstance(value[name], str):
            raise LanguageModelValidationError(
                f"interpretation field {name!r} is not a string or null"
            )


def interpret_infrastructure_request(
    provider: OllamaProvider,
    original_request: str,
) -> InfrastructureInterpretation:
    """Extract only question aspects supported by the uploaded-pair workflow."""
    text = original_request.strip()
    if not text:
        raise LanguageModelValidationError("infrastructure question is empty")
    if len(text) > MAX_REQUEST_CHARACTERS:
        raise LanguageModelValidationError(
            f"infrastructure question exceeds {MAX_REQUEST_CHARACTERS} characters"
        )
    system = (
        "Extract an uploaded-image infrastructure change question literally. Infrastructure means "
        "both building changes and road changes. Set building_changes_requested for buildings, "
        "houses, construction, or infrastructure, including any request about new, added, removed, "
        "or counted buildings. A count request does not replace the relevant class intent: 'How "
        "many new buildings?' sets both building_changes_requested and instance_count_requested. "
        "Set road_changes_requested for roads or "
        "infrastructure. Set general_change_requested only for a generic request such as 'what "
        "changed' or 'compare these images' that names neither class. Set instance_count_requested "
        "when the user asks how many, a count, a number, added/new instances, or removed instances. "
        "Set spatial_description_requested when the user asks where a change occurred or requests "
        "a named/relative location. For every true boolean return the shortest exact supporting "
        "quote from the user in its evidence field. Evidence must be null for false booleans. Do "
        "not infer dates, counts, objects, locations, or findings. Return only the required schema."
    )
    value, runtime = provider.structured_chat(
        (
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ),
        INFRASTRUCTURE_SCHEMA,
        max_tokens=256,
    )
    _validate_payload(value)
    folded = text.casefold()
    evidence: dict[str, str | None] = {}
    fields = (
        ("buildings", "building_changes_requested", "building_evidence"),
        ("roads", "road_changes_requested", "road_evidence"),
        ("general_change", "general_change_requested", "general_change_evidence"),
        ("instance_count", "instance_count_requested", "instance_count_evidence"),
        (
            "spatial_description",
            "spatial_description_requested",
            "spatial_description_evidence",
        ),
    )
    for label, boolean_field, evidence_field in fields:
        source = value[evidence_field]
        if value[boolean_field]:
            if not isinstance(source, str) or not source.strip():
                raise LanguageModelValidationError(f"{label} intent has no source-text evidence")
            source = source.strip()
            if source.casefold() not in folded:
                raise LanguageModelValidationError(
                    f"{label} evidence is not present in the question"
                )
            evidence[label] = source
        else:
            if source is not None:
                raise LanguageModelValidationError(
                    f"unrequested {label} intent supplied evidence"
                )
            evidence[label] = None
    requested_classes = tuple(
        label
        for label, field in (
            ("building_change", "building_changes_requested"),
            ("road_change", "road_changes_requested"),
        )
        if value[field]
    )
    if not requested_classes and not value["general_change_requested"]:
        raise LanguageModelValidationError(
            "the question does not request building, road, infrastructure, or general image change"
        )
    return InfrastructureInterpretation(
        original_request=text,
        requested_classes=requested_classes,
        general_change_requested=value["general_change_requested"],
        instance_count_requested=value["instance_count_requested"],
        spatial_description_requested=value["spatial_description_requested"],
        evidence=evidence,
        provider=provider.info.provider,
        model=provider.info.model,
        runtime=dict(runtime),
    )


def validate_acquisition_dates(
    before_value: str,
    after_value: str,
) -> tuple[str | None, str | None]:
    """Validate optional user-supplied ISO dates without inventing missing values."""
    values: list[str | None] = []
    parsed: list[date | None] = []
    for label, raw in (("before", before_value), ("after", after_value)):
        value = raw.strip()
        if not value:
            values.append(None)
            parsed.append(None)
            continue
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError as exc:
            raise LanguageModelValidationError(
                f"{label} acquisition date must use YYYY-MM-DD"
            ) from exc
        values.append(parsed_value.isoformat())
        parsed.append(parsed_value)
    if parsed[0] is not None and parsed[1] is not None and parsed[0] > parsed[1]:
        raise LanguageModelValidationError(
            "before acquisition date must not be later than after acquisition date"
        )
    return values[0], values[1]


def infrastructure_request_fingerprint(
    question: str,
    before_bytes: bytes | None,
    after_bytes: bytes | None,
    before_date: str,
    after_date: str,
    resize: bool,
) -> str:
    """Identify the inputs that a retained UI result belongs to."""
    digest = hashlib.sha256()
    metadata = json.dumps(
        {
            "question": question.strip(),
            "before_date": before_date.strip(),
            "after_date": after_date.strip(),
            "resize": resize,
        },
        sort_keys=True,
    ).encode("utf-8")
    digest.update(metadata)
    for value in (before_bytes, after_bytes):
        digest.update(len(value or b"").to_bytes(8, "big"))
        digest.update(value or b"")
    return digest.hexdigest()


def render_infrastructure_answer(
    interpretation: InfrastructureInterpretation,
    analysis_record: Mapping[str, Any],
    before_date: str | None,
    after_date: str | None,
) -> str:
    """Render checked prose from the MCI class mask and the user's supported intent."""
    distribution = analysis_record["model_outputs"]["mask_class_distribution"]
    building_pixels = int(distribution["2"]["pixels"])
    road_pixels = int(distribution["1"]["pixels"])
    if before_date and after_date:
        period = f" between the user-supplied dates {before_date} and {after_date}"
    elif before_date:
        period = f" relative to the user-supplied before date {before_date}"
    elif after_date:
        period = f" relative to the user-supplied after date {after_date}"
    else:
        period = ""

    paragraphs: list[str] = []
    if interpretation.instance_count_requested:
        paragraphs.append(
            "GeoLens cannot provide a validated count of new, added, or removed buildings. "
            "The current model marks change pixels; it does not detect and temporally match "
            "individual building instances."
        )

    requested = set(interpretation.requested_classes)
    if interpretation.general_change_requested:
        requested = {"building_change", "road_change"}
    findings: list[str] = []
    if "building_change" in requested:
        findings.append(
            "areas predicted as building change"
            if building_pixels
            else "no pixels predicted as building change"
        )
    if "road_change" in requested:
        findings.append(
            "areas predicted as road change"
            if road_pixels
            else "no pixels predicted as road change"
        )
    if len(findings) == 2:
        finding_text = f"{findings[0]} and {findings[1]}"
    else:
        finding_text = findings[0]
    paragraphs.append(f"For the uploaded aligned pair{period}, the MCI mask shows {finding_text}.")

    if "road_change" in requested and road_pixels:
        paragraphs.append(
            "Road-change pixels indicate a model-predicted visual change in that class; they do "
            "not by themselves establish that a road was newly constructed."
        )
    if interpretation.spatial_description_requested:
        paragraphs.append(
            "The overlay shows where pixels were marked, but this workflow does not turn them into "
            "a verified named location or object-level position."
        )
    paragraphs.append(
        "These are model predictions, not verified real-world observations. The raw caption is a "
        "separate model output and may disagree with the class mask."
    )
    return "\n\n".join(paragraphs)


def write_infrastructure_audit(
    interpretation: InfrastructureInterpretation,
    analysis_record: Mapping[str, Any],
    before_date: str | None,
    after_date: str | None,
) -> dict[str, Any]:
    """Persist the question, checked answer, model record reference, and limitations."""
    execution_path = Path(analysis_record["outputs"]["execution_record"]).resolve()
    output_directory = execution_path.parent
    answer_text = render_infrastructure_answer(
        interpretation,
        analysis_record,
        before_date,
        after_date,
    )
    answer_path = output_directory / "answer.md"
    audit_path = output_directory / "assistant.json"
    answer_path.write_text(answer_text.rstrip() + "\n", encoding="utf-8")
    audit: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "original_request": interpretation.original_request,
        "model_provider": f"{interpretation.provider}/{interpretation.model}",
        "validated_interpretation": interpretation.record(),
        "user_supplied_acquisition_dates": {
            "before": before_date,
            "after": after_date,
            "verified_from_imagery": False,
        },
        "tool_execution": {
            "tool": "change_agent_mci",
            "status": analysis_record.get("status"),
            "analysis_record": str(execution_path),
        },
        "answer": {
            "text": answer_text,
            "source": "deterministic rendering of validated question intent and MCI mask classes",
        },
        "evidence": {
            "before_image": analysis_record["inputs"]["before"]["path"],
            "after_image": analysis_record["inputs"]["after"]["path"],
            "overlay": analysis_record["outputs"]["overlay"],
            "class_mask": analysis_record["outputs"]["color_mask_with_legend"],
            "raw_caption": analysis_record["outputs"]["caption"],
        },
        "limitations": [
            "The mask is a model prediction without a reference-label accuracy measurement.",
            "Mask components are not building instances and are not counted as buildings.",
            "Road-change pixels do not prove road construction.",
            "The caption is a separate unverified model-head output and may disagree with the mask.",
        ],
        "outputs": {
            "answer_text": str(answer_path.resolve()),
            "assistant_record": str(audit_path.resolve()),
            "analysis_record": str(execution_path),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit
