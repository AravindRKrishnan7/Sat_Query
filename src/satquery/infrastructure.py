"""Validated language and answer contract for uploaded infrastructure image pairs."""

from __future__ import annotations

import hashlib
import json
import re
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


BUILDING_TERM = r"(?:house|houses|building|buildings|structure|structures|bungalow|bungalows|villa|villas|warehouse|warehouses)"
ROAD_TERM = r"(?:road|roads|street|streets|path|paths|bypass|bypasses)"
ADDITION_TERM = r"(?:added|appeared|built|constructed|erected)"
REMOVAL_TERM = r"(?:demolished|disappeared|removed|vanished)"
UNSAFE_CAPTION_TERMS = {
    "answer",
    "claim",
    "ignore",
    "instruction",
    "instructions",
    "pretend",
    "report",
}


def _matches_object_action(text: str, object_term: str, action_term: str) -> bool:
    object_before = re.search(
        rf"\b{object_term}\b(?:\s+(?:is|are|was|were|has|have|been)){{0,3}}\s+\b{action_term}\b",
        text,
    )
    action_before = re.search(
        rf"\b{action_term}\b\s+(?:(?:a|an|the|some|new|additional)\s+)?\b{object_term}\b",
        text,
    )
    return bool(object_before or action_before)


def assess_change_caption(raw_caption: str) -> dict[str, Any]:
    """Extract only conservative direction claims from the model caption."""
    normalized = " ".join(raw_caption.casefold().split())
    tokens = set(re.findall(r"[a-z-]+", normalized))
    unsafe_terms = sorted(tokens & UNSAFE_CAPTION_TERMS)
    assessment: dict[str, Any] = {
        "raw_caption": raw_caption,
        "normalized_caption": normalized,
        "safe_for_paraphrase": not unsafe_terms,
        "rejected_terms": unsafe_terms,
        "no_change": False,
        "building": {"mentioned": False, "direction": "unclear", "label": "buildings"},
        "road": {"mentioned": False, "direction": "unclear"},
    }
    if not normalized or unsafe_terms:
        return assessment

    no_change = bool(
        re.search(r"\b(?:no change|nothing changed|nothing has changed)\b", normalized)
        or normalized in {"same", "identical", "the same"}
    )
    building_mentioned = bool(re.search(rf"\b{BUILDING_TERM}\b", normalized))
    road_mentioned = bool(re.search(rf"\b{ROAD_TERM}\b", normalized))
    building_added = building_mentioned and (
        _matches_object_action(normalized, BUILDING_TERM, ADDITION_TERM)
        or bool(re.search(rf"\b(?:new|newly|additional)\s+{BUILDING_TERM}\b", normalized))
    )
    building_removed = building_mentioned and _matches_object_action(
        normalized,
        BUILDING_TERM,
        REMOVAL_TERM,
    )
    road_added = road_mentioned and (
        _matches_object_action(normalized, ROAD_TERM, ADDITION_TERM)
        or bool(re.search(rf"\b(?:new|newly|additional)\s+{ROAD_TERM}\b", normalized))
    )
    road_removed = road_mentioned and _matches_object_action(
        normalized,
        ROAD_TERM,
        REMOVAL_TERM,
    )

    def direction(added: bool, removed: bool) -> str:
        if no_change or (added and removed):
            return "conflicting" if added or removed else "no_change"
        if added:
            return "added"
        if removed:
            return "removed"
        return "unclear"

    assessment["no_change"] = no_change
    assessment["building"] = {
        "mentioned": building_mentioned,
        "direction": direction(building_added, building_removed),
        "label": "houses" if re.search(r"\bhouses?\b", normalized) else "buildings",
    }
    assessment["road"] = {
        "mentioned": road_mentioned,
        "direction": direction(road_added, road_removed),
    }
    return assessment


def _evidence_relationship(mask_present: bool, caption_direction: str) -> str:
    if caption_direction in {"added", "removed"}:
        return "agreement" if mask_present else "contradiction"
    if caption_direction == "no_change":
        return "contradiction" if mask_present else "agreement"
    if caption_direction == "conflicting":
        return "caption_conflicting"
    return "caption_unclear"


def infrastructure_answer_details(
    interpretation: InfrastructureInterpretation,
    analysis_record: Mapping[str, Any],
    before_date: str | None,
    after_date: str | None,
) -> dict[str, Any]:
    """Build ordinary-language prose while retaining an auditable evidence relationship."""
    distribution = analysis_record["model_outputs"]["mask_class_distribution"]
    building_present = int(distribution["2"]["pixels"]) > 0
    road_present = int(distribution["1"]["pixels"]) > 0
    raw_caption = str(analysis_record["model_outputs"]["caption"].get("text") or "")
    caption = assess_change_caption(raw_caption)
    building_caption = caption["building"]
    road_caption = caption["road"]
    relationships = {
        "building": _evidence_relationship(
            building_present,
            building_caption["direction"] if caption["safe_for_paraphrase"] else "unclear",
        ),
        "road": _evidence_relationship(
            road_present,
            road_caption["direction"] if caption["safe_for_paraphrase"] else "unclear",
        ),
    }
    requested = set(interpretation.requested_classes)
    if interpretation.general_change_requested:
        requested = {"building_change", "road_change"}
    if before_date and after_date:
        period = f" between {before_date} and {after_date}"
    elif before_date:
        period = f" after the user-supplied before date {before_date}"
    elif after_date:
        period = f" by the user-supplied after date {after_date}"
    else:
        period = " between the two images"

    sentences: list[str] = []
    building_requested = "building_change" in requested
    road_requested = "road_change" in requested
    building_direction = building_caption["direction"]
    road_direction = road_caption["direction"]

    joint_absence_handled = (
        building_requested and road_requested and not building_present and not road_present
    )
    if joint_absence_handled:
        if caption["no_change"] and caption["safe_for_paraphrase"]:
            sentences.append(f"The model did not identify building or road changes{period}.")
            sentences.append("No red or yellow change highlights were produced.")
            sentences.append("The model-generated description also indicates no change.")
        elif building_direction in {"added", "removed"}:
            verb = "added" if building_direction == "added" else "removed"
            label = building_caption["label"]
            sentences.append(
                f"The model-generated description suggests {label} were {verb}, but no building changes were highlighted."
            )
            sentences.append("No road changes were highlighted.")
        else:
            sentences.append(f"The model did not identify building or road changes{period}.")
            sentences.append("Its generated description does not provide a clear change explanation.")
    elif building_requested and building_present:
        if caption["safe_for_paraphrase"] and building_direction in {"added", "removed"}:
            verb = "added" if building_direction == "added" else "removed"
            sentences.append(
                f"The model suggests that {building_caption['label']} were {verb}{period}."
            )
        elif caption["safe_for_paraphrase"] and building_direction == "no_change":
            sentences.append(
                "The red highlights indicate building changes, but the model-generated description says no change."
            )
        else:
            sentences.append(f"The model identified building changes{period}.")
        if not sentences[-1].startswith("The red highlights"):
            sentences.append("The red highlights show the areas identified as building changes.")
    elif building_requested:
        if caption["safe_for_paraphrase"] and building_direction in {"added", "removed"}:
            verb = "added" if building_direction == "added" else "removed"
            sentences.append(
                f"The model-generated description suggests {building_caption['label']} were {verb}, but no building changes were highlighted."
            )
        else:
            sentences.append(f"The model did not identify building changes{period}.")

    if road_requested and not joint_absence_handled:
        if road_present:
            if caption["safe_for_paraphrase"] and road_direction in {"added", "removed"}:
                verb = "added" if road_direction == "added" else "removed"
                sentences.append(f"The model suggests a road was {verb}{period}, shown by yellow highlights.")
            else:
                sentences.append(
                    "Yellow highlights show predicted road changes, but they do not establish that a road was constructed or removed."
                )
        elif caption["safe_for_paraphrase"] and road_direction in {"added", "removed"}:
            verb = "added" if road_direction == "added" else "removed"
            sentences.append(
                f"The generated description suggests a road was {verb}, but no road changes were highlighted."
            )
        else:
            sentences.append("It did not detect road changes.")

    has_contradiction = "contradiction" in relationships.values() or (
        caption["safe_for_paraphrase"] and caption["no_change"] and (building_present or road_present)
    )
    if interpretation.instance_count_requested:
        sentences.append(
            "This is a model interpretation, and the method cannot reliably count individual new buildings."
        )
    elif interpretation.spatial_description_requested:
        sentences.append(
            "The overlay shows the highlighted areas, but it does not provide a verified named location."
        )
    elif has_contradiction:
        sentences.append(
            "The description and highlighted changes disagree, so the change direction is uncertain."
        )
    elif not caption["safe_for_paraphrase"]:
        sentences.append(
            "This is a model interpretation; the generated description was not used because it was not a clear change statement."
        )
    else:
        sentences.append("This is a model interpretation, not a verified real-world record.")

    # Keep the primary answer compact even when several requested categories need qualification.
    if len(sentences) > 4:
        sentences = sentences[:3] + [sentences[-1]]
    return {
        "text": " ".join(sentences),
        "caption_assessment": caption,
        "evidence_relationship": relationships,
        "mask_presence": {
            "building_change": building_present,
            "road_change": road_present,
        },
        "rules": {
            "caption_scope": "Only conservative building/road addition, removal, or no-change patterns may be paraphrased.",
            "discarded_claim_types": ["counts", "locations", "dates", "causes"],
            "spatial_consistency_inferred": False,
        },
    }


def render_infrastructure_answer(
    interpretation: InfrastructureInterpretation,
    analysis_record: Mapping[str, Any],
    before_date: str | None,
    after_date: str | None,
) -> str:
    """Render checked prose from the mask and conservatively assessed caption."""
    return infrastructure_answer_details(
        interpretation,
        analysis_record,
        before_date,
        after_date,
    )["text"]


def write_infrastructure_audit(
    interpretation: InfrastructureInterpretation,
    analysis_record: Mapping[str, Any],
    before_date: str | None,
    after_date: str | None,
) -> dict[str, Any]:
    """Persist the question, checked answer, model record reference, and limitations."""
    execution_path = Path(analysis_record["outputs"]["execution_record"]).resolve()
    output_directory = execution_path.parent
    answer_details = infrastructure_answer_details(
        interpretation,
        analysis_record,
        before_date,
        after_date,
    )
    answer_text = answer_details["text"]
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
            "source": "deterministic rendering of validated question intent, mask categories, and conservatively assessed raw caption",
        },
        "answer_validation": {
            key: value for key, value in answer_details.items() if key != "text"
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
