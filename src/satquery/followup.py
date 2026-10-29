"""Grounded, non-executing follow-up answers for completed GeoLens analyses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from satquery.language import (
    MAX_REQUEST_CHARACTERS,
    LanguageModelError,
    LanguageModelValidationError,
    OllamaProvider,
)


FOLLOW_UP_INTENTS = (
    "simplify",
    "road_changes",
    "building_count",
    "tree_count",
    "missing_observation",
    "evidence",
    "interval_summary",
    "cause",
    "vegetation_cover",
    "general_summary",
)

FOLLOW_UP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {"type": "string", "enum": list(FOLLOW_UP_INTENTS)},
        "evidence_quote": {"type": "string"},
        "start_date_or_year": {"type": ["string", "null"]},
        "end_date_or_year": {"type": ["string", "null"]},
    },
    "required": ["intent", "evidence_quote", "start_date_or_year", "end_date_or_year"],
}


@dataclass(frozen=True)
class FollowUpContext:
    workflow: str
    analysis_id: str
    record: Mapping[str, Any]
    base_answer: str
    selected_comparison_id: str | None = None


def _fallback_intent(question: str) -> dict[str, Any]:
    folded = question.casefold()
    if any(term in folded for term in ("how many building", "building count", "count building")):
        intent = "building_count"
    elif any(term in folded for term in ("how many tree", "tree count", "count tree")):
        intent = "tree_count"
    elif "road" in folded:
        intent = "road_changes"
    elif any(term in folded for term in ("missing", "no usable image", "unavailable year")):
        intent = "missing_observation"
    elif any(term in folded for term in ("evidence", "show me", "overlay", "image")):
        intent = "evidence"
    elif any(term in folded for term in ("why", "cause", "caused")):
        intent = "cause"
    elif any(term in folded for term in ("vegetation cover", "vegetation area", "tree area")):
        intent = "vegetation_cover"
    elif any(term in folded for term in ("simpler", "simply", "plain language")):
        intent = "simplify"
    elif "between" in folded or re.search(r"\b20\d{2}\b", folded):
        intent = "interval_summary"
    else:
        intent = "general_summary"
    dates = re.findall(r"\b20\d{2}(?:-\d{2}-\d{2})?\b", question)
    return {
        "intent": intent,
        "evidence_quote": question.strip(),
        "start_date_or_year": dates[0] if dates else None,
        "end_date_or_year": dates[1] if len(dates) > 1 else None,
    }


def _interpret_follow_up(
    question: str,
    provider: OllamaProvider | None,
) -> tuple[dict[str, Any], str, str | None, Mapping[str, Any] | None]:
    deterministic = _fallback_intent(question)
    if provider is None:
        return deterministic, "deterministic_fallback", "Ollama unavailable", None
    system = (
        "Classify a follow-up question about an already completed satellite analysis. Choose only "
        "one supplied intent. Copy the shortest exact supporting quote into evidence_quote. Copy "
        "a date or year only when it occurs literally in the question; otherwise use null. Do not "
        "answer the question, infer findings, invent dates, or request any tool execution."
    )
    try:
        value, runtime = provider.structured_chat(
            (
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ),
            FOLLOW_UP_SCHEMA,
            max_tokens=160,
        )
        if set(value) != set(FOLLOW_UP_SCHEMA["required"]):
            raise LanguageModelValidationError("follow-up interpretation fields are invalid")
        for name in ("start_date_or_year", "end_date_or_year"):
            if value[name] in {"", "null", "none", "None"}:
                value[name] = None
        quote = value["evidence_quote"]
        if not isinstance(quote, str) or not quote.strip() or quote.casefold() not in question.casefold():
            raise LanguageModelValidationError("follow-up evidence is not present in the question")
        for name in ("start_date_or_year", "end_date_or_year"):
            date_value = value[name]
            if date_value is not None and (
                not isinstance(date_value, str) or date_value not in question
            ):
                raise LanguageModelValidationError(
                    "follow-up date is not present in the question"
                )
        protected_intents = {
            "building_count",
            "tree_count",
            "cause",
            "vegetation_cover",
        }
        if deterministic["intent"] in protected_intents and value["intent"] != deterministic["intent"]:
            return (
                deterministic,
                f"ollama/{provider.info.model}+deterministic_safety_override",
                (
                    f"model intent {value['intent']!r} did not preserve explicit "
                    f"{deterministic['intent']!r} wording"
                ),
                runtime,
            )
        return value, f"ollama/{provider.info.model}", None, runtime
    except LanguageModelError as exc:
        return deterministic, "deterministic_fallback", str(exc), None


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selected_infrastructure_comparison(
    context: FollowUpContext,
    interpretation: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    comparisons = list(context.record["comparisons"])
    requested = [
        str(value)
        for value in (
            interpretation.get("start_date_or_year"),
            interpretation.get("end_date_or_year"),
        )
        if value
    ]
    if requested:
        matches = [
            item
            for item in comparisons
            if all(
                value in {item["before_date"], item["after_date"], item["before_date"][:4], item["after_date"][:4]}
                for value in requested
            )
        ]
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, (
                "Those dates do not identify a saved comparison in this analysis. Review the "
                "inputs and run a new analysis explicitly if you want different dates."
            )
        return None, "More than one saved interval matches those dates. Please name both dates."
    if context.selected_comparison_id:
        selected = next(
            (
                item
                for item in comparisons
                if item["id"] == context.selected_comparison_id
            ),
            None,
        )
        if selected is not None:
            return selected, None
    if len(comparisons) == 1:
        return comparisons[0], None
    return None, "Choose a comparison interval, or include both dates in your question."


def _infrastructure_answer(
    context: FollowUpContext,
    interpretation: Mapping[str, Any],
) -> tuple[str, list[str]]:
    intent = interpretation["intent"]
    comparison, clarification = _selected_infrastructure_comparison(
        context, interpretation
    )
    if intent in {"building_count", "cause"}:
        if intent == "building_count":
            return (
                "The saved result may highlight model-predicted building changes, but this method "
                "does not identify or reliably count individual buildings. It cannot answer how "
                "many buildings were added.",
                [],
            )
        return (
            "This analysis does not establish why a change occurred. It contains model predictions "
            "from the supplied images, not verified causes.",
            [],
        )
    if clarification:
        return clarification, []
    assert comparison is not None
    if comparison["status"] != "success":
        return (
            f"The {comparison['before_date']} to {comparison['after_date']} comparison failed: "
            f"{comparison.get('error', 'no result was produced')}.",
            [],
        )
    execution = _load_json(comparison["execution_record"])
    distribution = execution["model_outputs"]["mask_class_distribution"]
    evidence = [
        comparison["overlay"],
        comparison["execution_record"],
        comparison["raw_caption"],
    ]
    if intent == "road_changes":
        if int(distribution["1"]["pixels"]) > 0:
            text = (
                "For the selected interval, the model predicted road-change areas, shown in yellow "
                "on the overlay. That does not by itself prove a road was constructed or removed."
            )
        else:
            text = (
                "For the selected interval, the model did not predict any road-change areas. "
                "That is a model non-detection, not proof that the real scene was unchanged."
            )
        return text, evidence
    if intent == "evidence":
        return (
            "The selected interval's evidence is the exact prepared before and after images, the "
            "prediction overlay, the separate raw caption, and the child execution record. Open "
            "the evidence section below the comparison to inspect or download them.",
            evidence,
        )
    if intent in {"tree_count", "vegetation_cover"}:
        return (
            "This uploaded-image workflow does not provide validated tree counts or vegetation-cover "
            "area. Its supported outputs are building- and road-change predictions for the selected pair.",
            evidence,
        )
    return str(comparison["answer"]), evidence


def _historical_pair(
    context: FollowUpContext,
    interpretation: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    pairs = list(context.record["adjacent_differences"]) + [
        context.record["first_to_last_difference"]
    ]
    requested = [
        str(value)
        for value in (
            interpretation.get("start_date_or_year"),
            interpretation.get("end_date_or_year"),
        )
        if value
    ]
    if requested:
        matches = [
            item
            for item in pairs
            if all(
                value
                in {
                    str(item.get("before_year")),
                    str(item.get("after_year")),
                    str(item.get("before_date")),
                    str(item.get("after_date")),
                }
                for value in requested
            )
        ]
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, (
                "Those dates do not identify a saved vegetation comparison. A different period "
                "requires a reviewed plan and the existing Run action."
            )
        return None, "More than one saved comparison matches. Please include both years."
    if context.selected_comparison_id:
        selected = next(
            (item for item in pairs if item["label"] == context.selected_comparison_id),
            None,
        )
        if selected is not None:
            return selected, None
    return None, "Choose a comparison interval, or include both years in your question."


def _historical_answer(
    context: FollowUpContext,
    interpretation: Mapping[str, Any],
) -> tuple[str, list[str]]:
    intent = interpretation["intent"]
    if intent in {"building_count", "tree_count"}:
        return (
            "This vegetation analysis cannot count buildings or individual trees. NDVI is a "
            "satellite-based indicator related to vegetation greenness, not an object count.",
            [],
        )
    if intent == "vegetation_cover":
        return (
            "This result measures NDVI greenness, not vegetation-covered area. GeoLens has not "
            "validated a land-cover transition method for this analysis.",
            [],
        )
    if intent == "cause":
        return (
            "The saved NDVI result does not establish a cause, plant health, tree removal, or "
            "statistical significance.",
            [],
        )
    if intent == "missing_observation":
        missing = list(context.record.get("missing_years", []))
        if not missing:
            return "Every requested year has a suitable saved observation.", []
        reasons = {
            item["year"]: item.get("reason")
            for item in context.record["annual_observations"]
            if item["status"] != "available"
        }
        years = ", ".join(str(value) for value in missing)
        return (
            f"No usable observation was available for {years}. "
            + " ".join(str(reasons.get(value) or "") for value in missing).strip(),
            [],
        )
    pair, clarification = _historical_pair(context, interpretation)
    if intent == "evidence":
        references = [
            item["outputs"]["true_color_png"]
            for item in context.record["annual_observations"]
            if item["status"] == "available"
        ]
        references.append(context.record["outputs"]["execution_record"])
        return (
            "The saved evidence includes dated true-colour views, NDVI panels, available difference "
            "maps, the confirmed bounds, and the execution record. Open the evidence section to "
            "inspect the exact populations and provenance.",
            references,
        )
    if intent == "interval_summary":
        if clarification:
            return clarification, []
        assert pair is not None
        if pair["status"] != "available":
            return (
                f"That comparison is unavailable: {pair['reason']} No missing year was bridged.",
                [],
            )
        mean = float(pair["mean"])
        direction = "higher" if mean > 0 else ("lower" if mean < 0 else "unchanged on average")
        categories = pair["categories"]
        text = (
            f"Between {pair['before_date']} and {pair['after_date']}, average NDVI was {direction}. "
            "The map contains both local increases and decreases, so the average does not describe "
            "every location. This is greenness-related evidence, not vegetation area or cause."
        )
        refs = [pair["outputs"]["difference_png"]]
        if categories["increased"]["pixels"] and categories["decreased"]["pixels"]:
            return text, refs
        return text, refs
    return context.base_answer, [context.record["outputs"]["execution_record"]]


def answer_follow_up(
    question: str,
    context: FollowUpContext,
    provider: OllamaProvider | None = None,
) -> dict[str, Any]:
    """Answer from saved evidence only; never invokes analysis or retrieval."""
    text = question.strip()
    if not text:
        raise LanguageModelValidationError("follow-up question is empty")
    if len(text) > MAX_REQUEST_CHARACTERS:
        raise LanguageModelValidationError(
            f"follow-up question exceeds {MAX_REQUEST_CHARACTERS} characters"
        )
    interpretation, classifier, fallback_reason, classifier_runtime = _interpret_follow_up(
        text, provider
    )
    if context.workflow == "infrastructure_timeline":
        answer, refs = _infrastructure_answer(context, interpretation)
    elif context.workflow == "historical_vegetation":
        answer, refs = _historical_answer(context, interpretation)
    else:
        raise LanguageModelValidationError(f"unsupported follow-up workflow: {context.workflow}")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": context.analysis_id,
        "workflow": context.workflow,
        "question": text,
        "interpretation": dict(interpretation),
        "classifier": classifier,
        "classifier_runtime": dict(classifier_runtime) if classifier_runtime else None,
        "fallback_reason": fallback_reason,
        "answer": answer,
        "evidence_references": refs,
        "execution_performed": False,
    }
