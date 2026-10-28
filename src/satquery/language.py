"""Validated local-language-model planning and explanation for GeoLens."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from satquery.geolens import create_unique_run_directory
from satquery.historical import (
    ConfirmedAOI,
    HistoricalRequest,
    HistoricalValidationError,
    build_request,
    deterministic_summary,
    run_historical,
)


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
MAX_REQUEST_CHARACTERS = 2_000
ANALYSES = ("vegetation", "construction", "roads")
MISSING_FIELDS = (
    "analysis",
    "start_year",
    "end_year",
    "confirmed_boundary",
    "named_location_confirmation",
)

JsonTransport = Callable[[str, Mapping[str, Any] | None, float], dict[str, Any]]
HistoricalRunner = Callable[[HistoricalRequest, Path, Path], dict[str, Any]]


class LanguageModelError(RuntimeError):
    """Base error for local language-model integration."""


class LanguageModelUnavailableError(LanguageModelError):
    """Raised when the configured local model cannot be reached."""


class LanguageModelValidationError(LanguageModelError):
    """Raised when a model response fails the deterministic contract."""


@dataclass(frozen=True)
class ProviderInfo:
    available: bool
    provider: str
    endpoint: str
    model: str
    installed_models: tuple[str, ...]
    reason: str | None = None

    @property
    def identifier(self) -> str:
        return f"{self.provider}/{self.model}"

    def record(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "identifier": self.identifier,
            "installed_models": list(self.installed_models),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InterpretedRequest:
    original_request: str
    analyses: tuple[str, ...]
    start_year: int | None
    end_year: int | None
    uses_confirmed_boundary: bool
    boundary_source: str | None
    boundary_label: str | None
    named_location: str | None
    missing_information: tuple[str, ...]
    evidence: Mapping[str, str | None]
    provider: str
    model: str
    runtime: Mapping[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "original_request": self.original_request,
            "analyses": list(self.analyses),
            "start_year": self.start_year,
            "end_year": self.end_year,
            "boundary_reference": (
                "current_confirmed_map_boundary"
                if self.uses_confirmed_boundary
                else "missing_or_unspecified"
            ),
            "boundary_source": self.boundary_source,
            "boundary_label": self.boundary_label,
            "named_location": self.named_location,
            "missing_information": list(self.missing_information),
            "evidence": dict(self.evidence),
            "provider": self.provider,
            "model": self.model,
            "runtime": dict(self.runtime),
        }


@dataclass(frozen=True)
class ToolPlan:
    interpretation: InterpretedRequest
    status: str
    tools: tuple[Mapping[str, Any], ...]
    historical_request: HistoricalRequest | None
    validation_errors: tuple[str, ...]

    @property
    def runnable(self) -> bool:
        return any(tool.get("status") == "ready" for tool in self.tools)

    def record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "interpretation": self.interpretation.record(),
            "boundary_context": {
                "source": self.interpretation.boundary_source,
                "label": self.interpretation.boundary_label,
            },
            "tools": [dict(tool) for tool in self.tools],
            "validation_errors": list(self.validation_errors),
        }


def _normalise_ollama_url(value: str | None) -> str:
    endpoint = (value or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise LanguageModelValidationError(
            "GeoLens accepts only a loopback Ollama endpoint; remote paid services are not used implicitly"
        )
    return endpoint


def _request_json(
    url: str,
    payload: Mapping[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json", "User-Agent": "GeoLens-local/0.2"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - validated loopback URL.
            value = json.load(response)
    except (OSError, HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LanguageModelUnavailableError(f"local Ollama request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise LanguageModelValidationError("local Ollama returned a non-object JSON response")
    return value


def discover_ollama(
    endpoint: str | None = None,
    model: str | None = None,
    transport: JsonTransport = _request_json,
) -> ProviderInfo:
    """Discover a tested local Ollama model without loading or downloading it."""
    try:
        resolved_endpoint = _normalise_ollama_url(
            endpoint or os.environ.get("GEOLENS_OLLAMA_URL") or os.environ.get("OLLAMA_HOST")
        )
    except LanguageModelValidationError as exc:
        return ProviderInfo(False, "ollama", endpoint or "invalid", model or "", (), str(exc))
    resolved_model = model or os.environ.get("GEOLENS_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    try:
        response = transport(f"{resolved_endpoint}/api/tags", None, 2.0)
        raw_models = response.get("models")
        if not isinstance(raw_models, list):
            raise LanguageModelValidationError("Ollama model-list response has no model array")
        installed = tuple(
            sorted(
                str(item.get("name"))
                for item in raw_models
                if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            )
        )
    except LanguageModelError as exc:
        return ProviderInfo(
            False,
            "ollama",
            resolved_endpoint,
            resolved_model,
            (),
            f"{exc}. Start Ollama with `ollama serve`.",
        )
    if resolved_model not in installed:
        return ProviderInfo(
            False,
            "ollama",
            resolved_endpoint,
            resolved_model,
            installed,
            (
                f"configured model {resolved_model!r} is not installed; install it explicitly with "
                f"`ollama pull {resolved_model}` or set GEOLENS_OLLAMA_MODEL to a tested installed model"
            ),
        )
    return ProviderInfo(True, "ollama", resolved_endpoint, resolved_model, installed)


class OllamaProvider:
    """Small loopback-only Ollama client using JSON-schema structured output."""

    def __init__(
        self,
        info: ProviderInfo,
        transport: JsonTransport = _request_json,
    ) -> None:
        if not info.available:
            raise LanguageModelUnavailableError(info.reason or "local Ollama is unavailable")
        self.info = info
        self._transport = transport

    def structured_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        timeout_seconds: float = 180.0,
        max_tokens: int = 512,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self._transport(
            f"{self.info.endpoint}/api/chat",
            {
                "model": self.info.model,
                "messages": [dict(message) for message in messages],
                "stream": False,
                "think": False,
                "format": dict(schema),
                "options": {"temperature": 0, "num_predict": max_tokens},
                "keep_alive": "5m",
            },
            timeout_seconds,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise LanguageModelValidationError("Ollama response has no textual message content")
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LanguageModelValidationError("Ollama structured response is not valid JSON") from exc
        if not isinstance(value, dict):
            raise LanguageModelValidationError("Ollama structured response is not a JSON object")
        runtime = {
            key: response.get(key)
            for key in (
                "model",
                "done_reason",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "eval_count",
            )
        }
        return value, runtime


INTERPRETATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "vegetation_requested": {"type": "boolean"},
        "construction_requested": {"type": "boolean"},
        "roads_requested": {"type": "boolean"},
        "vegetation_evidence": {"type": ["string", "null"]},
        "construction_evidence": {"type": ["string", "null"]},
        "roads_evidence": {"type": ["string", "null"]},
        "start_year": {"type": ["integer", "null"]},
        "end_year": {"type": ["integer", "null"]},
        "uses_confirmed_boundary": {"type": "boolean"},
        "boundary_evidence": {"type": ["string", "null"]},
        "named_location": {"type": ["string", "null"]},
    },
    "required": [
        "vegetation_requested",
        "construction_requested",
        "roads_requested",
        "vegetation_evidence",
        "construction_evidence",
        "roads_evidence",
        "start_year",
        "end_year",
        "uses_confirmed_boundary",
        "boundary_evidence",
        "named_location",
    ],
}


def _validate_interpretation_payload(value: Mapping[str, Any]) -> None:
    expected = set(INTERPRETATION_SCHEMA["required"])
    if set(value) != expected:
        raise LanguageModelValidationError("interpretation has missing or unexpected fields")
    for name in (
        "vegetation_requested",
        "construction_requested",
        "roads_requested",
        "uses_confirmed_boundary",
    ):
        if not isinstance(value[name], bool):
            raise LanguageModelValidationError(f"interpretation field {name!r} is not boolean")
    for name in ("start_year", "end_year"):
        field = value[name]
        if field is not None and (isinstance(field, bool) or not isinstance(field, int)):
            raise LanguageModelValidationError(f"interpretation field {name!r} is not an integer or null")
    for name in (
        "vegetation_evidence",
        "construction_evidence",
        "roads_evidence",
        "boundary_evidence",
        "named_location",
    ):
        if value[name] is not None and not isinstance(value[name], str):
            raise LanguageModelValidationError(f"interpretation field {name!r} is not a string or null")


def interpret_request(
    provider: OllamaProvider,
    original_request: str,
    aoi_confirmed: bool,
    active_boundary_label: str | None = None,
    active_boundary_source: str | None = None,
) -> InterpretedRequest:
    text = original_request.strip()
    if not text:
        raise LanguageModelValidationError("natural-language request is empty")
    if len(text) > MAX_REQUEST_CHARACTERS:
        raise LanguageModelValidationError(
            f"natural-language request exceeds {MAX_REQUEST_CHARACTERS} characters"
        )
    system = (
        "Extract a GeoLens request literally. Set an analysis boolean true only when the user "
        "explicitly asks for it or an unambiguous synonym. Do not infer construction or roads "
        "from vegetation. The words here, this area, selected area, current area, or map boundary "
        "mean uses_confirmed_boundary=true. For every true analysis and boundary boolean, return the "
        "shortest exact supporting quote from the user in its evidence field. Evidence must be null "
        "when its boolean is false. named_location must be the shortest exact geographic place name "
        "explicitly requested by the user, or null when no different/named place is written. Words "
        "like here and this area are not named locations. Never invent years or coordinates. "
        "Examples: 'vegetation here from 2021 to 2025' means vegetation true, construction false, "
        "roads false. 'construction and roads here from 2020 to 2022' means vegetation false, "
        "construction true, roads true. Return only the required schema."
    )
    value, runtime = provider.structured_chat(
        (
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ),
        INTERPRETATION_SCHEMA,
        max_tokens=192,
    )
    _validate_interpretation_payload(value)
    folded_text = text.casefold()
    evidence: dict[str, str | None] = {}
    for analysis, boolean_field, evidence_field in (
        ("vegetation", "vegetation_requested", "vegetation_evidence"),
        ("construction", "construction_requested", "construction_evidence"),
        ("roads", "roads_requested", "roads_evidence"),
    ):
        supporting_text = value[evidence_field]
        if value[boolean_field]:
            if not isinstance(supporting_text, str) or not supporting_text.strip():
                raise LanguageModelValidationError(f"{analysis} intent has no source-text evidence")
            if supporting_text.strip().casefold() not in folded_text:
                raise LanguageModelValidationError(f"{analysis} evidence is not present in the request")
            evidence[analysis] = supporting_text.strip()
        else:
            if supporting_text is not None:
                raise LanguageModelValidationError(f"unrequested {analysis} intent supplied evidence")
            evidence[analysis] = None
    boundary_evidence = value["boundary_evidence"]
    if value["uses_confirmed_boundary"]:
        if not isinstance(boundary_evidence, str) or not boundary_evidence.strip():
            raise LanguageModelValidationError("boundary reference has no source-text evidence")
        if boundary_evidence.strip().casefold() not in folded_text:
            raise LanguageModelValidationError("boundary evidence is not present in the request")
        evidence["confirmed_boundary"] = boundary_evidence.strip()
    else:
        if boundary_evidence is not None:
            raise LanguageModelValidationError("unspecified boundary supplied evidence")
        evidence["confirmed_boundary"] = None
    named_location = value["named_location"]
    if named_location is not None:
        if not named_location.strip():
            raise LanguageModelValidationError("named location is empty")
        if named_location.strip().casefold() not in folded_text:
            raise LanguageModelValidationError("named location is not present in the request")
        named_location = named_location.strip()
    evidence["named_location"] = named_location
    runtime_record = dict(runtime)
    discarded_years: list[int] = []
    start_year = value["start_year"]
    end_year = value["end_year"]
    if start_year is not None and not re.search(rf"(?<!\d){start_year}(?!\d)", text):
        discarded_years.append(start_year)
        start_year = None
    if end_year is not None and not re.search(rf"(?<!\d){end_year}(?!\d)", text):
        discarded_years.append(end_year)
        end_year = None
    runtime_record["discarded_years_without_source_evidence"] = discarded_years
    runtime_record["model_uses_confirmed_boundary"] = value["uses_confirmed_boundary"]
    analyses = tuple(
        name
        for name, field in (
            ("vegetation", "vegetation_requested"),
            ("construction", "construction_requested"),
            ("roads", "roads_requested"),
        )
        if value[field]
    )
    missing: list[str] = []
    if not analyses:
        missing.append("analysis")
    if start_year is None:
        missing.append("start_year")
    if end_year is None:
        missing.append("end_year")
    resolved_label = active_boundary_label or ("Confirmed map area" if aoi_confirmed else None)
    resolved_source = active_boundary_source or ("application_state" if aoi_confirmed else None)
    location_conflict = False
    if named_location is not None and resolved_label is not None:
        normalised_location = " ".join(re.findall(r"[a-z0-9]+", named_location.casefold()))
        normalised_label = " ".join(re.findall(r"[a-z0-9]+", resolved_label.casefold()))
        location_conflict = normalised_location not in normalised_label
    elif named_location is not None:
        location_conflict = True
    boundary_resolved = aoi_confirmed and not location_conflict
    if not aoi_confirmed:
        missing.append("confirmed_boundary")
    elif location_conflict:
        missing.append("named_location_confirmation")
    return InterpretedRequest(
        text,
        analyses,
        start_year,
        end_year,
        boundary_resolved,
        resolved_source if boundary_resolved else None,
        resolved_label if boundary_resolved else None,
        named_location,
        tuple(missing),
        evidence,
        provider.info.provider,
        provider.info.model,
        runtime_record,
    )


def build_tool_plan(
    interpretation: InterpretedRequest,
    aoi: ConfirmedAOI | None,
    current_year: int | None = None,
) -> ToolPlan:
    tools: list[dict[str, Any]] = []
    errors: list[str] = []
    historical_request: HistoricalRequest | None = None
    for analysis in interpretation.analyses:
        if analysis == "vegetation":
            if interpretation.missing_information:
                tools.append({
                    "tool": "historical_vegetation",
                    "analysis": analysis,
                    "status": "blocked",
                    "reason": (
                        "Clarification required: "
                        + ", ".join(interpretation.missing_information)
                    ),
                    "validated_arguments": None,
                })
                continue
            try:
                historical_request = build_request(
                    aoi,
                    interpretation.start_year,  # type: ignore[arg-type]
                    interpretation.end_year,  # type: ignore[arg-type]
                    analysis="vegetation_ndvi",
                    current_year=current_year,
                )
                tools.append({
                    "tool": "historical_vegetation",
                    "analysis": analysis,
                    "status": "ready",
                    "validated_arguments": historical_request.record(),
                })
            except HistoricalValidationError as exc:
                errors.append(str(exc))
                tools.append({
                    "tool": "historical_vegetation",
                    "analysis": analysis,
                    "status": "blocked",
                    "reason": str(exc),
                    "validated_arguments": None,
                })
        else:
            tools.append({
                "tool": "construction_changes",
                "analysis": analysis,
                "status": "unsupported_in_historical_mode",
                "reason": (
                    f"{analysis.title()} analysis requires suitable aligned before/after 256×256 RGB "
                    "imagery and explicit confirmation in Construction changes mode. The confirmed "
                    "Sentinel-2 map boundary is not silently sent to the MCI model."
                ),
                "validated_arguments": None,
            })
    if not tools:
        errors.append("no analysis was identified")
    ready = any(tool["status"] == "ready" for tool in tools)
    unsupported = any(tool["status"] == "unsupported_in_historical_mode" for tool in tools)
    if ready and unsupported:
        status = "partial_support"
    elif ready:
        status = "ready"
    else:
        status = "blocked"
    return ToolPlan(interpretation, status, tuple(tools), historical_request, tuple(errors))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute_tool_plan(
    plan: ToolPlan,
    output_root: Path,
    historical_output_root: Path,
    cache_root: Path,
    runner: HistoricalRunner = run_historical,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Execute only the predefined validated historical tool and audit the outcome."""
    run_directory = create_unique_run_directory(output_root, "assistant")
    record_path = run_directory / "assistant.json"
    executions: list[dict[str, Any]] = []
    historical_result: dict[str, Any] | None = None
    for tool in plan.tools:
        execution = {
            "tool": tool.get("tool"),
            "analysis": tool.get("analysis"),
            "validated_arguments": tool.get("validated_arguments"),
        }
        if tool.get("status") != "ready":
            execution.update({
                "status": "not_run",
                "reason": tool.get("reason"),
                "result_references": None,
            })
            executions.append(execution)
            continue
        if tool.get("tool") != "historical_vegetation" or plan.historical_request is None:
            execution.update({
                "status": "failed",
                "error": "ready tool has no validated predefined implementation",
                "result_references": None,
            })
            executions.append(execution)
            continue
        try:
            historical_result = runner(plan.historical_request, historical_output_root, cache_root)
            execution.update({
                "status": "success",
                "result_references": {
                    "execution_record": historical_result.get("outputs", {}).get("execution_record"),
                    "summary": historical_result.get("outputs", {}).get("summary"),
                    "available_years": historical_result.get("available_years"),
                    "missing_years": historical_result.get("missing_years"),
                },
            })
        except Exception as exc:  # The audit record must survive a bounded tool failure.
            execution.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "result_references": None,
            })
        executions.append(execution)
    successful = any(item["status"] == "success" for item in executions)
    failed = any(item["status"] == "failed" for item in executions)
    not_run = any(item["status"] == "not_run" for item in executions)
    if successful:
        execution_status = "partial" if failed or not_run else "success"
    elif failed:
        execution_status = "failed"
    else:
        execution_status = "not_run"
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": execution_status,
        "original_request": plan.interpretation.original_request,
        "provider": {
            "name": plan.interpretation.provider,
            "model": plan.interpretation.model,
            "identifier": f"{plan.interpretation.provider}/{plan.interpretation.model}",
        },
        "validated_interpretation": plan.interpretation.record(),
        "tool_plan": plan.record(),
        "tool_executions": executions,
        "final_answer": None,
        "outputs": {"assistant_record": str(record_path.resolve())},
    }
    _write_json(record_path, record)
    return record, historical_result


def compact_historical_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Create the only result surface supplied to answer generation."""
    observations = []
    for item in result["annual_observations"]:
        observations.append({
            "year": item["year"],
            "status": item["status"],
            "acquisition_date": item.get("acquisition_date"),
            "valid_coverage": item.get("valid_coverage"),
            "pixel_population": item.get("per_year_population"),
        })
    adjacent = []
    for item in result["adjacent_differences"]:
        adjacent.append({
            "label": item["label"],
            "status": item["status"],
            "valid_coverage": item.get("valid_coverage"),
            "mean_ndvi_difference": item.get("mean"),
            "threshold": item.get("threshold"),
            "categories": item.get("categories"),
            "reason": item.get("reason"),
        })
    return {
        "requested_years": result["request"]["years"],
        "observations": observations,
        "available_years": result["available_years"],
        "missing_years": result["missing_years"],
        "common_valid_trend": result["common_valid_trend"],
        "adjacent_differences": adjacent,
        "first_to_last_difference": result["first_to_last_difference"],
        "limitations": result["limitations"],
        "outputs": result["outputs"],
    }


def supported_facts(result: Mapping[str, Any]) -> dict[str, str]:
    compact = compact_historical_result(result)
    available = [item for item in compact["observations"] if item["status"] == "available"]
    facts: dict[str, str] = {
        "actual_observations": "Actual acquisitions: "
        + (", ".join(f"{item['year']}: {item['acquisition_date']}" for item in available) or "none"),
        "missing_years": f"Unavailable requested years: {compact['missing_years'] or 'none'}.",
    }
    trend = compact["common_valid_trend"]
    facts["common_coverage"] = (
        f"All-observation common-valid coverage is {trend['coverage']:.2%} "
        f"({trend['pixels']} of {trend.get('aoi_pixels', 0)} AOI pixels); this population contains "
        "pixels valid across every available included observation."
    )
    if trend["sufficient"] and trend["annual_statistics"]:
        values = ", ".join(
            f"{item['year']} mean NDVI {item['mean']:.4f}"
            for item in trend["annual_statistics"]
        )
        facts["common_trend"] = f"On that fixed common-valid population: {values}."
    first_last = compact["first_to_last_difference"]
    if first_last["status"] == "available":
        facts["first_to_last"] = (
            f"The {first_last['before_year']}–{first_last['after_year']} pair-valid population covers "
            f"{first_last['valid_coverage']:.2%} of the AOI and has mean NDVI difference "
            f"{first_last['mean']:+.4f}; this pair population differs from the all-observation population."
        )
    facts["interpretation_limit"] = (
        "Threshold categories describe NDVI increase, decrease, or approximate stability only; "
        "they do not prove vegetation gain or loss, plant health, deforestation, damage, or cause."
    )
    return facts


def _deterministic_answer(result: Mapping[str, Any], reason: str) -> dict[str, Any]:
    facts = supported_facts(result)
    text = "**Deterministic fallback used.** " + reason + "\n\n"
    text += "\n".join(f"- {fact}" for fact in facts.values())
    text += "\n\n" + deterministic_summary(result)
    return {
        "text": text,
        "fallback_used": True,
        "fallback_reason": reason,
        "validated_fact_ids": list(facts),
        "model_runtime": None,
    }


def generate_supported_answer(
    provider: OllamaProvider,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate digit-free prose while injecting every numeric claim from validated facts."""
    compact = compact_historical_result(result)
    facts = supported_facts(result)
    fact_ids = list(facts)
    if not compact["common_valid_trend"]["sufficient"]:
        expected_headline = "insufficient_comparable_coverage"
        expected_interpretation = "insufficient_common_coverage"
    elif compact["missing_years"]:
        expected_headline = "partial_historical_ndvi_comparison"
        expected_interpretation = "missing_years_limit_comparison"
    else:
        expected_headline = "historical_ndvi_comparison"
        expected_interpretation = "cautious_ndvi_only"
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "headline_code": {
                "type": "string",
                "enum": [expected_headline],
            },
            "fact_ids": {
                "type": "array",
                "uniqueItems": True,
                "minItems": len(fact_ids),
                "maxItems": len(fact_ids),
                "items": {"type": "string", "enum": fact_ids},
            },
            "interpretation_code": {
                "type": "string",
                "enum": [expected_interpretation],
            },
        },
        "required": ["headline_code", "fact_ids", "interpretation_code"],
    }
    prompt = (
        "Write a cautious GeoLens explanation from the supplied structured result. Return every "
        "fact ID exactly once and return the schema-constrained headline and interpretation codes. "
        "Do not generate prose or numerical values.\n\n"
        f"Structured result: {json.dumps(compact, separators=(',', ':'))}\n"
        f"Available fact IDs: {json.dumps(fact_ids)}"
    )
    try:
        value, runtime = provider.structured_chat(
            (
                {"role": "system", "content": "You explain validated geospatial tool results only."},
                {"role": "user", "content": prompt},
            ),
            schema,
            max_tokens=384,
        )
        if set(value) != {"headline_code", "fact_ids", "interpretation_code"}:
            raise LanguageModelValidationError("answer has missing or unexpected fields")
        headline_code = value["headline_code"]
        interpretation_code = value["interpretation_code"]
        returned_ids = value["fact_ids"]
        if headline_code != expected_headline or interpretation_code != expected_interpretation:
            raise LanguageModelValidationError("answer selected a conclusion inconsistent with tool results")
        if not isinstance(returned_ids, list) or any(not isinstance(item, str) for item in returned_ids):
            raise LanguageModelValidationError("answer fact IDs must be a string array")
        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(fact_ids):
            raise LanguageModelValidationError("answer omitted or invented a fact ID")
        ordered_facts = [facts[item] for item in returned_ids]
        headlines = {
            "historical_ndvi_comparison": "Historical NDVI comparison",
            "partial_historical_ndvi_comparison": "Partial historical NDVI comparison",
            "insufficient_comparable_coverage": "Historical comparison with insufficient common coverage",
        }
        explanations = {
            "cautious_ndvi_only": (
                "The available observations support a cautious NDVI comparison on the documented "
                "pixel populations, subject to the tool limitations shown below."
            ),
            "missing_years_limit_comparison": (
                "The available observations support only a partial NDVI comparison. Missing requested "
                "years are not filled, bridged, or assigned findings."
            ),
            "insufficient_common_coverage": (
                "A directly comparable trend is withheld because the configured common-valid coverage "
                "requirement was not met."
            ),
        }
        text = f"### {headlines[headline_code]}\n\n"
        text += "\n".join(f"- {fact}" for fact in ordered_facts)
        text += f"\n\n{explanations[interpretation_code]}"
        return {
            "text": text,
            "fallback_used": False,
            "fallback_reason": None,
            "validated_fact_ids": returned_ids,
            "headline_code": headline_code,
            "interpretation_code": interpretation_code,
            "model_runtime": runtime,
        }
    except LanguageModelError as exc:
        return _deterministic_answer(result, f"Language-model answer rejected: {exc}")


def attach_final_answer(
    assistant_record: Mapping[str, Any],
    answer: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(assistant_record)
    updated["final_answer"] = dict(answer)
    path = Path(updated["outputs"]["assistant_record"])
    _write_json(path, updated)
    return updated
