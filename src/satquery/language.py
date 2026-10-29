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
    run_historical,
)


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
MAX_REQUEST_CHARACTERS = 2_000
ANALYSES = (
    "vegetation",
    "vegetation_cover",
    "tree_count",
    "construction",
    "roads",
)
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
                "explicit_current_map_reference"
                if self.uses_confirmed_boundary
                else "not_explicit_in_source_text"
            ),
            "named_location": self.named_location,
            "missing_information": list(self.missing_information),
            "evidence": dict(self.evidence),
            "provider": self.provider,
            "model": self.model,
            "runtime": dict(self.runtime),
        }


@dataclass(frozen=True)
class BoundaryResolution:
    confirmed: bool
    source: str | None
    active_aoi_source: str | None
    label: str | None
    bounds_wgs84: tuple[float, float, float, float] | None
    named_location_conflict: bool
    missing_information: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        return {
            "confirmed": self.confirmed,
            "source": self.source,
            "active_aoi_source": self.active_aoi_source,
            "label": self.label,
            "bounds_wgs84": list(self.bounds_wgs84) if self.bounds_wgs84 else None,
            "named_location_conflict": self.named_location_conflict,
            "missing_information": list(self.missing_information),
        }


@dataclass(frozen=True)
class ToolPlan:
    interpretation: InterpretedRequest
    boundary_resolution: BoundaryResolution
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
            "raw_model_interpretation": self.interpretation.record(),
            "interpretation": self.interpretation.record(),
            "boundary_resolution": self.boundary_resolution.record(),
            "boundary_context": self.boundary_resolution.record(),
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
        "vegetation_cover_requested": {"type": "boolean"},
        "tree_count_requested": {"type": "boolean"},
        "construction_requested": {"type": "boolean"},
        "roads_requested": {"type": "boolean"},
        "vegetation_evidence": {"type": ["string", "null"]},
        "vegetation_cover_evidence": {"type": ["string", "null"]},
        "tree_count_evidence": {"type": ["string", "null"]},
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
        "vegetation_cover_requested",
        "tree_count_requested",
        "construction_requested",
        "roads_requested",
        "vegetation_evidence",
        "vegetation_cover_evidence",
        "tree_count_evidence",
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
        "vegetation_cover_requested",
        "tree_count_requested",
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
        "vegetation_cover_evidence",
        "tree_count_evidence",
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
        "explicitly asks for it or an unambiguous synonym. Treat infrastructure as BOTH construction "
        "(buildings) and roads, using the exact word infrastructure as evidence for both. Treat "
        "buildings as construction. Keep vegetation greenness/NDVI, vegetation cover/extent, and "
        "individual-tree counts distinct: generic vegetation, greenness, or NDVI means "
        "vegetation_requested; vegetation cover, vegetation extent, or land-cover area means only "
        "vegetation_cover_requested unless greenness is also explicit; count trees, tree count, or "
        "individual trees means tree_count_requested. Do not infer any of these from another. "
        "The words here, this area, selected area, current area, or map boundary "
        "mean uses_confirmed_boundary=true. For every true analysis and boundary boolean, return the "
        "shortest exact supporting quote from the user in its evidence field. Evidence must be null "
        "when its boolean is false. named_location must be the shortest exact geographic place name "
        "explicitly requested by the user, or null when no different/named place is written. Words "
        "like here and this area are not named locations. Never invent years or coordinates. "
        "Examples: 'vegetation here from 2021 to 2025' means vegetation true and cover/tree false. "
        "'infrastructure from 2023 to 2025' means construction true and roads true. "
        "'vegetation cover and tree count' means cover true and tree count true, not greenness. "
        "Return only the required schema."
    )
    value, runtime = provider.structured_chat(
        (
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ),
        INTERPRETATION_SCHEMA,
        max_tokens=320,
    )
    _validate_interpretation_payload(value)
    folded_text = text.casefold()
    evidence: dict[str, str | None] = {}
    for analysis, boolean_field, evidence_field in (
        ("vegetation", "vegetation_requested", "vegetation_evidence"),
        ("vegetation_cover", "vegetation_cover_requested", "vegetation_cover_evidence"),
        ("tree_count", "tree_count_requested", "tree_count_evidence"),
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
    analyses = tuple(
        name
        for name, field in (
            ("vegetation", "vegetation_requested"),
            ("vegetation_cover", "vegetation_cover_requested"),
            ("tree_count", "tree_count_requested"),
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
    return InterpretedRequest(
        text,
        analyses,
        start_year,
        end_year,
        value["uses_confirmed_boundary"],
        named_location,
        tuple(missing),
        evidence,
        provider.info.provider,
        provider.info.model,
        runtime_record,
    )


def resolve_boundary_context(
    interpretation: InterpretedRequest,
    aoi: ConfirmedAOI | None,
    active_boundary_source: str | None = None,
    active_boundary_label: str | None = None,
) -> BoundaryResolution:
    """Resolve the validated model request against authoritative application state."""
    missing = list(interpretation.missing_information)
    label = active_boundary_label or ("Confirmed map area" if aoi is not None else None)
    named_location = interpretation.named_location
    location_conflict = False
    if named_location is not None and label is not None:
        normalised_location = " ".join(re.findall(r"[a-z0-9]+", named_location.casefold()))
        normalised_label = " ".join(re.findall(r"[a-z0-9]+", label.casefold()))
        location_conflict = normalised_location not in normalised_label
    elif named_location is not None:
        location_conflict = True
    if aoi is None:
        missing.append("confirmed_boundary")
    elif location_conflict:
        missing.append("named_location_confirmation")
    confirmed = aoi is not None and not location_conflict
    return BoundaryResolution(
        confirmed=confirmed,
        source="application_state" if confirmed else None,
        active_aoi_source=(active_boundary_source or "confirmed_map") if confirmed else None,
        label=label if confirmed else None,
        bounds_wgs84=aoi.bounds if confirmed else None,
        named_location_conflict=location_conflict,
        missing_information=tuple(missing),
    )


def _unsupported_analysis_details(analysis: str) -> tuple[str, str, str]:
    if analysis == "construction":
        return (
            "construction_changes",
            "building_change",
            "Building-change analysis cannot run for this confirmed historical area and period: "
            "no suitably dated, aligned high-resolution RGB pair is configured. The available "
            "10 m Sentinel-2 observations cannot validate individual buildings. MCI can localise "
            "class-change pixels on a suitable image pair, but it does not count or temporally "
            "match building instances.",
        )
    if analysis == "roads":
        return (
            "construction_changes",
            "road_change",
            "Road-change analysis cannot run for this confirmed historical area and period: no "
            "suitably dated, aligned high-resolution RGB pair is configured. The available 10 m "
            "Sentinel-2 observations are not sent to MCI, and a road-change mask would not by itself "
            "prove that a new road was constructed.",
        )
    if analysis == "vegetation_cover":
        return (
            "historical_vegetation_cover",
            "vegetation_cover",
            "Vegetation-cover analysis cannot run because GeoLens has no locally validated "
            "land-cover classifier and transition policy for this area. NDVI greenness is not "
            "substituted for cover, and Sentinel-2 SCL quality classes are not treated as a "
            "validated vegetation-extent map.",
        )
    if analysis == "tree_count":
        return (
            "historical_tree_count",
            "individual_tree_count",
            "Individual-tree counting cannot run because no suitably resolved, dated imagery and "
            "locally evaluated tree detector are configured for this area. NDVI and green pixels "
            "do not provide tree counts.",
        )
    raise LanguageModelValidationError(f"unsupported requested analysis: {analysis}")


def _unavailable_observation(
    analysis: str,
    interpretation: InterpretedRequest,
    reason: str,
) -> dict[str, Any]:
    _tool, change_type, _default_reason = _unsupported_analysis_details(analysis)
    years = [
        year
        for year in (interpretation.start_year, interpretation.end_year)
        if year is not None
    ]
    return {
        "change_type": change_type,
        "requested_years": years,
        "actual_dates": [],
        "count": None,
        "area_square_metres": None,
        "spatial_location": None,
        "evidence_image_refs": [],
        "method": None,
        "status": "not_run",
        "limitations": [reason],
    }


def build_tool_plan(
    interpretation: InterpretedRequest,
    aoi: ConfirmedAOI | None,
    current_year: int | None = None,
    active_boundary_source: str | None = None,
    active_boundary_label: str | None = None,
) -> ToolPlan:
    boundary_resolution = resolve_boundary_context(
        interpretation,
        aoi,
        active_boundary_source=active_boundary_source,
        active_boundary_label=active_boundary_label,
    )
    tools: list[dict[str, Any]] = []
    errors: list[str] = []
    historical_request: HistoricalRequest | None = None
    for analysis in interpretation.analyses:
        if analysis == "vegetation":
            if boundary_resolution.missing_information:
                tools.append({
                    "tool": "historical_vegetation",
                    "analysis": analysis,
                    "status": "blocked",
                    "reason": (
                        "Clarification required: "
                        + ", ".join(boundary_resolution.missing_information)
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
                    "observation_contract": {
                        "change_type": "vegetation_greenness_ndvi",
                        "count": None,
                        "area_square_metres": None,
                        "spatial_location": None,
                        "method": "Sentinel-2 Level-2A NDVI difference on validated common pixels",
                    },
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
            tool_name, _change_type, reason = _unsupported_analysis_details(analysis)
            if boundary_resolution.missing_information:
                reason = (
                    "Clarification required: "
                    + ", ".join(boundary_resolution.missing_information)
                    + ". "
                    + reason
                )
            tools.append({
                "tool": tool_name,
                "analysis": analysis,
                "status": "unsupported_in_historical_mode",
                "reason": reason,
                "validated_arguments": None,
                "structured_observation": _unavailable_observation(
                    analysis,
                    interpretation,
                    reason,
                ),
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
    return ToolPlan(
        interpretation,
        boundary_resolution,
        status,
        tuple(tools),
        historical_request,
        tuple(errors),
    )


def summarize_tool_plan(plan: ToolPlan) -> str:
    """Render a short deterministic explanation of a validated plan."""
    labels = {
        "vegetation": "vegetation greenness (NDVI)",
        "vegetation_cover": "vegetation cover",
        "tree_count": "individual-tree counts",
        "construction": "building changes",
        "roads": "road changes",
    }
    requested = ", ".join(labels[item] for item in plan.interpretation.analyses) or "no analysis"
    years = [plan.interpretation.start_year, plan.interpretation.end_year]
    period = (
        f"{years[0]}–{years[1]}"
        if all(year is not None for year in years)
        else "an unresolved period"
    )
    area = plan.boundary_resolution.label or "an unconfirmed area"
    opening = f"GeoLens understood the request as {requested} for {period}, using {area}."
    ready = [tool for tool in plan.tools if tool.get("status") == "ready"]
    unavailable = [tool for tool in plan.tools if tool.get("status") != "ready"]
    paragraphs = [opening]
    if ready:
        paragraphs.append(
            "The supported vegetation-greenness tool can run after you review the plan and choose Run."
        )
    if unavailable:
        paragraphs.extend(str(tool.get("reason")) for tool in unavailable)
    return "\n\n".join(paragraphs)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def structured_historical_observations(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe only supported NDVI comparisons using a stable observation contract."""
    observations: list[dict[str, Any]] = []
    limitations = [
        "NDVI is a vegetation-greenness indicator, not vegetation cover, plant health, tree count, or cause."
    ]
    for item in result.get("adjacent_differences", []):
        if not isinstance(item, Mapping) or item.get("status") != "available":
            continue
        outputs = item.get("outputs")
        difference_path = outputs.get("difference_png") if isinstance(outputs, Mapping) else None
        observations.append({
            "change_type": "vegetation_greenness_ndvi",
            "comparison": item.get("label"),
            "requested_years": [item.get("before_year"), item.get("after_year")],
            "actual_dates": [item.get("before_date"), item.get("after_date")],
            "count": None,
            "area_square_metres": None,
            "spatial_location": None,
            "evidence_image_refs": [difference_path] if difference_path else [],
            "method": "Sentinel-2 Level-2A NDVI difference on pixels valid on both dates",
            "status": "available",
            "metrics": {
                "mean_ndvi_difference": item.get("mean"),
                "valid_pixels": item.get("valid_pixels"),
                "valid_coverage": item.get("valid_coverage"),
                "threshold": item.get("threshold"),
                "categories": item.get("categories"),
            },
            "limitations": limitations,
        })
    return observations


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
                "structured_observations": [tool["structured_observation"]]
                if tool.get("structured_observation")
                else [],
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
            observations = structured_historical_observations(historical_result)
            execution.update({
                "status": "success",
                "result_references": {
                    "execution_record": historical_result.get("outputs", {}).get("execution_record"),
                    "summary": historical_result.get("outputs", {}).get("summary"),
                    "available_years": historical_result.get("available_years"),
                    "missing_years": historical_result.get("missing_years"),
                },
                "structured_observations": observations,
            })
        except Exception as exc:  # The audit record must survive a bounded tool failure.
            error = f"{type(exc).__name__}: {exc}"
            execution.update({
                "status": "failed",
                "error": error,
                "result_references": None,
                "structured_observations": [{
                    "change_type": "vegetation_greenness_ndvi",
                    "requested_years": list(plan.historical_request.years),
                    "actual_dates": [],
                    "count": None,
                    "area_square_metres": None,
                    "spatial_location": None,
                    "evidence_image_refs": [],
                    "method": "Sentinel-2 Level-2A NDVI comparison",
                    "status": "failed",
                    "limitations": [error],
                }],
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
        "raw_model_interpretation": plan.interpretation.record(),
        "validated_interpretation": plan.interpretation.record(),
        "resolved_request_context": plan.boundary_resolution.record(),
        "tool_plan": plan.record(),
        "tool_executions": executions,
        "structured_observations": [
            observation
            for execution in executions
            for observation in execution.get("structured_observations", [])
        ],
        "final_answer": None,
        "outputs": {"assistant_record": str(record_path.resolve())},
    }
    _write_json(record_path, record)
    return record, historical_result


def _execution_issues(execution_context: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not execution_context:
        return issues
    executions = execution_context.get("tool_executions")
    if not isinstance(executions, list):
        return issues
    for index, item in enumerate(executions):
        if not isinstance(item, Mapping) or item.get("status") == "success":
            continue
        issues.append({
            "id": f"execution_issue:{item.get('analysis', 'unknown')}:{index}",
            "analysis": item.get("analysis"),
            "status": item.get("status"),
            "reason": item.get("reason") or item.get("error"),
        })
    return issues


def compact_historical_result(
    result: Mapping[str, Any],
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the complete, bounded evidence surface supplied to answer generation."""
    observations = []
    for item in result["annual_observations"]:
        observations.append({
            "year": item["year"],
            "status": item["status"],
            "acquisition_date": item.get("acquisition_date"),
            "valid_coverage": item.get("valid_coverage"),
            "year_valid_ndvi": item.get("per_year_ndvi"),
            "pixel_population": item.get("per_year_population"),
            "unavailable_reason": item.get("reason"),
        })
    adjacent = []
    for item in result["adjacent_differences"]:
        adjacent.append({
            "label": item["label"],
            "status": item["status"],
            "before_year": item.get("before_year"),
            "after_year": item.get("after_year"),
            "before_date": item.get("before_date"),
            "after_date": item.get("after_date"),
            "valid_pixels": item.get("valid_pixels"),
            "aoi_pixels": item.get("aoi_pixels"),
            "valid_coverage": item.get("valid_coverage"),
            "mean_ndvi_difference": item.get("mean"),
            "threshold": item.get("threshold"),
            "categories": item.get("categories"),
            "population": "pixels valid on both dates in this comparison",
            "reason": item.get("reason"),
        })
    return {
        "requested_years": result["request"]["years"],
        "available_years": result["available_years"],
        "missing_years": result["missing_years"],
        "observations": observations,
        "common_valid_trend": result["common_valid_trend"],
        "adjacent_differences": adjacent,
        "first_to_last_difference": result["first_to_last_difference"],
        "configured_change_threshold": result["request"]["data_quality_requirements"][
            "change_threshold"
        ],
        "execution_issues": _execution_issues(execution_context),
        "structured_observations": structured_historical_observations(result),
        "limitations": result["limitations"],
        "outputs": result["outputs"],
    }


def _human_date(value: str) -> str:
    parsed = datetime.strptime(value, "%Y-%m-%d")
    return f"{parsed.day} {parsed.strftime('%B %Y')}"


def _signed_decimal(value: float) -> str:
    rounded = round(float(value), 4)
    return "0.0000" if rounded == 0 else f"{rounded:+.4f}"


def _answer_claim_ids(compact: Mapping[str, Any]) -> tuple[str, ...]:
    trend = compact["common_valid_trend"]
    claims = ["finding:common_trend" if trend["sufficient"] else "finding:insufficient_common"]
    claims.append("observations:actual_dates")
    if compact["missing_years"]:
        claims.append("observations:missing")
    for comparison in compact["adjacent_differences"]:
        prefix = "adjacent" if comparison["status"] == "available" else "adjacent_unavailable"
        claims.append(f"{prefix}:{comparison['label']}")
    endpoint = compact["first_to_last_difference"]
    available_intervals = {
        (item.get("before_year"), item.get("after_year"))
        for item in compact["adjacent_differences"]
        if item["status"] == "available"
    }
    if endpoint.get("status") == "available" and (
        endpoint.get("before_year"), endpoint.get("after_year")
    ) not in available_intervals:
        claims.append("comparison:first_to_last")
    claims.extend(issue["id"] for issue in compact["execution_issues"])
    claims.append("limitation:ndvi")
    return tuple(claims)


def supported_facts(
    result: Mapping[str, Any],
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return auditable claim identifiers and their evidence-bound plain-language rendering."""
    compact = compact_historical_result(result, execution_context)
    return {
        claim_id: _render_claim(claim_id, compact)
        for claim_id in _answer_claim_ids(compact)
    }


def _render_claim(claim_id: str, compact: Mapping[str, Any]) -> str:
    trend = compact["common_valid_trend"]
    if claim_id == "finding:common_trend":
        statistics = trend["annual_statistics"]
        first, last = statistics[0], statistics[-1]
        if last["mean"] > first["mean"]:
            comparison = f"higher in {last['year']} than in {first['year']}"
        elif last["mean"] < first["mean"]:
            comparison = f"lower in {last['year']} than in {first['year']}"
        else:
            comparison = f"the same in {last['year']} as in {first['year']}"
        return (
            f"Across the parts of the area that could be compared in every usable observation, "
            f"average satellite-observed vegetation greenness was {comparison}. "
            f"This endpoint comparison does not mean every part changed "
            "in the same direction or that change was steady between observations."
        )
    if claim_id == "finding:insufficient_common":
        return (
            "GeoLens could not report a directly comparable multi-year greenness finding because "
            "too little of the same area passed the validity checks in every usable observation. "
            "The per-year values cover different parts of the area, so they are not presented as "
            "one comparable trend."
        )
    if claim_id == "observations:actual_dates":
        available = [item for item in compact["observations"] if item["status"] == "available"]
        dates = ", ".join(
            f"{item['year']} ({_human_date(item['acquisition_date'])})" for item in available
        )
        return (
            f"The usable satellite observations were {dates}."
            if dates
            else "No usable observation was available."
        )
    if claim_id == "observations:missing":
        missing_years = compact["missing_years"]
        years = ", ".join(str(year) for year in missing_years)
        reference = "that year" if len(missing_years) == 1 else "those years"
        return (
            f"No suitable observation met the configured quality rules for {years}. GeoLens reports "
            f"no finding for {reference} and does not bridge an affected interval as an "
            "adjacent-year comparison."
        )
    if claim_id.startswith("adjacent:"):
        label = claim_id.split(":", 1)[1]
        item = next(value for value in compact["adjacent_differences"] if value["label"] == label)
        categories = item["categories"]
        mean = item["mean_ndvi_difference"]
        if mean > 0:
            direction = "higher on the later date"
        elif mean < 0:
            direction = "lower on the later date"
        else:
            direction = "unchanged on average"
        increased = categories["increased"]["fraction"] > 0
        decreased = categories["decreased"]["fraction"] > 0
        if increased and decreased:
            spatial_summary = "The difference map contains both increases and decreases."
        elif increased:
            spatial_summary = "The difference map contains areas of increase."
        elif decreased:
            spatial_summary = "The difference map contains areas of decrease."
        else:
            spatial_summary = "No area crossed the configured increase or decrease threshold."
        return (
            f"**{_human_date(item['before_date'])} to {_human_date(item['after_date'])}:** average "
            f"satellite-observed greenness was {direction} across the parts that could be compared. "
            f"{spatial_summary}"
        )
    if claim_id.startswith("adjacent_unavailable:"):
        label = claim_id.split(":", 1)[1]
        return f"**{label.replace('-to-', ' to ')}:** no adjacent-year comparison is available."
    if claim_id == "comparison:first_to_last":
        item = compact["first_to_last_difference"]
        if item["mean"] > 0:
            direction = "higher on the later date"
        elif item["mean"] < 0:
            direction = "lower on the later date"
        else:
            direction = "unchanged on average"
        return (
            f"From {_human_date(item['before_date'])} to {_human_date(item['after_date'])}, average "
            f"satellite-observed greenness was {direction} in the separate first-to-last "
            "comparison. That comparison does not use exactly the same common area as the "
            "multi-observation finding above."
        )
    if claim_id.startswith("execution_issue:"):
        issue = next(value for value in compact["execution_issues"] if value["id"] == claim_id)
        labels = {
            "construction": "building-change",
            "roads": "road-change",
            "vegetation_cover": "vegetation-cover",
            "tree_count": "individual-tree-count",
        }
        raw_analysis = str(issue.get("analysis") or "requested")
        analysis = labels.get(raw_analysis, raw_analysis.replace("_", " "))
        if issue["status"] == "not_run":
            return f"The requested {analysis} analysis was not run. {issue.get('reason')}"
        return f"The requested {analysis} analysis did not complete, so no finding is reported for it."
    if claim_id == "limitation:ndvi":
        return (
            "This comparison uses NDVI, a satellite-based indicator related to vegetation greenness. "
            "It does not establish tree counts, vegetation-covered area, plant health, causes, or "
            "statistical significance."
        )
    raise LanguageModelValidationError(f"unsupported answer claim identifier: {claim_id}")


def _render_conversational_answer(compact: Mapping[str, Any], claim_ids: Sequence[str]) -> str:
    rendered = {claim_id: _render_claim(claim_id, compact) for claim_id in claim_ids}
    paragraphs = [rendered[claim_ids[0]], rendered["observations:actual_dates"]]
    if "observations:missing" in rendered:
        paragraphs.append(rendered["observations:missing"])
    comparison_ids = [
        claim_id for claim_id in claim_ids
        if claim_id.startswith("adjacent:") or claim_id == "comparison:first_to_last"
    ]
    if comparison_ids:
        paragraphs.append("**Available comparisons**\n\n" + "\n".join(
            f"- {rendered[claim_id]}" for claim_id in comparison_ids
        ))
    unavailable_ids = [
        claim_id for claim_id in claim_ids
        if claim_id.startswith("adjacent_unavailable:")
    ]
    if unavailable_ids:
        paragraphs.append("\n".join(f"- {rendered[claim_id]}" for claim_id in unavailable_ids))
    issue_ids = [claim_id for claim_id in claim_ids if claim_id.startswith("execution_issue:")]
    if issue_ids:
        paragraphs.append("\n".join(rendered[claim_id] for claim_id in issue_ids))
    paragraphs.append(rendered["limitation:ndvi"])
    return "\n\n".join(paragraphs)


def _deterministic_answer(
    result: Mapping[str, Any],
    reason: str,
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    compact = compact_historical_result(result, execution_context)
    claim_ids = _answer_claim_ids(compact)
    return {
        "text": _render_conversational_answer(compact, claim_ids),
        "fallback_used": True,
        "fallback_reason": reason,
        "validated_claim_ids": list(claim_ids),
        "validated_fact_ids": list(claim_ids),
        "answer_kind": "deterministic_fallback",
        "model_claim_plan": None,
        "model_runtime": None,
    }


def generate_supported_answer(
    provider: OllamaProvider,
    result: Mapping[str, Any],
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use a schema-constrained claim plan, then render only evidence-backed prose."""
    compact = compact_historical_result(result, execution_context)
    claim_ids = _answer_claim_ids(compact)
    trend = compact["common_valid_trend"]
    if not trend["sufficient"]:
        expected_kind = "insufficient_common_coverage"
    elif compact["missing_years"]:
        expected_kind = "partial_historical_comparison"
    else:
        expected_kind = "historical_comparison"
    expected_lead = claim_ids[0]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer_kind": {"type": "string", "enum": [expected_kind]},
            "lead_claim_id": {"type": "string", "enum": [expected_lead]},
            "claim_ids": {
                "type": "array",
                "uniqueItems": True,
                "minItems": len(claim_ids),
                "maxItems": len(claim_ids),
                "items": {"type": "string", "enum": list(claim_ids)},
            },
        },
        "required": ["answer_kind", "lead_claim_id", "claim_ids"],
    }
    prompt = (
        "Select the complete evidence-backed claim plan for a concise, non-technical GeoLens answer. "
        "Return every allowed claim ID exactly once. Do not write prose, dates, numbers, locations, "
        "causes, or findings outside the schema. The renderer will insert checked language and values.\n\n"
        f"Structured result: {json.dumps(compact, separators=(',', ':'))}\n"
        f"Required claim IDs: {json.dumps(claim_ids)}"
    )
    try:
        value, runtime = provider.structured_chat(
            (
                {"role": "system", "content": "You select only validated geospatial result claims."},
                {"role": "user", "content": prompt},
            ),
            schema,
            max_tokens=512,
        )
        if set(value) != {"answer_kind", "lead_claim_id", "claim_ids"}:
            raise LanguageModelValidationError("answer plan has missing or unexpected fields")
        if value["answer_kind"] != expected_kind or value["lead_claim_id"] != expected_lead:
            raise LanguageModelValidationError("answer plan selected an unsupported conclusion")
        returned_ids = value["claim_ids"]
        if not isinstance(returned_ids, list) or any(not isinstance(item, str) for item in returned_ids):
            raise LanguageModelValidationError("answer claim IDs must be a string array")
        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(claim_ids):
            raise LanguageModelValidationError("answer plan omitted or invented a claim ID")
        return {
            "text": _render_conversational_answer(compact, claim_ids),
            "fallback_used": False,
            "fallback_reason": None,
            "validated_claim_ids": list(claim_ids),
            "validated_fact_ids": list(claim_ids),
            "answer_kind": expected_kind,
            "model_claim_plan": {
                "answer_kind": value["answer_kind"],
                "lead_claim_id": value["lead_claim_id"],
                "claim_ids": list(returned_ids),
            },
            "model_runtime": runtime,
        }
    except LanguageModelError as exc:
        return _deterministic_answer(
            result,
            f"Language-model answer rejected: {exc}",
            execution_context,
        )


def attach_final_answer(
    assistant_record: Mapping[str, Any],
    answer: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(assistant_record)
    updated["final_answer"] = dict(answer)
    path = Path(updated["outputs"]["assistant_record"])
    answer_path = path.with_name("answer.md")
    temporary = answer_path.with_suffix(".tmp.md")
    temporary.write_text(str(answer.get("text", "")) + "\n", encoding="utf-8")
    temporary.replace(answer_path)
    updated["outputs"] = {**updated["outputs"], "answer_text": str(answer_path.resolve())}
    _write_json(path, updated)
    return updated
