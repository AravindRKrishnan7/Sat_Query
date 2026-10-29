"""Local job service around GeoLens' existing reusable analysis modules.

This module deliberately contains no HTTP framework code.  FastAPI and
Streamlit both remain thin presentation layers over the same validated
analysis functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from satquery.change_agent import load_model, run as run_change_agent
from satquery.followup import FollowUpContext, answer_follow_up
from satquery.geolens import PreparedImage, create_unique_run_directory, prepare_uploaded_image
from satquery.historical import DEMO_AOI_LABEL, DEMO_BOUNDS, ConfirmedAOI, confirm_aoi
from satquery.infrastructure import (
    InfrastructureInterpretation,
    interpret_infrastructure_request,
)
from satquery.infrastructure_timeline import (
    TimelineObservation,
    execute_infrastructure_timeline,
    timeline_request_fingerprint,
    validate_timeline_observations,
)
from satquery.language import (
    OllamaProvider,
    ProviderInfo,
    ToolPlan,
    attach_final_answer,
    build_tool_plan,
    discover_ollama,
    execute_tool_plan,
    generate_supported_answer,
    interpret_request,
    summarize_tool_plan,
)
from satquery.reports import build_historical_pdf, build_infrastructure_pdf


class ApiServiceError(ValueError):
    """Raised for a validated local-API request error."""


ProviderDiscovery = Callable[[], ProviderInfo]
InfrastructureInterpreter = Callable[[OllamaProvider, str], InfrastructureInterpretation]
HistoricalInterpreter = Callable[[OllamaProvider, str], Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    name = Path(value).name.strip()
    if not name:
        return "upload"
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in name)[:120]


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class GeoLensApiService:
    """Bounded, in-process local execution and artifact registry."""

    def __init__(
        self,
        *,
        project_root: Path,
        output_root: Path | None = None,
        cache_root: Path | None = None,
        provider_discovery: ProviderDiscovery = discover_ollama,
        infrastructure_interpreter: InfrastructureInterpreter = interpret_infrastructure_request,
        historical_interpreter: HistoricalInterpreter = interpret_request,
        max_workers: int = 1,
    ) -> None:
        if max_workers != 1:
            raise ApiServiceError("GeoLens local execution is intentionally limited to one worker")
        self.project_root = project_root.resolve()
        self.output_root = (output_root or self.project_root / "outputs" / "geolens").resolve()
        self.api_root = self.output_root / "api"
        self.historical_output_root = self.output_root / "historical"
        self.assistant_output_root = self.output_root / "assistant"
        self.cache_root = (cache_root or self.project_root / "data" / "cache" / "geolens-historical").resolve()
        self.checkpoint = self.project_root / "weights" / "MCI_model.pth"
        self.change_source = self.project_root / ".cache" / "upstream" / "change-agent"
        self._provider_discovery = provider_discovery
        self._infrastructure_interpreter = infrastructure_interpreter
        self._historical_interpreter = historical_interpreter
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geolens-analysis")
        self._lock = threading.RLock()
        self._model_lock = threading.Lock()
        self._model_cache: tuple[Any, Any, Any, dict[str, Any]] | None = None
        self._plans: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._artifacts: dict[str, dict[str, Path]] = {}

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def provider_status(self) -> dict[str, Any]:
        return self._provider_discovery().record()

    def _provider(self) -> OllamaProvider:
        info = self._provider_discovery()
        if not info.available:
            raise ApiServiceError(info.reason or "local Ollama is unavailable")
        return OllamaProvider(info)

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:10]}"

    def _analysis_directory(self, analysis_id: str) -> Path:
        path = (self.api_root / analysis_id).resolve()
        if path.parent != self.api_root.resolve():
            raise ApiServiceError("invalid analysis identifier")
        return path

    def _register_artifact(self, analysis_id: str, path: Path, label: str | None = None) -> str:
        resolved = path.resolve()
        allowed = (self.output_root, self.api_root)
        if not resolved.is_file() or not any(resolved.is_relative_to(root) for root in allowed):
            raise ApiServiceError("analysis artifact is missing or outside the registered output roots")
        with self._lock:
            registry = self._artifacts.setdefault(analysis_id, {})
            for key, registered in registry.items():
                if registered == resolved:
                    return key
            suffix = _safe_name(label or resolved.name)
            key = f"{uuid.uuid4().hex[:12]}-{suffix}"
            registry[key] = resolved
            return key

    def artifact_path(self, analysis_id: str, artifact_id: str) -> Path:
        with self._lock:
            path = self._artifacts.get(analysis_id, {}).get(artifact_id)
        if path is None or not path.is_file():
            raise ApiServiceError("artifact is not registered for this analysis")
        return path

    @staticmethod
    def artifact_url(analysis_id: str, artifact_id: str) -> str:
        return f"/api/analyses/{analysis_id}/artifacts/{artifact_id}"

    def _artifact_reference(self, analysis_id: str, value: str | Path | None, label: str | None = None) -> str | None:
        if not value:
            return None
        path = Path(value)
        if not path.is_file():
            return None
        artifact_id = self._register_artifact(analysis_id, path, label)
        return self.artifact_url(analysis_id, artifact_id)

    def create_infrastructure_plan(
        self,
        *,
        question: str,
        filenames: Sequence[str],
        upload_bytes: Sequence[bytes],
        dates: Sequence[str],
        resize: bool,
        include_endpoint: bool,
        alignment_confirmed: bool,
    ) -> dict[str, Any]:
        if not alignment_confirmed:
            raise ApiServiceError(
                "confirm that every image shows the same extent and is already spatially aligned"
            )
        if len(upload_bytes) != len(filenames):
            raise ApiServiceError("every uploaded image must have a filename")
        prepared: list[PreparedImage] = [
            prepare_uploaded_image(value, f"image {index}", resize)
            for index, value in enumerate(upload_bytes, start=1)
        ]
        observations = validate_timeline_observations(filenames, prepared, dates)
        if include_endpoint and len(observations) == 2:
            include_endpoint = False
        if not self.checkpoint.is_file():
            raise ApiServiceError(f"verified checkpoint is missing: {self.checkpoint}")
        if not self.change_source.is_dir():
            raise ApiServiceError(f"prepared Change-Agent source is missing: {self.change_source}")
        interpretation = self._infrastructure_interpreter(self._provider(), question)
        fingerprint = timeline_request_fingerprint(
            question, upload_bytes, dates, resize, include_endpoint
        )
        analysis_id = self._new_id("infrastructure")
        directory = self._analysis_directory(analysis_id)
        review_directory = directory / "review"
        review_directory.mkdir(parents=True, exist_ok=False)
        review: list[dict[str, Any]] = []
        by_upload_order = {index: value for index, value in enumerate(upload_bytes, start=1)}
        for sequence, observation in enumerate(observations, start=1):
            path = review_directory / f"{sequence:02d}-{observation.acquisition_date.isoformat()}.png"
            observation.prepared.image.save(path, format="PNG")
            artifact_id = self._register_artifact(analysis_id, path, path.name)
            original = by_upload_order[observation.upload_order]
            review.append(
                {
                    "chronological_order": sequence,
                    "upload_order": observation.upload_order,
                    "filename": observation.filename,
                    "acquisition_date": observation.acquisition_date.isoformat(),
                    "original_sha256": _hash_bytes(original),
                    "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "original_size": list(observation.prepared.original_size),
                    "resized": observation.prepared.resized,
                    "preview_url": self.artifact_url(analysis_id, artifact_id),
                }
            )
        public = {
            "analysis_id": analysis_id,
            "workflow": "infrastructure_timeline",
            "status": "validated",
            "question": question.strip(),
            "request_fingerprint": fingerprint,
            "alignment_confirmed": True,
            "resize_requested": resize,
            "include_endpoint": include_endpoint,
            "chronological_observations": review,
            "comparison_count": len(observations) - 1 + (1 if include_endpoint and len(observations) > 2 else 0),
            "interpretation": interpretation.record(),
            "created_at_utc": _utc_now(),
        }
        _json_write(directory / "plan.json", public)
        with self._lock:
            self._plans[analysis_id] = {
                "workflow": "infrastructure_timeline",
                "directory": directory,
                "observations": observations,
                "question": question.strip(),
                "fingerprint": fingerprint,
                "interpretation": interpretation,
                "resize": resize,
                "include_endpoint": include_endpoint,
                "public": public,
            }
        return public

    def create_vegetation_plan(
        self,
        *,
        question: str,
        bounds: Sequence[float] | None = None,
        area_label: str | None = None,
    ) -> dict[str, Any]:
        selected_bounds = tuple(bounds) if bounds is not None else DEMO_BOUNDS
        label = (area_label or (DEMO_AOI_LABEL if bounds is None else "Custom map area")).strip()
        aoi = confirm_aoi(selected_bounds)
        provider = self._provider()
        interpretation = self._historical_interpreter(provider, question)
        plan = build_tool_plan(
            interpretation,
            aoi,
            current_year=datetime.now(timezone.utc).year,
            active_boundary_source="demo_preset" if bounds is None else "api_confirmed_bounds",
            active_boundary_label=label,
        )
        analysis_id = self._new_id("vegetation")
        directory = self._analysis_directory(analysis_id)
        directory.mkdir(parents=True, exist_ok=False)
        public = {
            "analysis_id": analysis_id,
            "workflow": "historical_vegetation",
            "status": "validated" if plan.runnable else "blocked",
            "question": question.strip(),
            "area": {"label": label, **aoi.record()},
            "summary": summarize_tool_plan(plan),
            "plan": plan.record(),
            "created_at_utc": _utc_now(),
        }
        _json_write(directory / "plan.json", public)
        with self._lock:
            self._plans[analysis_id] = {
                "workflow": "historical_vegetation",
                "directory": directory,
                "question": question.strip(),
                "aoi": aoi,
                "area_label": label,
                "plan": plan,
                "provider_info": provider.info,
                "public": public,
            }
        return public

    def vegetation_preset(self) -> dict[str, Any]:
        """Return validated preset context and an optional matching saved preview.

        This is intentionally read-only: it does not create a plan, run the
        historical engine, query the provider, or populate the raster cache.
        """
        aoi = confirm_aoi(DEMO_BOUNDS)
        context_id = "vegetation-preset"
        preview: dict[str, Any] | None = None
        if self.historical_output_root.is_dir():
            records = sorted(
                self.historical_output_root.glob("run-*/run.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for record_path in records:
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                stored_bounds = record.get("request", {}).get("aoi", {}).get("bounds_wgs84")
                if not isinstance(stored_bounds, list) or len(stored_bounds) != 4:
                    continue
                if tuple(round(float(value), 8) for value in stored_bounds) != aoi.bounds:
                    continue
                candidates = [
                    item
                    for item in record.get("annual_observations", [])
                    if item.get("status") == "available"
                    and item.get("acquisition_date")
                    and item.get("outputs", {}).get("true_color_png")
                ]
                candidates.sort(key=lambda item: str(item["acquisition_date"]), reverse=True)
                for observation in candidates:
                    image_path = Path(observation["outputs"]["true_color_png"]).resolve()
                    if not image_path.is_file() or not image_path.is_relative_to(record_path.parent):
                        continue
                    image_id = self._register_artifact(context_id, image_path, image_path.name)
                    record_id = self._register_artifact(context_id, record_path, record_path.name)
                    preview = {
                        "year": observation.get("year"),
                        "acquisition_date": observation["acquisition_date"],
                        "image_url": self.artifact_url(context_id, image_id),
                        "source_record_url": self.artifact_url(context_id, record_id),
                        "label": "Previous saved observation; not an input to the pending analysis",
                    }
                    break
                if preview is not None:
                    break
        return {
            "area": {"label": DEMO_AOI_LABEL, **aoi.record()},
            "boundary_source": "backend_validated_demo_preset",
            "preview": preview,
            "preview_status": (
                "matching_saved_observation" if preview is not None else "unavailable"
            ),
            "preview_unavailable_reason": (
                None
                if preview is not None
                else "No matching saved true-colour observation is available. Imagery will appear after retrieval and analysis."
            ),
            "analysis_performed": False,
            "retrieval_performed": False,
        }

    def plan(self, analysis_id: str) -> dict[str, Any]:
        with self._lock:
            plan = self._plans.get(analysis_id)
        if plan is None:
            raise ApiServiceError("analysis plan was not found or the local server restarted")
        return dict(plan["public"])

    def start(self, analysis_id: str) -> dict[str, Any]:
        with self._lock:
            if analysis_id in self._jobs:
                raise ApiServiceError("this validated plan has already been started")
            plan = self._plans.get(analysis_id)
            if plan is None:
                raise ApiServiceError("analysis plan was not found or the local server restarted")
            if plan["workflow"] == "historical_vegetation" and not plan["plan"].runnable:
                raise ApiServiceError("the validated vegetation plan is blocked and cannot run")
            job = {
                "analysis_id": analysis_id,
                "workflow": plan["workflow"],
                "status": "queued",
                "progress": 0,
                "message": "Waiting for the local analysis worker",
                "created_at_utc": _utc_now(),
                "result": None,
                "error": None,
            }
            self._jobs[analysis_id] = job
            self._persist_job(job, plan["directory"])
            self._futures[analysis_id] = self._executor.submit(self._execute, analysis_id)
            return self._job_public(job)

    def _persist_job(self, job: Mapping[str, Any], directory: Path) -> None:
        safe = {key: value for key, value in job.items() if key != "result"}
        if isinstance(job.get("result"), Mapping):
            safe["result"] = job["result"]
        _json_write(directory / "job.json", safe)

    def _update_job(self, analysis_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[analysis_id]
            job.update(changes)
            plan = self._plans[analysis_id]
            self._persist_job(job, plan["directory"])

    def _execute(self, analysis_id: str) -> None:
        try:
            with self._lock:
                workflow = self._plans[analysis_id]["workflow"]
            self._update_job(
                analysis_id,
                status="running",
                progress=2,
                message="Validated request accepted by the local worker",
                started_at_utc=_utc_now(),
            )
            if workflow == "infrastructure_timeline":
                result = self._execute_infrastructure(analysis_id)
            else:
                result = self._execute_vegetation(analysis_id)
            status = result.get("status", "success")
            self._update_job(
                analysis_id,
                status="partial" if status == "partial" else "success",
                progress=100,
                message="Analysis complete" if status != "partial" else "Analysis complete with partial results",
                completed_at_utc=_utc_now(),
                result=result,
            )
        except Exception as exc:
            self._update_job(
                analysis_id,
                status="failed",
                progress=100,
                message="Analysis failed",
                completed_at_utc=_utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _cached_model_loader(self, source: Path, checkpoint: Path, device: Any, torch: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        with self._model_lock:
            if self._model_cache is None:
                self._model_cache = load_model(source, checkpoint, device, torch)
            return self._model_cache

    def _execute_infrastructure(self, analysis_id: str) -> dict[str, Any]:
        with self._lock:
            plan = self._plans[analysis_id]
        total = len(plan["observations"]) - 1 + (
            1 if plan["include_endpoint"] and len(plan["observations"]) > 2 else 0
        )
        completed = 0

        def comparison_runner(before: Path, after: Path, output: Path) -> Mapping[str, Any]:
            nonlocal completed
            self._update_job(
                analysis_id,
                progress=10 + round((completed / max(total, 1)) * 75),
                message=f"Running comparison {completed + 1} of {total}",
            )
            args = argparse.Namespace(
                before=before,
                after=after,
                checkpoint=self.checkpoint,
                output_dir=output,
                device="cpu",
                source_dir=self.change_source,
            )
            record = run_change_agent(args, model_loader=self._cached_model_loader)
            completed += 1
            self._update_job(
                analysis_id,
                progress=10 + round((completed / max(total, 1)) * 75),
                message=f"Completed comparison {completed} of {total}",
            )
            return record

        timeline = execute_infrastructure_timeline(
            parent_directory=plan["directory"],
            question=plan["question"],
            fingerprint=plan["fingerprint"],
            interpretation=plan["interpretation"],
            observations=plan["observations"],
            resize_requested=plan["resize"],
            include_endpoint=plan["include_endpoint"],
            comparison_runner=comparison_runner,
        )
        self._update_job(analysis_id, progress=92, message="Registering saved evidence")
        return self._normalise_infrastructure(analysis_id, timeline)

    def _execute_vegetation(self, analysis_id: str) -> dict[str, Any]:
        with self._lock:
            plan = self._plans[analysis_id]
        tool_plan: ToolPlan = plan["plan"]
        self._update_job(analysis_id, progress=10, message="Checking cached observations and retrieval policy")
        execution, result = execute_tool_plan(
            tool_plan,
            self.assistant_output_root,
            self.historical_output_root,
            self.cache_root,
        )
        if result is None:
            raise ApiServiceError("the validated historical tool produced no result")
        self._update_job(analysis_id, progress=85, message="Preparing the evidence-backed answer")
        answer = generate_supported_answer(OllamaProvider(plan["provider_info"]), result, execution)
        execution = attach_final_answer(execution, answer)
        return self._normalise_vegetation(
            analysis_id,
            result,
            execution,
            answer,
            plan["area_label"],
        )

    def _normalise_infrastructure(self, analysis_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        for item in record.get("observations", []):
            observations.append(
                {
                    "id": item["id"],
                    "date": item["acquisition_date"],
                    "filename": item["original_upload"]["filename"],
                    "original_sha256": item["original_upload"]["sha256"],
                    "prepared_sha256": item["prepared_file"]["sha256"],
                    "resized": item["preparation"]["resized"],
                    "image_url": self._artifact_reference(analysis_id, item["prepared_file"]["path"]),
                }
            )
        comparisons: list[dict[str, Any]] = []
        for item in record.get("comparisons", []):
            public = {
                "id": item["id"],
                "kind": item["kind"],
                "before_observation": item["before_observation"],
                "after_observation": item["after_observation"],
                "before_date": item["before_date"],
                "after_date": item["after_date"],
                "status": item["status"],
                "answer": item.get("answer"),
                "error": item.get("error"),
            }
            if item.get("status") == "success":
                public.update(
                    {
                        "overlay_url": self._artifact_reference(analysis_id, item.get("overlay")),
                        "class_mask_url": self._artifact_reference(analysis_id, item.get("class_mask")),
                        "caption_url": self._artifact_reference(analysis_id, item.get("raw_caption")),
                        "execution_record_url": self._artifact_reference(analysis_id, item.get("execution_record")),
                        "answer_url": self._artifact_reference(analysis_id, item.get("answer_file")),
                    }
                )
            comparisons.append(public)
        answer_path = record["outputs"]["timeline_answer"]
        answer = Path(answer_path).read_text(encoding="utf-8").strip()
        return {
            "analysis_id": analysis_id,
            "workflow": "infrastructure_timeline",
            "status": record.get("status"),
            "question": record.get("question"),
            "answer": answer,
            "observations": observations,
            "comparisons": comparisons,
            "record_url": self._artifact_reference(analysis_id, record["outputs"]["timeline_record"]),
            "answer_url": self._artifact_reference(analysis_id, answer_path),
            "limitations": [
                "The images and dates were supplied by the user; alignment and acquisition metadata were not independently verified.",
                "The mask and caption are separate model predictions and may disagree.",
                "Highlighted regions are not individual-building counts or verified road construction.",
            ],
            "raw_record": dict(record),
        }

    def _normalise_vegetation(
        self,
        analysis_id: str,
        record: Mapping[str, Any],
        execution: Mapping[str, Any],
        answer: Mapping[str, Any],
        area_label: str,
    ) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        for item in record.get("annual_observations", []):
            public = {
                "year": item["year"],
                "status": item["status"],
                "reason": item.get("reason"),
                "acquisition_date": item.get("acquisition_date"),
            }
            if item.get("status") == "available":
                outputs = item.get("outputs", {})
                public.update(
                    {
                        "true_color_url": self._artifact_reference(analysis_id, outputs.get("true_color_png")),
                        "ndvi_url": self._artifact_reference(analysis_id, outputs.get("ndvi_png")),
                    }
                )
            observations.append(public)
        differences: list[dict[str, Any]] = []
        for item in list(record.get("adjacent_differences", [])) + [record.get("first_to_last_difference", {})]:
            if not item:
                continue
            differences.append(
                {
                    "label": item.get("label") or f"{item.get('before_year')}-to-{item.get('after_year')}",
                    "before_year": item.get("before_year"),
                    "after_year": item.get("after_year"),
                    "before_date": item.get("before_date"),
                    "after_date": item.get("after_date"),
                    "status": item.get("status"),
                    "reason": item.get("reason"),
                    "difference_url": self._artifact_reference(
                        analysis_id, item.get("outputs", {}).get("difference_png")
                    ),
                }
            )
        return {
            "analysis_id": analysis_id,
            "workflow": "historical_vegetation",
            "status": record.get("status"),
            "question": execution.get("original_request"),
            "area_label": area_label,
            "area": record.get("request", {}).get("aoi"),
            "answer": answer.get("text", ""),
            "answer_validation": {
                "fallback_used": answer.get("fallback_used"),
                "fallback_reason": answer.get("fallback_reason"),
                "validated_claim_ids": answer.get("validated_claim_ids"),
            },
            "available_years": record.get("available_years", []),
            "missing_years": record.get("missing_years", []),
            "observations": observations,
            "differences": differences,
            "record_url": self._artifact_reference(analysis_id, record["outputs"]["execution_record"]),
            "assistant_record_url": self._artifact_reference(analysis_id, execution["outputs"]["assistant_record"]),
            "answer_url": self._artifact_reference(analysis_id, execution["outputs"]["answer_text"]),
            "limitations": list(record.get("limitations", [])),
            "raw_record": dict(record),
            "raw_execution": dict(execution),
        }

    def _job_public(self, job: Mapping[str, Any]) -> dict[str, Any]:
        result = job.get("result")
        public_result = None
        if isinstance(result, Mapping):
            public_result = {
                key: value
                for key, value in result.items()
                if key not in {"raw_record", "raw_execution"}
            }
        return {
            "analysis_id": job["analysis_id"],
            "workflow": job["workflow"],
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "created_at_utc": job["created_at_utc"],
            "started_at_utc": job.get("started_at_utc"),
            "completed_at_utc": job.get("completed_at_utc"),
            "error": job.get("error"),
            "result": public_result,
        }

    def job(self, analysis_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(analysis_id)
        if job is None:
            raise ApiServiceError("analysis job was not found or the local server restarted")
        return self._job_public(job)

    def follow_up(
        self,
        analysis_id: str,
        question: str,
        selected_comparison_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(analysis_id)
        if job is None:
            raise ApiServiceError("analysis job was not found or the local server restarted")
        if job["status"] not in {"success", "partial"} or not isinstance(job.get("result"), Mapping):
            raise ApiServiceError("follow-ups require a completed analysis")
        result = job["result"]
        workflow = result["workflow"]
        if workflow == "infrastructure_timeline":
            record = result["raw_record"]
            base_answer = result["answer"]
        else:
            record = result["raw_record"]
            base_answer = result["answer"]
        provider: OllamaProvider | None = None
        try:
            provider = self._provider()
        except ApiServiceError:
            provider = None
        response = answer_follow_up(
            question,
            FollowUpContext(
                workflow=workflow,
                analysis_id=analysis_id,
                record=record,
                base_answer=base_answer,
                selected_comparison_id=selected_comparison_id,
            ),
            provider,
        )
        directory = self._analysis_directory(analysis_id) / "followups"
        path = directory / f"followup-{uuid.uuid4().hex[:12]}.json"
        _json_write(path, response)
        artifact_id = self._register_artifact(analysis_id, path, path.name)
        return {**response, "record_url": self.artifact_url(analysis_id, artifact_id)}

    def build_report(self, analysis_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(analysis_id)
        if job is None:
            raise ApiServiceError("analysis job was not found or the local server restarted")
        if job["status"] not in {"success", "partial"} or not isinstance(job.get("result"), Mapping):
            raise ApiServiceError("report export requires a completed analysis")
        result = job["result"]
        destination = self._analysis_directory(analysis_id) / "reports" / "geolens-report.pdf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if result["workflow"] == "infrastructure_timeline":
            build_infrastructure_pdf(result["raw_record"], destination)
        else:
            build_historical_pdf(
                result["raw_record"],
                destination,
                original_request=str(result.get("question") or ""),
                answer=str(result.get("answer") or ""),
                area_label=str(result.get("area_label") or "Confirmed map area"),
            )
        artifact_id = self._register_artifact(analysis_id, destination, destination.name)
        return {
            "analysis_id": analysis_id,
            "report_url": self.artifact_url(analysis_id, artifact_id),
            "generated_from_saved_results": True,
        }
