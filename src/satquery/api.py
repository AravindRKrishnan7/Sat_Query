"""Loopback-only FastAPI adapter for the GeoLens local frontend."""

from __future__ import annotations

import json
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from satquery.api_service import ApiServiceError, GeoLensApiService
from satquery.geolens import MAX_UPLOAD_BYTES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTEND_ORIGIN = "http://127.0.0.1:5173"


class VegetationPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    area_label: str | None = Field(default=None, max_length=120)


class FollowUpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    selected_comparison_id: str | None = Field(default=None, max_length=120)


def create_app(service: GeoLensApiService | None = None) -> FastAPI:
    owned_service = service is None
    runtime = service or GeoLensApiService(project_root=PROJECT_ROOT)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owned_service:
            runtime.close()

    app = FastAPI(
        title="GeoLens local API",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    origin = os.environ.get("GEOLENS_FRONTEND_ORIGIN", DEFAULT_FRONTEND_ORIGIN).rstrip("/")
    if origin not in {"http://127.0.0.1:5173", "http://localhost:5173"}:
        raise RuntimeError(
            "GEOLENS_FRONTEND_ORIGIN must be the local Vite origin on port 5173"
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.geolens = runtime

    def translate_error(exc: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(exc) or type(exc).__name__)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "execution": "bounded single-worker local service",
            "frontend_origin": origin,
        }

    @app.get("/api/provider")
    def provider() -> dict[str, Any]:
        return runtime.provider_status()

    @app.post("/api/infrastructure/plans")
    def infrastructure_plan(
        files: Annotated[list[UploadFile], File(description="Two to four PNG/JPEG observations")],
        question: Annotated[str, Form(min_length=1, max_length=2_000)],
        dates: Annotated[str, Form(description="JSON array of ISO acquisition dates")],
        resize: Annotated[bool, Form()] = False,
        include_endpoint: Annotated[bool, Form()] = False,
        alignment_confirmed: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        try:
            date_values = json.loads(dates)
            if not isinstance(date_values, list) or any(
                not isinstance(value, str) for value in date_values
            ):
                raise ApiServiceError("dates must be a JSON array of ISO date strings")
            payloads: list[bytes] = []
            filenames: list[str] = []
            for uploaded in files:
                value = uploaded.file.read(MAX_UPLOAD_BYTES + 1)
                if len(value) > MAX_UPLOAD_BYTES:
                    raise ApiServiceError(
                        f"{uploaded.filename or 'upload'} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
                    )
                payloads.append(value)
                filenames.append(uploaded.filename or "upload")
            return runtime.create_infrastructure_plan(
                question=question,
                filenames=filenames,
                upload_bytes=payloads,
                dates=date_values,
                resize=resize,
                include_endpoint=include_endpoint,
                alignment_confirmed=alignment_confirmed,
            )
        except (ApiServiceError, ValueError, json.JSONDecodeError) as exc:
            raise translate_error(exc) from exc

    @app.post("/api/vegetation/plans")
    def vegetation_plan(request: VegetationPlanRequest) -> dict[str, Any]:
        try:
            return runtime.create_vegetation_plan(
                question=request.question,
                bounds=request.bounds,
                area_label=request.area_label,
            )
        except (ApiServiceError, ValueError) as exc:
            raise translate_error(exc) from exc

    @app.get("/api/vegetation/preset")
    def vegetation_preset() -> dict[str, Any]:
        try:
            return runtime.vegetation_preset()
        except (ApiServiceError, ValueError, OSError) as exc:
            raise translate_error(exc) from exc

    @app.get("/api/plans/{analysis_id}")
    def get_plan(analysis_id: str) -> dict[str, Any]:
        try:
            return runtime.plan(analysis_id)
        except ApiServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/plans/{analysis_id}/run")
    def run_plan(analysis_id: str) -> dict[str, Any]:
        try:
            return runtime.start(analysis_id)
        except ApiServiceError as exc:
            raise translate_error(exc) from exc

    @app.get("/api/jobs/{analysis_id}")
    def get_job(analysis_id: str) -> dict[str, Any]:
        try:
            return runtime.job(analysis_id)
        except ApiServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/analyses/{analysis_id}/followups")
    def follow_up(analysis_id: str, request: FollowUpRequest) -> dict[str, Any]:
        try:
            return runtime.follow_up(
                analysis_id,
                request.question,
                request.selected_comparison_id,
            )
        except (ApiServiceError, ValueError) as exc:
            raise translate_error(exc) from exc

    @app.post("/api/analyses/{analysis_id}/report")
    def report(analysis_id: str) -> dict[str, Any]:
        try:
            return runtime.build_report(analysis_id)
        except (ApiServiceError, ValueError, OSError) as exc:
            raise translate_error(exc) from exc

    @app.get("/api/analyses/{analysis_id}/artifacts/{artifact_id}")
    def artifact(analysis_id: str, artifact_id: str) -> FileResponse:
        try:
            path = runtime.artifact_path(analysis_id, artifact_id)
        except ApiServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(
            path,
            media_type=media_type or "application/octet-stream",
            filename=path.name,
        )

    return app


app = create_app()
