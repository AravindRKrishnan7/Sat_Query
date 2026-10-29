from __future__ import annotations

import io
import json
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from satquery.api import create_app  # noqa: E402
from satquery.api_service import GeoLensApiService  # noqa: E402
from satquery.infrastructure import InfrastructureInterpretation  # noqa: E402
from satquery.language import InterpretedRequest, ProviderInfo  # noqa: E402


def _png(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _provider() -> ProviderInfo:
    return ProviderInfo(
        True,
        "ollama",
        "http://127.0.0.1:11434",
        "qwen3:8b",
        ("qwen3:8b",),
        None,
    )


def _infrastructure_interpretation(_provider: object, question: str) -> InfrastructureInterpretation:
    return InfrastructureInterpretation(
        original_request=question,
        requested_classes=("building_change", "road_change"),
        general_change_requested=False,
        instance_count_requested=False,
        spatial_description_requested=False,
        evidence={
            "buildings": "building",
            "roads": "road",
            "general_change": None,
            "instance_count": None,
            "spatial_description": None,
        },
        provider="ollama",
        model="qwen3:8b",
        runtime={"test": True},
    )


def _historical_interpretation(_provider: object, question: str) -> InterpretedRequest:
    return InterpretedRequest(
        original_request=question,
        analyses=("vegetation",),
        start_year=2021,
        end_year=2025,
        uses_confirmed_boundary=False,
        named_location=None,
        missing_information=(),
        evidence={
            "vegetation": "vegetation",
            "vegetation_cover": None,
            "tree_count": None,
            "construction": None,
            "roads": None,
            "boundary": None,
            "named_location": None,
        },
        provider="ollama",
        model="qwen3:8b",
        runtime={"test": True},
    )


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.output_root = Path(self.temporary.name) / "outputs" / "geolens"
        self.service = GeoLensApiService(
            project_root=PROJECT_ROOT,
            output_root=self.output_root,
            cache_root=Path(self.temporary.name) / "cache",
            provider_discovery=_provider,
            infrastructure_interpreter=_infrastructure_interpretation,
            historical_interpreter=_historical_interpretation,
        )
        self.client = TestClient(create_app(self.service))

    def tearDown(self) -> None:
        self.client.close()
        self.service.close()
        self.temporary.cleanup()

    def _infrastructure_plan(self, *, dates: list[str] | None = None, question: str = "building and road changes"):
        dates = dates or ["2025-01-02", "2023-01-02"]
        return self.client.post(
            "/api/infrastructure/plans",
            data={
                "question": question,
                "dates": json.dumps(dates),
                "resize": "false",
                "include_endpoint": "false",
                "alignment_confirmed": "true",
            },
            files=[
                ("files", ("later.png", _png((20, 40, 60)), "image/png")),
                ("files", ("earlier.png", _png((80, 100, 120)), "image/png")),
            ],
        )

    def test_health_and_exact_cors_origin(self) -> None:
        response = self.client.get(
            "/api/health", headers={"Origin": "http://127.0.0.1:5173"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://127.0.0.1:5173",
        )

    def test_infrastructure_plan_sorts_and_records_exact_inputs(self) -> None:
        response = self._infrastructure_plan()
        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()
        self.assertEqual(
            [item["acquisition_date"] for item in plan["chronological_observations"]],
            ["2023-01-02", "2025-01-02"],
        )
        self.assertEqual(plan["comparison_count"], 1)
        self.assertTrue(all(len(item["original_sha256"]) == 64 for item in plan["chronological_observations"]))
        self.assertTrue(all(item["preview_url"].startswith("/api/analyses/") for item in plan["chronological_observations"]))

    def test_duplicate_dates_and_missing_alignment_are_rejected(self) -> None:
        duplicate = self._infrastructure_plan(dates=["2024-01-01", "2024-01-01"])
        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("unique", duplicate.json()["detail"])
        missing_alignment = self.client.post(
            "/api/infrastructure/plans",
            data={
                "question": "building and road changes",
                "dates": '["2023-01-01","2024-01-01"]',
                "alignment_confirmed": "false",
            },
            files=[
                ("files", ("a.png", _png((1, 2, 3)), "image/png")),
                ("files", ("b.png", _png((4, 5, 6)), "image/png")),
            ],
        )
        self.assertEqual(missing_alignment.status_code, 400)
        self.assertIn("aligned", missing_alignment.json()["detail"])

    def test_changed_request_gets_new_identity_and_job_runs_once(self) -> None:
        first = self._infrastructure_plan(question="building and road changes").json()
        second = self._infrastructure_plan(question="building and road changes now").json()
        self.assertNotEqual(first["analysis_id"], second["analysis_id"])
        self.assertNotEqual(first["request_fingerprint"], second["request_fingerprint"])

        def fake_execution(analysis_id: str):
            return {
                "analysis_id": analysis_id,
                "workflow": "infrastructure_timeline",
                "status": "success",
                "answer": "Saved test result.",
                "raw_record": {"status": "success"},
            }

        self.service._execute_infrastructure = fake_execution  # type: ignore[method-assign]
        started = self.client.post(f"/api/plans/{first['analysis_id']}/run")
        self.assertEqual(started.status_code, 200)
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{first['analysis_id']}").json()
            if job["status"] == "success":
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "success")
        self.assertNotIn("raw_record", job["result"])
        repeated = self.client.post(f"/api/plans/{first['analysis_id']}/run")
        self.assertEqual(repeated.status_code, 400)
        self.assertIn("already", repeated.json()["detail"])

    def test_partial_job_result_is_preserved(self) -> None:
        plan = self._infrastructure_plan().json()

        def fake_partial(analysis_id: str):
            return {
                "analysis_id": analysis_id,
                "workflow": "infrastructure_timeline",
                "status": "partial",
                "answer": "One interval completed and one failed.",
                "comparisons": [
                    {"id": "comparison-01", "status": "success"},
                    {"id": "comparison-02", "status": "failed", "error": "test failure"},
                ],
                "raw_record": {"status": "partial"},
            }

        self.service._execute_infrastructure = fake_partial  # type: ignore[method-assign]
        self.client.post(f"/api/plans/{plan['analysis_id']}/run")
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{plan['analysis_id']}").json()
            if job["status"] == "partial":
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "partial")
        self.assertEqual(job["result"]["comparisons"][0]["status"], "success")
        self.assertEqual(job["result"]["comparisons"][1]["error"], "test failure")

    def test_follow_up_and_report_use_completed_record_without_new_job(self) -> None:
        plan = self._infrastructure_plan().json()
        analysis_id = plan["analysis_id"]
        raw_record = {
            "outputs": {
                "timeline_record": str(
                    self.output_root / "api" / analysis_id / "timeline.json"
                )
            },
            "observations": [],
            "comparisons": [],
        }
        completed = {
            "analysis_id": analysis_id,
            "workflow": "infrastructure_timeline",
            "status": "success",
            "progress": 100,
            "message": "Analysis complete",
            "created_at_utc": "2026-10-29T00:00:00+00:00",
            "result": {
                "analysis_id": analysis_id,
                "workflow": "infrastructure_timeline",
                "status": "success",
                "answer": "The model highlighted building-change areas.",
                "raw_record": raw_record,
            },
            "error": None,
        }
        self.service._jobs[analysis_id] = completed
        self.service._provider_discovery = lambda: ProviderInfo(
            False, "ollama", "http://127.0.0.1:11434", "qwen3:8b", (), "offline"
        )
        follow_up = self.client.post(
            f"/api/analyses/{analysis_id}/followups",
            json={"question": "How many buildings were added?"},
        )
        self.assertEqual(follow_up.status_code, 200, follow_up.text)
        self.assertIn("does not identify or reliably count", follow_up.json()["answer"])
        self.assertFalse(follow_up.json()["execution_performed"])

        def fake_report(_record: object, destination: Path) -> Path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"%PDF-1.4\n% test\n")
            return destination

        with patch("satquery.api_service.build_infrastructure_pdf", side_effect=fake_report) as builder:
            report = self.client.post(f"/api/analyses/{analysis_id}/report")
        self.assertEqual(report.status_code, 200, report.text)
        self.assertTrue(report.json()["generated_from_saved_results"])
        builder.assert_called_once()
        self.assertEqual(self.service._jobs[analysis_id]["status"], "success")

    def test_artifacts_are_scoped_to_one_analysis(self) -> None:
        plan = self._infrastructure_plan().json()
        url = plan["chronological_observations"][0]["preview_url"]
        self.assertEqual(self.client.get(url).status_code, 200)
        other = url.replace(plan["analysis_id"], "another-analysis")
        self.assertEqual(self.client.get(other).status_code, 404)
        self.assertEqual(
            self.client.get(
                f"/api/analyses/{plan['analysis_id']}/artifacts/not-registered"
            ).status_code,
            404,
        )

    def test_demo_vegetation_plan_uses_validated_preset(self) -> None:
        response = self.client.post(
            "/api/vegetation/plans",
            json={"question": "Compare vegetation from 2021 to 2025."},
        )
        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()
        self.assertEqual(plan["status"], "validated")
        self.assertEqual(plan["area"]["label"], "Alappuzha demo area")
        self.assertEqual(plan["area"]["bounds_wgs84"], [76.34, 9.48, 76.36, 9.5])
        self.assertEqual(
            plan["plan"]["boundary_resolution"]["source"], "application_state"
        )

    def test_oversized_custom_vegetation_area_is_rejected(self) -> None:
        response = self.client.post(
            "/api/vegetation/plans",
            json={
                "question": "Compare vegetation from 2021 to 2025.",
                "bounds": [76.0, 9.0, 77.0, 10.0],
                "area_label": "Too large",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("limit", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
