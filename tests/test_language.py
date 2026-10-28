from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.historical import (  # noqa: E402
    DEMO_AOI_LABEL,
    DEMO_BOUNDS,
    activate_confirmed_aoi,
    confirm_aoi,
    update_aoi_state,
)
from satquery.language import (  # noqa: E402
    LanguageModelUnavailableError,
    OllamaProvider,
    ProviderInfo,
    build_tool_plan,
    discover_ollama,
    execute_tool_plan,
    generate_supported_answer,
    interpret_request,
    supported_facts,
)


class FakeProvider:
    def __init__(self, payload=None, responder=None) -> None:
        self.info = ProviderInfo(
            True,
            "ollama",
            "http://127.0.0.1:11434",
            "test-model",
            ("test-model",),
        )
        self.payload = payload
        self.responder = responder

    def structured_chat(self, _messages, schema, **_kwargs):
        value = self.responder(schema) if self.responder else self.payload
        return value, {"model": "test-model", "eval_count": 10}


def interpretation_payload(
    *,
    vegetation: bool = False,
    construction: bool = False,
    roads: bool = False,
    start_year=2021,
    end_year=2025,
    boundary: bool = True,
    named_location: str | None = None,
) -> dict:
    return {
        "vegetation_requested": vegetation,
        "construction_requested": construction,
        "roads_requested": roads,
        "vegetation_evidence": "vegetation" if vegetation else None,
        "construction_evidence": "construction" if construction else None,
        "roads_evidence": "roads" if roads else None,
        "start_year": start_year,
        "end_year": end_year,
        "uses_confirmed_boundary": boundary,
        "boundary_evidence": "here" if boundary else None,
        "named_location": named_location,
    }


def result_fixture() -> dict:
    return {
        "request": {
            "years": [2021, 2022],
            "aoi": {"bounds_wgs84": list(DEMO_BOUNDS), "area_square_km": 4.878},
        },
        "annual_observations": [
            {
                "year": 2021,
                "status": "available",
                "acquisition_date": "2021-02-07",
                "valid_coverage": 0.9,
                "per_year_population": "year-valid pixels",
            },
            {"year": 2022, "status": "unavailable"},
        ],
        "available_years": [2021],
        "missing_years": [2022],
        "common_valid_trend": {
            "sufficient": False,
            "coverage": 0.75,
            "pixels": 75,
            "aoi_pixels": 100,
            "population": "pixels valid across every available observation",
            "reason": "At least two observations are required.",
            "annual_statistics": [],
        },
        "adjacent_differences": [
            {"label": "2021-to-2022", "status": "unavailable", "reason": "missing year"}
        ],
        "first_to_last_difference": {"status": "unavailable"},
        "limitations": ["NDVI does not establish cause."],
        "outputs": {"summary": "/tmp/summary.txt", "execution_record": "/tmp/run.json"},
    }


class ProviderDiscoveryTests(unittest.TestCase):
    def test_unavailable_local_service_is_reported_without_fallback(self) -> None:
        def unavailable(_url, _payload, _timeout):
            raise LanguageModelUnavailableError("offline")

        info = discover_ollama(transport=unavailable)
        self.assertFalse(info.available)
        self.assertIn("ollama serve", info.reason or "")


class InterpretationTests(unittest.TestCase):
    def test_demo_preset_sets_validated_confirmed_boundary_and_invalidates_old_plan(self) -> None:
        state = {
            "historical_draft_bounds": (76.35, 9.48, 76.37, 9.50),
            "historical_confirmed_aoi": confirm_aoi((76.35, 9.48, 76.37, 9.50)),
            "historical_aoi_source": "custom_map_selection",
            "historical_aoi_label": "Custom map area",
            "assistant_plan": object(),
            "historical_result": object(),
        }
        confirmed, changed = activate_confirmed_aoi(
            state, DEMO_BOUNDS, "demo_preset", DEMO_AOI_LABEL
        )
        self.assertTrue(changed)
        self.assertEqual(confirmed.bounds, DEMO_BOUNDS)
        self.assertEqual(state["historical_confirmed_aoi"], confirmed)
        self.assertEqual(state["historical_aoi_label"], DEMO_AOI_LABEL)
        self.assertAlmostEqual(confirmed.area_square_km, 4.8780529, places=6)
        self.assertNotIn("assistant_plan", state)
        self.assertNotIn("historical_result", state)

    def test_question_without_here_uses_active_demo_boundary(self) -> None:
        provider = FakeProvider(interpretation_payload(vegetation=True, boundary=False))
        interpreted = interpret_request(
            provider,
            "Compare vegetation from 2021 to 2025.",
            aoi_confirmed=True,
            active_boundary_label=DEMO_AOI_LABEL,
            active_boundary_source="demo_preset",
        )
        self.assertEqual(interpreted.missing_information, ())
        self.assertEqual(interpreted.boundary_source, "demo_preset")
        self.assertIsNone(interpreted.evidence["confirmed_boundary"])
        plan = build_tool_plan(interpreted, confirm_aoi(DEMO_BOUNDS), current_year=2026)
        self.assertTrue(plan.runnable)
        self.assertEqual(plan.record()["boundary_context"]["label"], DEMO_AOI_LABEL)

    def test_switching_to_custom_aoi_invalidates_demo_plan(self) -> None:
        state = {}
        activate_confirmed_aoi(state, DEMO_BOUNDS, "demo_preset", DEMO_AOI_LABEL)
        state["assistant_plan"] = object()
        state["historical_result"] = object()
        changed = update_aoi_state(state, (76.35, 9.48, 76.37, 9.50))
        self.assertTrue(changed)
        self.assertNotIn("historical_confirmed_aoi", state)
        self.assertNotIn("assistant_plan", state)
        self.assertNotIn("historical_result", state)

    def test_conflicting_named_location_requires_confirmation(self) -> None:
        provider = FakeProvider(
            interpretation_payload(
                vegetation=True,
                boundary=False,
                named_location="Kochi",
            )
        )
        interpreted = interpret_request(
            provider,
            "Compare vegetation in Kochi from 2021 to 2025.",
            aoi_confirmed=True,
            active_boundary_label=DEMO_AOI_LABEL,
            active_boundary_source="demo_preset",
        )
        self.assertIn("named_location_confirmation", interpreted.missing_information)
        self.assertIsNone(interpreted.boundary_source)
        plan = build_tool_plan(interpreted, confirm_aoi(DEMO_BOUNDS), current_year=2026)
        self.assertFalse(plan.runnable)

    def test_missing_dates_are_clarifications_and_invented_years_are_discarded(self) -> None:
        provider = FakeProvider(
            interpretation_payload(vegetation=True, start_year=2020, end_year=2024)
        )
        interpreted = interpret_request(provider, "Compare vegetation here.", aoi_confirmed=True)
        self.assertIsNone(interpreted.start_year)
        self.assertIsNone(interpreted.end_year)
        self.assertEqual(interpreted.missing_information, ("start_year", "end_year"))
        self.assertEqual(
            interpreted.runtime["discarded_years_without_source_evidence"], [2020, 2024]
        )

    def test_unconfirmed_aoi_blocks_an_otherwise_valid_vegetation_plan(self) -> None:
        provider = FakeProvider(interpretation_payload(vegetation=True))
        interpreted = interpret_request(
            provider, "Compare vegetation here from 2021 to 2025.", aoi_confirmed=False
        )
        self.assertIn("confirmed_boundary", interpreted.missing_information)
        plan = build_tool_plan(interpreted, None, current_year=2026)
        self.assertEqual(plan.status, "blocked")
        self.assertFalse(plan.runnable)

    def test_unsupported_construction_and_roads_intents_are_preserved(self) -> None:
        provider = FakeProvider(interpretation_payload(construction=True, roads=True))
        interpreted = interpret_request(
            provider,
            "Compare construction and roads here from 2021 to 2025.",
            aoi_confirmed=True,
        )
        plan = build_tool_plan(interpreted, confirm_aoi(DEMO_BOUNDS), current_year=2026)
        self.assertEqual(interpreted.analyses, ("construction", "roads"))
        self.assertEqual(plan.status, "blocked")
        self.assertEqual(
            [tool["status"] for tool in plan.tools],
            ["unsupported_in_historical_mode", "unsupported_in_historical_mode"],
        )

    def test_combined_request_retains_partial_support(self) -> None:
        provider = FakeProvider(interpretation_payload(vegetation=True, construction=True))
        interpreted = interpret_request(
            provider,
            "Compare vegetation and construction here from 2021 to 2025.",
            aoi_confirmed=True,
        )
        plan = build_tool_plan(interpreted, confirm_aoi(DEMO_BOUNDS), current_year=2026)
        self.assertEqual(plan.status, "partial_support")
        self.assertTrue(plan.runnable)
        self.assertIsNotNone(plan.historical_request)


class ToolExecutionTests(unittest.TestCase):
    def test_combined_execution_is_recorded_as_partial_support(self) -> None:
        provider = FakeProvider(interpretation_payload(vegetation=True, construction=True))
        interpreted = interpret_request(
            provider,
            "Compare vegetation and construction here from 2021 to 2025.",
            aoi_confirmed=True,
        )
        plan = build_tool_plan(interpreted, confirm_aoi(DEMO_BOUNDS), current_year=2026)

        def successful_runner(_request, _output_root, _cache_root):
            return {"outputs": {}, "available_years": [2021], "missing_years": [2022]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, result = execute_tool_plan(
                plan,
                root / "assistant",
                root / "historical",
                root / "cache",
                runner=successful_runner,
            )
        self.assertIsNotNone(result)
        self.assertEqual(record["status"], "partial")
        self.assertEqual(
            [item["status"] for item in record["tool_executions"]],
            ["success", "not_run"],
        )

    def test_tool_failure_is_recorded_without_arbitrary_execution(self) -> None:
        provider = FakeProvider(interpretation_payload(vegetation=True))
        interpreted = interpret_request(
            provider, "Compare vegetation here from 2021 to 2025.", aoi_confirmed=True
        )
        plan = build_tool_plan(interpreted, confirm_aoi(DEMO_BOUNDS), current_year=2026)

        def failed_runner(_request, _output_root, _cache_root):
            raise RuntimeError("fixture failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, result = execute_tool_plan(
                plan,
                root / "assistant",
                root / "historical",
                root / "cache",
                runner=failed_runner,
            )
            self.assertIsNone(result)
            self.assertEqual(record["status"], "failed")
            self.assertIn("fixture failure", record["tool_executions"][0]["error"])
            saved = json.loads(Path(record["outputs"]["assistant_record"]).read_text())
            self.assertEqual(saved["original_request"], interpreted.original_request)


class SupportedAnswerTests(unittest.TestCase):
    def test_missing_numerical_fact_reference_uses_deterministic_fallback(self) -> None:
        result = result_fixture()

        def response(schema):
            ids = list(schema["properties"]["fact_ids"]["items"]["enum"])
            return {
                "headline_code": "insufficient_comparable_coverage",
                "fact_ids": ids[:-1],
                "interpretation_code": "insufficient_common_coverage",
            }

        answer = generate_supported_answer(FakeProvider(responder=response), result)
        self.assertTrue(answer["fallback_used"])
        self.assertIn("fact ID", answer["fallback_reason"])

    def test_valid_answer_uses_only_deterministic_fact_values(self) -> None:
        result = result_fixture()

        def response(schema):
            ids = list(schema["properties"]["fact_ids"]["items"]["enum"])
            return {
                "headline_code": "insufficient_comparable_coverage",
                "fact_ids": ids,
                "interpretation_code": "insufficient_common_coverage",
            }

        answer = generate_supported_answer(FakeProvider(responder=response), result)
        self.assertFalse(answer["fallback_used"])
        for fact in supported_facts(result).values():
            self.assertIn(fact, answer["text"])


if __name__ == "__main__":
    unittest.main()
