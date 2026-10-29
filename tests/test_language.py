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
    attach_final_answer,
    build_tool_plan,
    discover_ollama,
    execute_tool_plan,
    generate_supported_answer,
    interpret_request,
    structured_historical_observations,
    summarize_tool_plan,
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
    vegetation_cover: bool = False,
    tree_count: bool = False,
    construction: bool = False,
    roads: bool = False,
    start_year=2021,
    end_year=2025,
    boundary: bool = True,
    named_location: str | None = None,
    vegetation_evidence: str | None = None,
    vegetation_cover_evidence: str | None = None,
    tree_count_evidence: str | None = None,
    construction_evidence: str | None = None,
    roads_evidence: str | None = None,
) -> dict:
    return {
        "vegetation_requested": vegetation,
        "vegetation_cover_requested": vegetation_cover,
        "tree_count_requested": tree_count,
        "construction_requested": construction,
        "roads_requested": roads,
        "vegetation_evidence": (vegetation_evidence or "vegetation") if vegetation else None,
        "vegetation_cover_evidence": (
            vegetation_cover_evidence or "vegetation cover"
        ) if vegetation_cover else None,
        "tree_count_evidence": (tree_count_evidence or "tree count") if tree_count else None,
        "construction_evidence": (construction_evidence or "construction") if construction else None,
        "roads_evidence": (roads_evidence or "roads") if roads else None,
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
            "data_quality_requirements": {"change_threshold": 0.1},
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


def complete_result_fixture() -> dict:
    categories = {
        "increased": {"pixels": 20, "fraction": 0.2},
        "decreased": {"pixels": 15, "fraction": 0.15},
        "approximately_stable": {"pixels": 65, "fraction": 0.65},
    }
    observations = [
        {
            "year": year,
            "status": "available",
            "acquisition_date": date,
            "valid_coverage": 0.9,
            "per_year_population": "pixels valid for this observation",
            "per_year_ndvi": {"mean": mean, "median": mean},
        }
        for year, date, mean in (
            (2021, "2021-02-07", 0.40),
            (2022, "2022-02-12", 0.43),
            (2023, "2023-01-08", 0.42),
        )
    ]
    adjacent = []
    for before, after, before_date, after_date, mean in (
        (2021, 2022, "2021-02-07", "2022-02-12", 0.03),
        (2022, 2023, "2022-02-12", "2023-01-08", -0.01),
    ):
        adjacent.append({
            "label": f"{before}-to-{after}",
            "status": "available",
            "before_year": before,
            "after_year": after,
            "before_date": before_date,
            "after_date": after_date,
            "valid_pixels": 100,
            "aoi_pixels": 120,
            "valid_coverage": 100 / 120,
            "threshold": 0.1,
            "categories": categories,
            "mean": mean,
        })
    return {
        "request": {
            "years": [2021, 2022, 2023],
            "aoi": {"bounds_wgs84": list(DEMO_BOUNDS), "area_square_km": 4.878},
            "data_quality_requirements": {"change_threshold": 0.1},
        },
        "annual_observations": observations,
        "available_years": [2021, 2022, 2023],
        "missing_years": [],
        "common_valid_trend": {
            "sufficient": True,
            "coverage": 0.8,
            "pixels": 96,
            "aoi_pixels": 120,
            "population": "pixels valid across every available observation",
            "reason": None,
            "annual_statistics": [
                {
                    "year": item["year"],
                    "acquisition_date": item["acquisition_date"],
                    "mean": item["per_year_ndvi"]["mean"],
                    "median": item["per_year_ndvi"]["median"],
                }
                for item in observations
            ],
        },
        "adjacent_differences": adjacent,
        "first_to_last_difference": {
            "status": "available",
            "before_year": 2021,
            "after_year": 2023,
            "before_date": "2021-02-07",
            "after_date": "2023-01-08",
            "valid_pixels": 98,
            "aoi_pixels": 120,
            "valid_coverage": 98 / 120,
            "threshold": 0.1,
            "categories": categories,
            "mean": 0.02,
        },
        "limitations": ["NDVI does not establish cause."],
        "outputs": {"summary": "/tmp/summary.txt", "execution_record": "/tmp/run.json"},
    }


def valid_answer_plan(schema: dict) -> dict:
    properties = schema["properties"]
    return {
        "answer_kind": properties["answer_kind"]["enum"][0],
        "lead_claim_id": properties["lead_claim_id"]["enum"][0],
        "claim_ids": list(properties["claim_ids"]["items"]["enum"]),
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
        )
        self.assertEqual(interpreted.missing_information, ())
        self.assertFalse(interpreted.uses_confirmed_boundary)
        self.assertIsNone(interpreted.evidence["confirmed_boundary"])
        plan = build_tool_plan(
            interpreted,
            confirm_aoi(DEMO_BOUNDS),
            current_year=2026,
            active_boundary_source="demo_preset",
            active_boundary_label=DEMO_AOI_LABEL,
        )
        self.assertTrue(plan.runnable)
        self.assertEqual(plan.boundary_resolution.source, "application_state")
        self.assertEqual(plan.boundary_resolution.active_aoi_source, "demo_preset")
        self.assertEqual(plan.record()["boundary_context"]["label"], DEMO_AOI_LABEL)

    def test_exact_observed_question_resolves_confirmed_aoi_into_ready_plan(self) -> None:
        provider = FakeProvider(
            interpretation_payload(
                vegetation=True,
                boundary=False,
                start_year=2022,
                end_year=2024,
            )
        )
        interpreted = interpret_request(provider, "compare the vegetation from 2022 to 2024")
        plan = build_tool_plan(
            interpreted,
            confirm_aoi(DEMO_BOUNDS),
            current_year=2026,
            active_boundary_source="demo_preset",
            active_boundary_label=DEMO_AOI_LABEL,
        )
        self.assertFalse(interpreted.uses_confirmed_boundary)
        self.assertEqual(plan.boundary_resolution.missing_information, ())
        self.assertEqual(plan.status, "ready")
        self.assertEqual(plan.historical_request.aoi.bounds, DEMO_BOUNDS)
        self.assertEqual(plan.historical_request.years, (2022, 2023, 2024))
        record = plan.record()
        self.assertEqual(
            record["raw_model_interpretation"]["boundary_reference"],
            "not_explicit_in_source_text",
        )
        self.assertEqual(record["boundary_resolution"]["source"], "application_state")

    def test_reconfirming_same_aoi_invalidates_stale_blocked_plan(self) -> None:
        state = {}
        activate_confirmed_aoi(state, DEMO_BOUNDS, "demo_preset", DEMO_AOI_LABEL)
        state["assistant_plan"] = object()
        state["historical_chat"] = [{"role": "assistant", "content": "blocked"}]
        _confirmed, changed = activate_confirmed_aoi(
            state, DEMO_BOUNDS, "demo_preset", DEMO_AOI_LABEL
        )
        self.assertFalse(changed)
        self.assertNotIn("assistant_plan", state)
        self.assertNotIn("historical_chat", state)

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
        )
        plan = build_tool_plan(
            interpreted,
            confirm_aoi(DEMO_BOUNDS),
            current_year=2026,
            active_boundary_source="demo_preset",
            active_boundary_label=DEMO_AOI_LABEL,
        )
        self.assertIn(
            "named_location_confirmation", plan.boundary_resolution.missing_information
        )
        self.assertFalse(plan.runnable)

    def test_missing_dates_are_clarifications_and_invented_years_are_discarded(self) -> None:
        provider = FakeProvider(
            interpretation_payload(vegetation=True, start_year=2020, end_year=2024)
        )
        interpreted = interpret_request(provider, "Compare vegetation here.")
        self.assertIsNone(interpreted.start_year)
        self.assertIsNone(interpreted.end_year)
        self.assertEqual(interpreted.missing_information, ("start_year", "end_year"))
        self.assertEqual(
            interpreted.runtime["discarded_years_without_source_evidence"], [2020, 2024]
        )

    def test_unconfirmed_aoi_blocks_an_otherwise_valid_vegetation_plan(self) -> None:
        provider = FakeProvider(interpretation_payload(vegetation=True))
        interpreted = interpret_request(provider, "Compare vegetation here from 2021 to 2025.")
        plan = build_tool_plan(interpreted, None, current_year=2026)
        self.assertIn("confirmed_boundary", plan.boundary_resolution.missing_information)
        self.assertEqual(plan.status, "blocked")
        self.assertFalse(plan.runnable)

    def test_unsupported_construction_and_roads_intents_are_preserved(self) -> None:
        provider = FakeProvider(interpretation_payload(construction=True, roads=True))
        interpreted = interpret_request(
            provider,
            "Compare construction and roads here from 2021 to 2025.",
        )
        plan = build_tool_plan(interpreted, confirm_aoi(DEMO_BOUNDS), current_year=2026)
        self.assertEqual(interpreted.analyses, ("construction", "roads"))
        self.assertEqual(plan.status, "blocked")
        self.assertEqual(
            [tool["status"] for tool in plan.tools],
            ["unsupported_in_historical_mode", "unsupported_in_historical_mode"],
        )

    def test_infrastructure_resolves_to_buildings_and_roads_with_dates_and_aoi(self) -> None:
        provider = FakeProvider(interpretation_payload(
            construction=True,
            roads=True,
            start_year=2023,
            end_year=2025,
            boundary=False,
            construction_evidence="infrastructure",
            roads_evidence="infrastructure",
        ))
        interpreted = interpret_request(
            provider,
            "Compare infrastructure changes between 2023 and 2025.",
        )
        plan = build_tool_plan(
            interpreted,
            confirm_aoi(DEMO_BOUNDS),
            current_year=2026,
            active_boundary_source="demo_preset",
            active_boundary_label=DEMO_AOI_LABEL,
        )
        self.assertEqual(("construction", "roads"), interpreted.analyses)
        self.assertEqual((2023, 2025), (interpreted.start_year, interpreted.end_year))
        self.assertEqual("application_state", plan.boundary_resolution.source)
        self.assertEqual(DEMO_BOUNDS, plan.boundary_resolution.bounds_wgs84)
        self.assertEqual("blocked", plan.status)
        self.assertFalse(plan.runnable)
        observations = [tool["structured_observation"] for tool in plan.tools]
        self.assertEqual(["building_change", "road_change"], [item["change_type"] for item in observations])
        self.assertTrue(all(item["requested_years"] == [2023, 2025] for item in observations))
        self.assertTrue(all(item["actual_dates"] == [] for item in observations))
        self.assertTrue(all(item["count"] is None for item in observations))
        summary = summarize_tool_plan(plan)
        self.assertIn("building changes, road changes", summary)
        self.assertIn("2023–2025", summary)
        self.assertIn(DEMO_AOI_LABEL, summary)
        self.assertIn("no suitably dated, aligned high-resolution RGB pair", summary)

    def test_buildings_and_roads_are_preserved_as_separate_intents(self) -> None:
        provider = FakeProvider(interpretation_payload(
            construction=True,
            roads=True,
            construction_evidence="buildings",
            roads_evidence="roads",
        ))
        interpreted = interpret_request(
            provider,
            "Compare buildings and roads here from 2021 to 2025.",
        )
        self.assertEqual(("construction", "roads"), interpreted.analyses)

    def test_vegetation_cover_greenness_and_tree_counts_are_distinct(self) -> None:
        cover = interpret_request(
            FakeProvider(interpretation_payload(vegetation_cover=True)),
            "Compare vegetation cover here from 2021 to 2025.",
        )
        self.assertEqual(("vegetation_cover",), cover.analyses)
        cover_plan = build_tool_plan(cover, confirm_aoi(DEMO_BOUNDS), current_year=2026)
        self.assertFalse(cover_plan.runnable)
        self.assertIn("NDVI greenness is not substituted for cover", cover_plan.tools[0]["reason"])

        trees = interpret_request(
            FakeProvider(interpretation_payload(tree_count=True)),
            "Compare tree count here from 2021 to 2025.",
        )
        self.assertEqual(("tree_count",), trees.analyses)
        tree_plan = build_tool_plan(trees, confirm_aoi(DEMO_BOUNDS), current_year=2026)
        self.assertFalse(tree_plan.runnable)
        self.assertIn("NDVI and green pixels do not provide tree counts", tree_plan.tools[0]["reason"])

        greenness = interpret_request(
            FakeProvider(interpretation_payload(vegetation=True)),
            "Compare vegetation here from 2021 to 2025.",
        )
        self.assertEqual(("vegetation",), greenness.analyses)
        self.assertTrue(build_tool_plan(greenness, confirm_aoi(DEMO_BOUNDS), current_year=2026).runnable)

    def test_combined_request_retains_partial_support(self) -> None:
        provider = FakeProvider(interpretation_payload(vegetation=True, construction=True))
        interpreted = interpret_request(
            provider,
            "Compare vegetation and construction here from 2021 to 2025.",
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
        interpreted = interpret_request(provider, "Compare vegetation here from 2021 to 2025.")
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

    def test_not_run_infrastructure_writes_auditable_empty_observations(self) -> None:
        interpreted = interpret_request(
            FakeProvider(interpretation_payload(
                construction=True,
                roads=True,
                start_year=2023,
                end_year=2025,
                boundary=False,
                construction_evidence="infrastructure",
                roads_evidence="infrastructure",
            )),
            "Compare infrastructure changes between 2023 and 2025.",
        )
        plan = build_tool_plan(interpreted, confirm_aoi(DEMO_BOUNDS), current_year=2026)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, result = execute_tool_plan(
                plan,
                root / "assistant",
                root / "historical",
                root / "cache",
            )
        self.assertIsNone(result)
        self.assertEqual("not_run", record["status"])
        self.assertEqual(2, len(record["structured_observations"]))
        self.assertEqual(
            {"building_change", "road_change"},
            {item["change_type"] for item in record["structured_observations"]},
        )
        self.assertTrue(all(item["actual_dates"] == [] for item in record["structured_observations"]))
        self.assertTrue(all(item["evidence_image_refs"] == [] for item in record["structured_observations"]))

    def test_supported_ndvi_observations_keep_counts_area_and_locations_unclaimed(self) -> None:
        observations = structured_historical_observations(complete_result_fixture())
        self.assertEqual(2, len(observations))
        self.assertEqual(
            ["2021-02-07", "2022-02-12"],
            observations[0]["actual_dates"],
        )
        self.assertIsNone(observations[0]["count"])
        self.assertIsNone(observations[0]["area_square_metres"])
        self.assertIsNone(observations[0]["spatial_location"])
        self.assertEqual("vegetation_greenness_ndvi", observations[0]["change_type"])


class SupportedAnswerTests(unittest.TestCase):
    def test_complete_multi_year_result_is_conversational_and_evidence_bound(self) -> None:
        result = complete_result_fixture()
        answer = generate_supported_answer(
            FakeProvider(responder=valid_answer_plan), result
        )
        self.assertFalse(answer["fallback_used"])
        self.assertIn("satellite-based indicator related to vegetation greenness", answer["text"])
        self.assertIn("7 February 2021", answer["text"])
        self.assertIn("12 February 2022", answer["text"])
        self.assertIn("Available comparisons", answer["text"])
        self.assertNotIn("plant health improved", answer["text"])

    def test_missing_year_is_natural_and_is_not_bridged_as_adjacent(self) -> None:
        result = result_fixture()
        answer = generate_supported_answer(
            FakeProvider(responder=valid_answer_plan), result
        )
        self.assertIn("No suitable observation", answer["text"])
        self.assertIn("does not bridge", answer["text"])
        self.assertIn("2021 to 2022", answer["text"])
        self.assertNotIn("2021 to 2023", answer["text"])

    def test_insufficient_common_coverage_withholds_comparable_trend(self) -> None:
        answer = generate_supported_answer(
            FakeProvider(responder=valid_answer_plan), result_fixture()
        )
        self.assertIn("could not report a directly comparable multi-year NDVI trend", answer["text"])
        self.assertIn("different pixel populations", answer["text"])

    def test_small_average_with_mixed_local_categories_is_not_called_stable(self) -> None:
        result = complete_result_fixture()
        result["adjacent_differences"][0]["mean"] = 0.0001
        result["adjacent_differences"][0]["categories"] = {
            "increased": {"pixels": 40, "fraction": 0.40},
            "decreased": {"pixels": 35, "fraction": 0.35},
            "approximately_stable": {"pixels": 25, "fraction": 0.25},
        }
        answer = generate_supported_answer(
            FakeProvider(responder=valid_answer_plan), result
        )
        self.assertIn("mean NDVI difference was +0.0001", answer["text"])
        self.assertIn("The map was mixed", answer["text"])
        self.assertIn("40.0% increased", answer["text"])
        self.assertNotIn("the area was stable", answer["text"].lower())

    def test_combined_request_reports_unsupported_analysis_without_inventing_result(self) -> None:
        execution = {
            "tool_executions": [
                {"analysis": "vegetation", "status": "success"},
                {
                    "analysis": "construction",
                    "status": "not_run",
                    "reason": "requires aligned RGB inputs",
                },
            ]
        }
        answer = generate_supported_answer(
            FakeProvider(responder=valid_answer_plan),
            complete_result_fixture(),
            execution,
        )
        self.assertIn("building-change analysis was not run", answer["text"])
        self.assertNotIn("construction change was found", answer["text"])

    def test_invented_numbers_or_dates_reject_answer_plan(self) -> None:
        def response(schema):
            value = valid_answer_plan(schema)
            value["invented_date"] = "2099-01-01"
            value["invented_number"] = 99.9
            return value

        answer = generate_supported_answer(
            FakeProvider(responder=response), complete_result_fixture()
        )
        self.assertTrue(answer["fallback_used"])
        self.assertIn("unexpected fields", answer["fallback_reason"])
        self.assertNotIn("2099", answer["text"])
        self.assertNotIn("99.9", answer["text"])

    def test_unavailable_model_uses_readable_deterministic_fallback(self) -> None:
        def unavailable(_schema):
            raise LanguageModelUnavailableError("offline")

        answer = generate_supported_answer(
            FakeProvider(responder=unavailable), complete_result_fixture()
        )
        self.assertTrue(answer["fallback_used"])
        self.assertIn("satellite-based indicator related to vegetation greenness", answer["text"])
        self.assertNotIn("{", answer["text"])
        self.assertIn("offline", answer["fallback_reason"])

    def test_every_supported_claim_uses_checked_renderer(self) -> None:
        result = complete_result_fixture()
        answer = generate_supported_answer(
            FakeProvider(responder=valid_answer_plan), result
        )
        for fact in supported_facts(result).values():
            self.assertIn(fact, answer["text"])

    def test_attached_answer_is_downloadable_and_machine_record_is_preserved(self) -> None:
        answer = generate_supported_answer(
            FakeProvider(responder=valid_answer_plan), complete_result_fixture()
        )
        with tempfile.TemporaryDirectory() as temporary:
            record_path = Path(temporary) / "assistant.json"
            record_path.write_text("{}\n", encoding="utf-8")
            record = {"outputs": {"assistant_record": str(record_path)}}
            updated = attach_final_answer(record, answer)
            answer_path = Path(updated["outputs"]["answer_text"])
            self.assertTrue(answer_path.is_file())
            self.assertEqual(answer_path.read_text().strip(), answer["text"])
            saved = json.loads(record_path.read_text())
            self.assertEqual(saved["final_answer"]["text"], answer["text"])
            self.assertEqual(
                saved["final_answer"]["model_claim_plan"], answer["model_claim_plan"]
            )
            self.assertEqual(saved["outputs"]["answer_text"], str(answer_path.resolve()))


if __name__ == "__main__":
    unittest.main()
