from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.followup import FollowUpContext, answer_follow_up  # noqa: E402
from satquery.language import LanguageModelUnavailableError, ProviderInfo  # noqa: E402


class FailingProvider:
    info = ProviderInfo(True, "ollama", "http://127.0.0.1:11434", "test", ("test",))

    def structured_chat(self, *args, **kwargs):
        raise LanguageModelUnavailableError("offline")


class MisclassifyingProvider:
    info = ProviderInfo(True, "ollama", "http://127.0.0.1:11434", "test", ("test",))

    def structured_chat(self, *args, **kwargs):
        return (
            {
                "intent": "interval_summary",
                "evidence_quote": "How many buildings were added?",
                "start_date_or_year": "null",
                "end_date_or_year": "null",
            },
            {"model": "test"},
        )


def infrastructure_context(root: Path, selected: str | None = "comparison-01") -> FollowUpContext:
    execution = {
        "model_outputs": {
            "mask_class_distribution": {
                "0": {"pixels": 90},
                "1": {"pixels": 0},
                "2": {"pixels": 10},
            }
        }
    }
    execution_path = root / "run.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    record = {
        "comparisons": [
            {
                "id": "comparison-01",
                "status": "success",
                "before_date": "2022-01-01",
                "after_date": "2023-01-01",
                "answer": "The model predicted building changes and no road changes.",
                "execution_record": str(execution_path),
                "overlay": str(root / "overlay.png"),
                "raw_caption": str(root / "caption.txt"),
            },
            {
                "id": "comparison-02",
                "status": "success",
                "before_date": "2023-01-01",
                "after_date": "2024-01-01",
                "answer": "The model did not identify changes.",
                "execution_record": str(execution_path),
                "overlay": str(root / "overlay-2.png"),
                "raw_caption": str(root / "caption-2.txt"),
            },
        ]
    }
    return FollowUpContext(
        workflow="infrastructure_timeline",
        analysis_id="timeline-abc",
        record=record,
        base_answer="Timeline summary",
        selected_comparison_id=selected,
    )


def historical_context() -> FollowUpContext:
    record = {
        "missing_years": [2022],
        "annual_observations": [
            {"year": 2021, "status": "available", "outputs": {"true_color_png": "2021.png"}},
            {"year": 2022, "status": "unavailable", "reason": "insufficient valid coverage"},
            {"year": 2023, "status": "available", "outputs": {"true_color_png": "2023.png"}},
        ],
        "adjacent_differences": [
            {
                "label": "2021-to-2022",
                "status": "unavailable",
                "before_year": 2021,
                "after_year": 2022,
                "reason": "one observation is unavailable",
            },
            {
                "label": "2022-to-2023",
                "status": "unavailable",
                "before_year": 2022,
                "after_year": 2023,
                "reason": "one observation is unavailable",
            },
            {
                "label": "2023-to-2024",
                "status": "available",
                "before_year": 2023,
                "after_year": 2024,
                "before_date": "2023-01-08",
                "after_date": "2024-01-18",
                "mean": -0.01,
                "categories": {
                    "increased": {"pixels": 5},
                    "decreased": {"pixels": 6},
                },
                "outputs": {"difference_png": "difference.png"},
            },
        ],
        "first_to_last_difference": {
            "label": "2021-to-2024-first-last",
            "status": "available",
            "before_year": 2021,
            "after_year": 2024,
            "before_date": "2021-02-07",
            "after_date": "2024-01-18",
            "mean": 0.02,
            "categories": {
                "increased": {"pixels": 8},
                "decreased": {"pixels": 3},
            },
            "outputs": {"difference_png": "first-last.png"},
        },
        "outputs": {"execution_record": "historical.json"},
    }
    return FollowUpContext(
        workflow="historical_vegetation",
        analysis_id="historical-abc",
        record=record,
        base_answer="Saved vegetation answer",
        selected_comparison_id="2023-to-2024",
    )


class GroundedFollowUpTests(unittest.TestCase):
    def test_infrastructure_road_answer_uses_active_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = answer_follow_up(
                "Were any road changes detected?",
                infrastructure_context(Path(temporary)),
            )
        self.assertEqual("timeline-abc", result["analysis_id"])
        self.assertIn("did not predict any road-change", result["answer"])
        self.assertFalse(result["execution_performed"])

    def test_count_and_cause_questions_do_not_invent_findings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = infrastructure_context(Path(temporary))
            count = answer_follow_up("How many buildings were added?", context)
            cause = answer_follow_up("Why did the change happen?", context)
        self.assertIn("does not identify or reliably count", count["answer"])
        self.assertNotRegex(count["answer"], r"\b\d+ buildings\b")
        self.assertIn("does not establish why", cause["answer"])

    def test_ambiguous_interval_requires_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = answer_follow_up(
                "What happened between these dates?",
                infrastructure_context(Path(temporary), selected=None),
            )
        self.assertIn("Choose a comparison interval", result["answer"])

    def test_different_dates_require_reviewed_new_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = answer_follow_up(
                "What happened between 2020 and 2021?",
                infrastructure_context(Path(temporary)),
            )
        self.assertIn("run a new analysis explicitly", result["answer"])

    def test_ollama_failure_uses_honest_deterministic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = answer_follow_up(
                "Were any road changes detected?",
                infrastructure_context(Path(temporary)),
                FailingProvider(),
            )
        self.assertEqual("deterministic_fallback", result["classifier"])
        self.assertIn("offline", result["fallback_reason"])

    def test_explicit_count_overrides_incorrect_model_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = answer_follow_up(
                "How many buildings were added?",
                infrastructure_context(Path(temporary)),
                MisclassifyingProvider(),
            )
        self.assertIn("deterministic_safety_override", result["classifier"])
        self.assertIn("does not identify or reliably count", result["answer"])

    def test_missing_year_and_unavailable_interval_remain_explicit(self) -> None:
        context = historical_context()
        missing = answer_follow_up("Which year has no usable image?", context)
        unavailable = answer_follow_up("What happened between 2021 and 2022?", context)
        self.assertIn("2022", missing["answer"])
        self.assertIn("insufficient valid coverage", missing["answer"])
        self.assertIn("unavailable", unavailable["answer"])
        self.assertIn("No missing year was bridged", unavailable["answer"])

    def test_historical_interval_preserves_population_limitations(self) -> None:
        result = answer_follow_up(
            "What happened between 2023 and 2024?", historical_context()
        )
        self.assertIn("average NDVI was lower", result["answer"])
        self.assertIn("both local increases and decreases", result["answer"])
        self.assertIn("not vegetation area or cause", result["answer"])

    def test_tree_count_and_cover_remain_unsupported(self) -> None:
        context = historical_context()
        count = answer_follow_up("How many trees are there?", context)
        cover = answer_follow_up("Did vegetation cover expand?", context)
        self.assertIn("cannot count", count["answer"])
        self.assertIn("not vegetation-covered area", cover["answer"])


if __name__ == "__main__":
    unittest.main()
