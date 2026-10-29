from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.infrastructure import (  # noqa: E402
    InfrastructureInterpretation,
    infrastructure_request_fingerprint,
    interpret_infrastructure_request,
    missing_image_message,
    render_infrastructure_answer,
    validate_acquisition_dates,
    write_infrastructure_audit,
)
from satquery.language import (  # noqa: E402
    LanguageModelValidationError,
    ProviderInfo,
)


class FakeProvider:
    def __init__(self, payload: dict) -> None:
        self.info = ProviderInfo(
            True,
            "ollama",
            "http://127.0.0.1:11434",
            "test-model",
            ("test-model",),
        )
        self.payload = payload

    def structured_chat(self, _messages, _schema, **_kwargs):
        return self.payload, {"model": "test-model", "eval_count": 10}


def payload(
    *,
    buildings: bool = False,
    roads: bool = False,
    general: bool = False,
    count: bool = False,
    spatial: bool = False,
) -> dict:
    return {
        "building_changes_requested": buildings,
        "road_changes_requested": roads,
        "general_change_requested": general,
        "instance_count_requested": count,
        "spatial_description_requested": spatial,
        "building_evidence": "infrastructure" if buildings else None,
        "road_evidence": "infrastructure" if roads else None,
        "general_change_evidence": "what changed" if general else None,
        "instance_count_evidence": "how many" if count else None,
        "spatial_description_evidence": "where" if spatial else None,
    }


def interpretation(*, count: bool = False, spatial: bool = False) -> InfrastructureInterpretation:
    return InfrastructureInterpretation(
        original_request="How many new buildings and where?" if count else "What infrastructure changed?",
        requested_classes=("building_change",) if count else ("building_change", "road_change"),
        general_change_requested=False,
        instance_count_requested=count,
        spatial_description_requested=spatial,
        evidence={},
        provider="ollama",
        model="test-model",
        runtime={},
    )


def analysis_record(root: Path) -> dict:
    before = root / "before.png"
    after = root / "after.png"
    overlay = root / "overlay.png"
    mask = root / "mask.png"
    caption = root / "caption.txt"
    execution = root / "run.json"
    for path in (before, after, overlay, mask):
        path.write_bytes(b"png")
    caption.write_text("some houses are built\n", encoding="utf-8")
    record = {
        "status": "success",
        "inputs": {
            "before": {"path": str(before)},
            "after": {"path": str(after)},
        },
        "model_outputs": {
            "mask_class_distribution": {
                "0": {"pixels": 80, "fraction": 0.80},
                "1": {"pixels": 5, "fraction": 0.05},
                "2": {"pixels": 15, "fraction": 0.15},
            },
            "caption": {"text": "some houses are built"},
        },
        "outputs": {
            "overlay": str(overlay),
            "color_mask_with_legend": str(mask),
            "caption": str(caption),
            "execution_record": str(execution),
        },
    }
    execution.write_text(json.dumps(record), encoding="utf-8")
    return record


class InfrastructureInterpretationTests(unittest.TestCase):
    def test_infrastructure_maps_to_buildings_and_roads(self) -> None:
        result = interpret_infrastructure_request(
            FakeProvider(payload(buildings=True, roads=True)),
            "What infrastructure changed?",
        )
        self.assertEqual(("building_change", "road_change"), result.requested_classes)
        self.assertFalse(result.instance_count_requested)

    def test_source_evidence_is_required(self) -> None:
        value = payload(buildings=True)
        value["building_evidence"] = "invented phrase"
        with self.assertRaisesRegex(LanguageModelValidationError, "not present"):
            interpret_infrastructure_request(FakeProvider(value), "Show infrastructure changes")

    def test_missing_image_requests_the_other_half(self) -> None:
        self.assertIn("after image", missing_image_message(True, False) or "")
        self.assertIn("before image", missing_image_message(False, True) or "")
        self.assertIsNone(missing_image_message(True, True))
        self.assertIsNone(missing_image_message(False, False))

    def test_optional_dates_are_validated_without_invention(self) -> None:
        self.assertEqual((None, None), validate_acquisition_dates("", ""))
        self.assertEqual(
            ("2023-01-01", "2025-01-01"),
            validate_acquisition_dates("2023-01-01", "2025-01-01"),
        )
        with self.assertRaisesRegex(LanguageModelValidationError, "must not be later"):
            validate_acquisition_dates("2025-01-01", "2023-01-01")

    def test_fingerprint_changes_only_when_request_inputs_change(self) -> None:
        first = infrastructure_request_fingerprint("question", b"a", b"b", "", "", False)
        same = infrastructure_request_fingerprint("question", b"a", b"b", "", "", False)
        changed = infrastructure_request_fingerprint("new question", b"a", b"b", "", "", False)
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)


class InfrastructureAnswerTests(unittest.TestCase):
    def test_count_question_never_turns_mask_regions_into_building_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = analysis_record(Path(temporary))
            answer = render_infrastructure_answer(
                interpretation(count=True),
                record,
                "2023-01-01",
                "2025-01-01",
            )
        self.assertIn("cannot provide a validated count", answer)
        self.assertIn("marks change pixels", answer)
        self.assertNotIn("15 buildings", answer)
        self.assertNotIn("new buildings were detected", answer)

    def test_road_and_spatial_claims_remain_limited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = analysis_record(Path(temporary))
            answer = render_infrastructure_answer(
                interpretation(spatial=True),
                record,
                None,
                None,
            )
        self.assertIn("building change", answer)
        self.assertIn("road change", answer)
        self.assertIn("do not by themselves establish", answer)
        self.assertIn("does not turn them into a verified named location", answer)

    def test_audit_preserves_answer_and_original_mci_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = analysis_record(Path(temporary))
            audit = write_infrastructure_audit(
                interpretation(),
                record,
                None,
                None,
            )
            answer_path = Path(audit["outputs"]["answer_text"])
            audit_path = Path(audit["outputs"]["assistant_record"])
            original_path = Path(audit["outputs"]["analysis_record"])
            self.assertTrue(answer_path.is_file())
            self.assertTrue(audit_path.is_file())
            self.assertTrue(original_path.is_file())
            saved = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["answer"]["text"], audit["answer"]["text"])


if __name__ == "__main__":
    unittest.main()
