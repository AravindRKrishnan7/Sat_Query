"""Illustrated PDF reports built only from persisted GeoLens results."""

from __future__ import annotations

import hashlib
import json
import re
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#0B1F33")
TEAL = colors.HexColor("#0F766E")
PALE_TEAL = colors.HexColor("#E6F5F3")
SLATE = colors.HexColor("#475569")
LIGHT_LINE = colors.HexColor("#D9E4EA")
LIGHT_BG = colors.HexColor("#F6F9FB")


class ReportBuildError(ValueError):
    """Raised when a saved result cannot support an evidence-linked report."""


def _plain(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
    text = text.replace("×", "x").replace("→", "to").replace("←", "from")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("`", "")
    return text.strip()


def _paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(_plain(text)).replace("\n", "<br/>"), style)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_path(value: Any, label: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise ReportBuildError(f"saved {label} is unavailable: {path}")
    return path


def _verified_image(reference: Mapping[str, Any], label: str) -> Path:
    path = _existing_path(reference.get("path"), label)
    expected = str(reference.get("sha256", ""))
    if not expected or _sha256(path) != expected:
        raise ReportBuildError(f"saved {label} does not match its recorded SHA-256")
    return path


def _image(path: Path, *, max_width: float, max_height: float) -> Image:
    with PillowImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "GeoLensTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=29,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=4 * mm,
        ),
        "subtitle": ParagraphStyle(
            "GeoLensSubtitle",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=15,
            textColor=SLATE,
            spaceAfter=5 * mm,
        ),
        "heading": ParagraphStyle(
            "GeoLensHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=NAVY,
            spaceBefore=5 * mm,
            spaceAfter=3 * mm,
        ),
        "subheading": ParagraphStyle(
            "GeoLensSubheading",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=TEAL,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "GeoLensBody",
            parent=base["BodyText"],
            fontSize=9.5,
            leading=14,
            textColor=NAVY,
            spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "GeoLensSmall",
            parent=base["BodyText"],
            fontSize=7.8,
            leading=10.5,
            textColor=SLATE,
            spaceAfter=1.5 * mm,
        ),
        "caption": ParagraphStyle(
            "GeoLensCaption",
            parent=base["BodyText"],
            fontSize=7.8,
            leading=10.5,
            textColor=SLATE,
            alignment=TA_CENTER,
            spaceBefore=1.5 * mm,
        ),
        "callout": ParagraphStyle(
            "GeoLensCallout",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=15.5,
            textColor=NAVY,
            leftIndent=2 * mm,
            rightIndent=2 * mm,
        ),
    }


def _header_footer(canvas: Any, document: SimpleDocTemplate, analysis_id: str) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(LIGHT_LINE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(NAVY)
    canvas.drawString(18 * mm, height - 10.5 * mm, "GeoLens evidence report")
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(SLATE)
    canvas.drawRightString(width - 18 * mm, height - 10.5 * mm, analysis_id)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.drawString(18 * mm, 8.5 * mm, "Generated locally from saved analysis records")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def _document(destination: Path, analysis_id: str) -> SimpleDocTemplate:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=f"GeoLens report - {analysis_id}",
        author="GeoLens",
        subject="Evidence-linked satellite analysis report",
    )


def _fact_table(rows: Sequence[tuple[str, Any]], styles: Mapping[str, ParagraphStyle]) -> Table:
    data = [
        [_paragraph(label, styles["small"]), _paragraph(value, styles["small"])]
        for label, value in rows
    ]
    table = Table(data, colWidths=[42 * mm, 116 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
                ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, LIGHT_LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _callout(text: str, styles: Mapping[str, ParagraphStyle]) -> Table:
    table = Table([[_paragraph(text, styles["callout"])]], colWidths=[158 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL),
                ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _image_pair(
    left_path: Path,
    left_caption: str,
    right_path: Path,
    right_caption: str,
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    cells = [
        [
            _image(left_path, max_width=76 * mm, max_height=57 * mm),
            _image(right_path, max_width=76 * mm, max_height=57 * mm),
        ],
        [
            _paragraph(left_caption, styles["caption"]),
            _paragraph(right_caption, styles["caption"]),
        ],
    ]
    table = Table(cells, colWidths=[79 * mm, 79 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _answer_flowables(text: str, styles: Mapping[str, ParagraphStyle]) -> list[Any]:
    flowables: list[Any] = []
    for block in re.split(r"\n\s*\n", _plain(text)):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if all(line.lstrip().startswith("- ") for line in lines):
            for line in lines:
                flowables.append(_paragraph(f"- {line.lstrip()[2:]}", styles["body"]))
        elif len(lines) == 1 and lines[0].startswith("Available comparisons"):
            flowables.append(_paragraph(lines[0], styles["subheading"]))
        else:
            flowables.append(_paragraph(" ".join(lines), styles["body"]))
    return flowables


def _comparison_metric_summary(comparison: Mapping[str, Any]) -> str:
    categories = comparison.get("categories", {})
    increased = categories.get("increased", {}).get("fraction")
    decreased = categories.get("decreased", {}).get("fraction")
    stable = categories.get("approximately_stable", {}).get("fraction")
    if all(value is not None for value in (increased, decreased, stable)):
        return (
            f"Pair-valid coverage {float(comparison.get('valid_coverage', 0.0)):.1%}; "
            f"mean NDVI difference {float(comparison.get('mean', 0.0)):.6f}; "
            f"category threshold +/-{float(comparison.get('threshold', 0.0)):.3f}; "
            f"increased {float(increased):.1%}, decreased {float(decreased):.1%}, "
            f"little difference {float(stable):.1%}; "
            f"pair-valid pixels {int(comparison.get('valid_pixels', 0))}."
        )
    legacy = comparison.get("metrics", comparison.get("statistics", {}))
    return f"Saved pairwise metrics: {legacy}"


def _build(
    destination: Path,
    analysis_id: str,
    story: Sequence[Any],
) -> Path:
    document = _document(destination, analysis_id)
    callback = lambda canvas, doc: _header_footer(canvas, doc, analysis_id)
    document.build(list(story), onFirstPage=callback, onLaterPages=callback)
    if not destination.is_file() or destination.stat().st_size < 1_000:
        raise ReportBuildError("PDF report was not written correctly")
    return destination.resolve()


def build_infrastructure_pdf(
    record: Mapping[str, Any],
    destination: Path,
) -> Path:
    """Build an illustrated infrastructure timeline report without rerunning inference."""
    timeline_path = _existing_path(record.get("outputs", {}).get("timeline_record"), "timeline record")
    analysis_id = timeline_path.parent.name
    observations = list(record.get("observations", []))
    comparisons = list(record.get("comparisons", []))
    if len(observations) < 2 or not comparisons:
        raise ReportBuildError("timeline report requires saved observations and comparisons")
    styles = _styles()
    story: list[Any] = [
        Spacer(1, 5 * mm),
        _paragraph("GeoLens", styles["subtitle"]),
        _paragraph("Buildings and infrastructure timeline", styles["title"]),
        _paragraph(
            "An illustrated record of the uploaded observations and the model predictions saved "
            "for each requested comparison.",
            styles["subtitle"],
        ),
        _fact_table(
            (
                ("Analysis ID", analysis_id),
                ("Status", record.get("status", "unknown")),
                ("Question", record.get("question", "")),
                ("Created", record.get("created_at_utc", "unknown")),
                ("Execution record", timeline_path),
            ),
            styles,
        ),
        _paragraph("Result summary", styles["heading"]),
    ]
    explicitly_synthetic = "synthetic" in str(record.get("question", "")).casefold() or any(
        "synthetic" in str(item.get("original_upload", {}).get("filename", "")).casefold()
        for item in observations
    )
    if explicitly_synthetic:
        story.append(
            _callout(
                "This sequence was explicitly supplied as synthetic. Its dates are artificial "
                "demonstration labels, not verified acquisition dates.",
                styles,
            )
        )
    answer_path = _existing_path(record.get("outputs", {}).get("timeline_answer"), "timeline answer")
    story.extend(_answer_flowables(answer_path.read_text(encoding="utf-8"), styles))
    story.append(
        _callout(
            "These are model predictions for uploaded, user-confirmed aligned image pairs. "
            "They are not verified real-world observations. Highlighted regions are not building "
            "counts, and the separate caption and change mask may disagree.",
            styles,
        )
    )
    story.extend([PageBreak(), _paragraph("Uploaded observations", styles["heading"])])
    for observation in observations:
        original = observation["original_upload"]
        prepared = observation["prepared_file"]
        prepared_path = _verified_image(prepared, f"prepared observation {observation['id']}")
        story.append(
            KeepTogether(
                [
                    _paragraph(
                        f"{observation['chronological_order']}. {observation['acquisition_date']}",
                        styles["subheading"],
                    ),
                    _image(prepared_path, max_width=83 * mm, max_height=57 * mm),
                    _paragraph(
                        f"{original['filename']} | original SHA-256 {original['sha256']} | "
                        f"prepared SHA-256 {prepared['sha256']} | "
                        f"{observation['preparation']['resampling']}",
                        styles["small"],
                    ),
                    _paragraph(
                        "The acquisition date was supplied by the user and was not verified from "
                        "image metadata.",
                        styles["small"],
                    ),
                ]
            )
        )

    observation_by_id = {item["id"]: item for item in observations}
    for comparison in comparisons:
        story.extend([PageBreak(), _paragraph("Comparison evidence", styles["heading"])])
        interval = f"{comparison['before_date']} to {comparison['after_date']}"
        kind = "Independent endpoint comparison" if comparison.get("kind") == "endpoint" else "Consecutive comparison"
        story.append(_paragraph(f"{kind}: {interval}", styles["subheading"]))
        if comparison.get("status") != "success":
            story.append(
                _callout(
                    f"This comparison failed and no finding is reported: {comparison.get('error', 'unknown error')}",
                    styles,
                )
            )
            continue
        before = observation_by_id[comparison["before_observation"]]
        after = observation_by_id[comparison["after_observation"]]
        before_path = _verified_image(before["prepared_file"], "comparison before input")
        after_path = _verified_image(after["prepared_file"], "comparison after input")
        execution_path = _existing_path(comparison.get("execution_record"), "comparison execution record")
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if execution.get("inputs", {}).get("before", {}).get("sha256") != before["prepared_file"]["sha256"]:
            raise ReportBuildError(f"comparison {comparison['id']} before identity is inconsistent")
        if execution.get("inputs", {}).get("after", {}).get("sha256") != after["prepared_file"]["sha256"]:
            raise ReportBuildError(f"comparison {comparison['id']} after identity is inconsistent")
        story.append(
            _image_pair(
                before_path,
                f"Before: {comparison['before_date']}",
                after_path,
                f"After: {comparison['after_date']}",
                styles,
            )
        )
        story.extend([Spacer(1, 2 * mm), _callout(str(comparison.get("answer", "")), styles)])
        overlay = _existing_path(comparison.get("overlay"), "comparison overlay")
        story.append(Spacer(1, 3 * mm))
        story.append(_image(overlay, max_width=145 * mm, max_height=72 * mm))
        story.append(
            _paragraph(
                "Prediction overlay. Red: predicted building changes. Yellow: predicted road changes.",
                styles["caption"],
            )
        )
        caption = execution.get("model_outputs", {}).get("caption", {}).get("text") or "[empty]"
        distribution = execution.get("model_outputs", {}).get("mask_class_distribution", {})
        artifacts = execution.get("artifacts", {})
        checkpoint = artifacts.get("checkpoint", {})
        model_identity = (
            f"Change-Agent MCI checkpoint SHA-256 {checkpoint.get('sha256', 'not recorded')}; "
            f"source revision {artifacts.get('change_agent_revision_actual', 'not recorded')}; "
            f"device {execution.get('runtime', {}).get('selected_device', 'not recorded')}"
        )
        story.append(_paragraph("Saved evidence", styles["subheading"]))
        story.append(
            _fact_table(
                (
                    ("Raw model caption", caption),
                    ("Mask class populations", distribution),
                    ("Model identity", model_identity),
                    ("Input identity", "verified against prepared paths and SHA-256 values"),
                    ("Execution record", execution_path),
                ),
                styles,
            )
        )
    story.extend(
        [
            PageBreak(),
            _paragraph("Method and limitations", styles["heading"]),
            _paragraph(
                "GeoLens ran the existing two-image Change-Agent MCI checkpoint independently for "
                "each labelled interval. Resizing, when requested, prepared a 256x256 RGB copy; it "
                "did not align images or establish suitable ground resolution.",
                styles["body"],
            ),
            _paragraph(
                "The change mask localises predicted building-change and road-change categories. "
                "The raw caption is a separate model output. Neither output is ground truth, and "
                "the workflow does not count individual buildings, determine construction dates, "
                "or prove that a detected road-region change is road construction.",
                styles["body"],
            ),
            _fact_table(
                (
                    ("Parent record", timeline_path),
                    ("Request fingerprint", record.get("request_fingerprint", "")),
                    ("Endpoint policy", record.get("execution_policy", {})),
                ),
                styles,
            ),
        ]
    )
    return _build(destination.resolve(), analysis_id, story)


def build_historical_pdf(
    record: Mapping[str, Any],
    destination: Path,
    *,
    original_request: str,
    answer: str,
    area_label: str,
) -> Path:
    """Build an illustrated historical-NDVI report from one saved execution."""
    execution_path = _existing_path(record.get("outputs", {}).get("execution_record"), "historical execution record")
    analysis_id = execution_path.parent.name
    observations = list(record.get("annual_observations", []))
    if not observations:
        raise ReportBuildError("historical report requires saved annual observations")
    request = record.get("request", {})
    aoi = request.get("aoi", {})
    styles = _styles()
    story: list[Any] = [
        Spacer(1, 5 * mm),
        _paragraph("GeoLens", styles["subtitle"]),
        _paragraph("Vegetation history", styles["title"]),
        _paragraph(
            "An illustrated report of the saved Sentinel-2 observations and NDVI comparisons.",
            styles["subtitle"],
        ),
        _fact_table(
            (
                ("Analysis ID", analysis_id),
                ("Status", record.get("status", "unknown")),
                ("Question", original_request),
                ("Area", area_label),
                ("WGS84 bounds", aoi.get("bounds_wgs84", [])),
                ("Calculated area", f"{float(aoi.get('area_square_km', 0.0)):.3f} square kilometres"),
                ("Requested years", request.get("years", [])),
                ("Execution record", execution_path),
            ),
            styles,
        ),
        _paragraph("Answer", styles["heading"]),
    ]
    story.extend(_answer_flowables(answer, styles))
    story.append(
        _callout(
            "NDVI is a satellite-based indicator related to vegetation greenness. It does not "
            "establish tree counts, vegetation-covered area, plant health, causes, or statistical "
            "significance.",
            styles,
        )
    )

    available = [item for item in observations if item.get("status") == "available"]
    missing = [item for item in observations if item.get("status") != "available"]
    story.extend([PageBreak(), _paragraph("Annual observations", styles["heading"])])
    for observation in available:
        outputs = observation.get("outputs", {})
        true_color = _existing_path(outputs.get("true_color_png"), "true-colour preview")
        ndvi = _existing_path(outputs.get("ndvi_png"), "NDVI preview")
        story.append(
            KeepTogether(
                [
                    _paragraph(
                        f"{observation['year']} - acquired {observation['acquisition_date']}",
                        styles["subheading"],
                    ),
                    _image_pair(
                        true_color,
                        "True colour (fixed display-only stretch)",
                        ndvi,
                        "NDVI (grey pixels excluded)",
                        styles,
                    ),
                    _paragraph(
                        f"Sentinel item {observation.get('item_id')} | "
                        f"{float(observation.get('valid_coverage', 0.0)):.1%} year-valid coverage | "
                        f"platform {observation.get('platform', 'unknown')}",
                        styles["small"],
                    ),
                ]
            )
        )
    if missing:
        story.append(_paragraph("Unavailable requested observations", styles["subheading"]))
        for observation in missing:
            story.append(
                _paragraph(
                    f"{observation.get('year')}: {observation.get('reason', 'no suitable observation')}",
                    styles["body"],
                )
            )

    comparisons = list(record.get("adjacent_differences", []))
    first_last = record.get("first_to_last_difference")
    if isinstance(first_last, Mapping):
        comparisons.append({**first_last, "comparison_kind": "first_to_last"})
    story.extend([PageBreak(), _paragraph("NDVI comparison maps", styles["heading"])])
    available_comparisons = [item for item in comparisons if item.get("status") == "available"]
    unavailable_comparisons = [item for item in comparisons if item.get("status") != "available"]
    for comparison in available_comparisons:
        kind = "First-to-last" if comparison.get("comparison_kind") == "first_to_last" else "Adjacent"
        difference = _existing_path(
            comparison.get("outputs", {}).get("difference_png"),
            "NDVI difference preview",
        )
        story.append(
            KeepTogether(
                [
                    _paragraph(
                        f"{kind}: {comparison.get('before_date')} to {comparison.get('after_date')}",
                        styles["subheading"],
                    ),
                    _image(difference, max_width=132 * mm, max_height=76 * mm),
                    _paragraph(
                        "Blue: NDVI increased. Red: NDVI decreased. White: little difference. "
                        "Grey: excluded pixels.",
                        styles["caption"],
                    ),
                    _paragraph(
                        _comparison_metric_summary(comparison),
                        styles["small"],
                    ),
                ]
            )
        )
    if unavailable_comparisons:
        story.append(_paragraph("Unavailable intervals", styles["subheading"]))
        for comparison in unavailable_comparisons:
            story.append(
                _paragraph(
                    f"{comparison.get('label')}: {comparison.get('reason', 'comparison unavailable')}",
                    styles["body"],
                )
            )

    provider = record.get("provider", {})
    grid = record.get("grid", {})
    trend = record.get("common_valid_trend", {})
    first_available = available[0] if available else {}
    red_asset = first_available.get("scaling_and_assets", {}).get("red", {})
    cache = record.get("cache", {})
    story.extend(
        [
            PageBreak(),
            _paragraph("Method, provenance and limitations", styles["heading"]),
            _paragraph(
                "GeoLens used the saved Sentinel-2 Level-2A observations and the corrected NDVI "
                "pipeline. Reflectance scaling, quality masking and denominator validity rules "
                "come from the execution record; this export does not rerun retrieval or analysis.",
                styles["body"],
            ),
            _fact_table(
                (
                    (
                        "Provider",
                        f"{provider.get('name', 'unknown')}; collection "
                        f"{provider.get('collection', 'unknown')}",
                    ),
                    (
                        "Analysis grid",
                        f"{grid.get('width', 'unknown')} x {grid.get('height', 'unknown')} pixels; "
                        f"{grid.get('crs', 'unknown')}; {grid.get('resolution_metres', 'unknown')} m; "
                        f"continuous bands {grid.get('continuous_resampling', 'unknown')}; "
                        f"SCL {grid.get('categorical_resampling', 'unknown')}",
                    ),
                    (
                        "Reflectance scaling",
                        f"scale {red_asset.get('applied_scale', 'recorded per acquisition')}; "
                        f"offset {red_asset.get('applied_offset', 'recorded per acquisition')}; "
                        f"{red_asset.get('scale_offset_source', 'see execution record')}",
                    ),
                    ("Available years", record.get("available_years", [])),
                    ("Missing years", record.get("missing_years", [])),
                    (
                        "Common-valid population",
                        f"{int(trend.get('pixels', 0))} pixels; "
                        f"{float(trend.get('coverage', 0.0)):.1%} coverage; "
                        f"included years {trend.get('included_years', [])}; "
                        f"minimum required {float(trend.get('minimum_required_coverage', 0.0)):.1%}",
                    ),
                    (
                        "Cache provenance",
                        f"{cache.get('selected_asset_windows_loaded_from_cache', 0)} selected windows "
                        f"loaded from exact-AOI cache; "
                        f"{cache.get('selected_asset_windows_retrieved', 0)} retrieved",
                    ),
                    ("Execution record", execution_path),
                ),
                styles,
            ),
            Spacer(1, 3 * mm),
            _paragraph(
                "Per-year statistics and multi-observation common-pixel statistics can use "
                "different pixel populations. An unavailable year is not bridged and relabelled "
                "as an adjacent-year comparison. A small average difference does not mean every "
                "part of the area was stable.",
                styles["body"],
            ),
        ]
    )
    return _build(destination.resolve(), analysis_id, story)
