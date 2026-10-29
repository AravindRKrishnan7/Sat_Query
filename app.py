#!/usr/bin/env python3
"""Local Streamlit interface for SatQuery's existing analysis tools."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.change_agent import (  # noqa: E402
    CLASS_NAMES,
    InputValidationError,
    ModelSafetyError,
    load_model,
    run as run_change_agent,
)
from satquery.geolens import (  # noqa: E402
    GeoLensInputError,
    GeoLensResultError,
    MAX_UPLOAD_BYTES,
    create_unique_run_directory,
    prepare_uploaded_image,
    save_prepared_pair,
    validate_result_record,
)
from satquery.followup import FollowUpContext, answer_follow_up  # noqa: E402
from satquery.historical import (  # noqa: E402
    DEFAULT_AREA_LIMIT_SQUARE_KM,
    DEMO_AOI_LABEL,
    DEMO_BOUNDS,
    MAX_ANNUAL_OBSERVATIONS,
    SUPPORTED_FIRST_YEAR,
    ConfirmedAOI,
    HistoricalNetworkError,
    HistoricalRequest,
    HistoricalValidationError,
    bounds_from_geojson,
    activate_confirmed_aoi,
    build_request,
    confirm_aoi,
    run_historical,
    update_aoi_state,
)
from satquery.infrastructure import (  # noqa: E402
    infrastructure_request_fingerprint,
    interpret_infrastructure_request,
    missing_image_message,
    validate_acquisition_dates,
    write_infrastructure_audit,
)
from satquery.infrastructure_timeline import (  # noqa: E402
    InfrastructureTimelineError,
    execute_infrastructure_timeline,
    timeline_request_fingerprint,
    validate_timeline_observations,
)
from satquery.language import (  # noqa: E402
    LanguageModelError,
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
from satquery.presentation import (  # noqa: E402
    PresentationIdentityError,
    comparison_input_paths,
    compose_before_after_reveal,
)
from satquery.vegetation import (  # noqa: E402
    VegetationValidationError,
    load_config,
    run as run_vegetation,
)


CHECKPOINT = PROJECT_ROOT / "weights" / "MCI_model.pth"
CHANGE_SOURCE = PROJECT_ROOT / ".cache" / "upstream" / "change-agent"
VEGETATION_CONFIG = PROJECT_ROOT / "configs" / "vegetation.real-example.json"
GEOLENS_OUTPUTS = PROJECT_ROOT / "outputs" / "geolens"
HISTORICAL_CACHE = PROJECT_ROOT / "data" / "cache" / "geolens-historical"


@st.cache_resource(show_spinner="Loading the verified Change-Agent model on CPU...")
def _cached_change_model(
    source_directory: str,
    checkpoint: str,
    device_name: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    return load_model(Path(source_directory), Path(checkpoint), torch.device(device_name), torch)


def _streamlit_model_loader(source: Path, checkpoint: Path, device: Any, _torch: Any) -> tuple:
    return _cached_change_model(str(source.resolve()), str(checkpoint.resolve()), str(device))


@st.cache_data(ttl=10, show_spinner=False)
def _cached_language_provider() -> ProviderInfo:
    return discover_ollama()


def _download(path: Path, label: str, mime: str, key: str) -> None:
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        key=key,
    )


def _inject_geolens_theme() -> None:
    """Apply a small stable theme layer without targeting generated DOM positions."""
    st.markdown(
        """
        <style>
        :root {
          --gl-navy: #0b1f33;
          --gl-navy-soft: #12304a;
          --gl-teal: #0f766e;
          --gl-teal-bright: #2dd4bf;
          --gl-paper: #f7fafc;
          --gl-muted: #64748b;
          --gl-line: #dbe5ec;
        }
        .stApp { background: var(--gl-paper); color: var(--gl-navy); }
        [data-testid="stSidebar"] { background: var(--gl-navy); }
        [data-testid="stSidebar"] * { color: #edf7f7; }
        [data-testid="stSidebar"] [data-baseweb="radio"] label {
          border-radius: 0.55rem; padding: 0.2rem 0.45rem;
        }
        [data-testid="stSidebar"] [aria-checked="true"] {
          background: rgba(45, 212, 191, 0.16);
        }
        [data-testid="stMainBlockContainer"] {
          max-width: 1180px; padding-top: 1.4rem; padding-bottom: 4rem;
        }
        [data-testid="stAlert"] { border-radius: 0.65rem; }
        [data-testid="stExpander"] {
          border: 1px solid var(--gl-line); border-radius: 0.65rem;
          background: rgba(255,255,255,0.72);
        }
        [data-testid="stChatMessage"] {
          border: 1px solid var(--gl-line); border-radius: 0.75rem;
          background: #ffffff;
        }
        .stButton > button[kind="primary"] {
          background: var(--gl-teal); border-color: var(--gl-teal);
          font-weight: 650;
        }
        .stButton > button[kind="primary"]:hover {
          background: #115e59; border-color: #115e59;
        }
        .geolens-hero {
          display: flex; align-items: center; gap: 0.9rem;
          padding: 0.2rem 0 1.15rem; border-bottom: 1px solid var(--gl-line);
          margin-bottom: 1.25rem;
        }
        .geolens-mark {
          width: 2.45rem; height: 2.45rem; border-radius: 0.72rem;
          display: grid; place-items: center; color: white; font-weight: 800;
          background: linear-gradient(145deg, var(--gl-navy-soft), var(--gl-teal));
        }
        .geolens-hero h1 { font-size: 1.55rem; line-height: 1.2; margin: 0; }
        .geolens-hero p { color: var(--gl-muted); margin: 0.15rem 0 0; }
        .geolens-step {
          color: var(--gl-teal); font-size: 0.78rem; font-weight: 750;
          letter-spacing: 0.06em; text-transform: uppercase; margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _step(label: str) -> None:
    st.markdown(f'<div class="geolens-step">{label}</div>', unsafe_allow_html=True)


def _render_follow_up_chat(context: FollowUpContext) -> None:
    """Render a result-bound conversation without invoking an analysis engine."""
    st.subheader("Ask about this completed analysis")
    st.caption(
        f"Grounded in analysis `{context.analysis_id}`. Follow-ups use saved evidence only and "
        "do not rerun inference or satellite retrieval."
    )
    threads = st.session_state.setdefault("followup_threads", {})
    messages = threads.setdefault(context.analysis_id, [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("classifier"):
                st.caption(f"Grounding: {message['classifier']}")

    with st.form(f"followup-form-{context.analysis_id}", clear_on_submit=True):
        question = st.text_input(
            "Follow-up question",
            placeholder="Example: Were any road changes detected?",
            max_chars=2_000,
        )
        submitted = st.form_submit_button("Ask about this result")
    if submitted:
        if not question.strip():
            st.warning("Enter a follow-up question.")
            return
        provider_info = _cached_language_provider()
        provider = OllamaProvider(provider_info) if provider_info.available else None
        try:
            response = answer_follow_up(question, context, provider)
        except LanguageModelError as exc:
            st.error(f"Follow-up could not be validated: {exc}")
            return
        messages.extend(
            (
                {"role": "user", "content": question, "analysis_id": context.analysis_id},
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "analysis_id": context.analysis_id,
                    "classifier": response["classifier"],
                    "audit": response,
                },
            )
        )
        st.rerun()


def _show_construction_result(
    record: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    try:
        outputs = validate_result_record(
            record,
            (
                "indexed_mask",
                "color_mask_with_legend",
                "overlay",
                "comparison",
                "caption",
                "execution_record",
            ),
        )
    except GeoLensResultError as exc:
        st.error(str(exc))
        return

    st.subheader("GeoLens answer")
    st.markdown(str(audit["answer"]["text"]))

    st.subheader("Uploaded image comparison")
    before_path = Path(record["inputs"]["before"]["path"])
    after_path = Path(record["inputs"]["after"]["path"])
    supplied_dates = audit["user_supplied_acquisition_dates"]
    before_caption = "Before — prepared model input"
    after_caption = "After — prepared model input"
    if supplied_dates.get("before"):
        before_caption += f" · user-supplied date {supplied_dates['before']}"
    if supplied_dates.get("after"):
        after_caption += f" · user-supplied date {supplied_dates['after']}"
    before_column, after_column = st.columns(2)
    before_column.image(str(before_path), caption=before_caption, width="stretch")
    after_column.image(str(after_path), caption=after_caption, width="stretch")

    st.image(
        str(outputs["overlay"]),
        caption="MCI prediction overlay on the prepared after image",
        width="stretch",
    )

    st.markdown(
        "**Overlay legend:** Red: predicted building changes. Yellow: predicted road changes."
    )
    st.caption(
        "The overlay is a model prediction on a caller-supplied aligned pair. It is not a verified "
        "inventory of buildings or roads."
    )

    with st.expander("View evidence and technical details"):
        st.markdown("**How the answer was produced and checked**")
        st.write(
            "The local language model extracted the question intent only. The application combined "
            "the recorded change categories with a conservative assessment of the separate raw "
            "caption. Only explicit building/road addition, removal, or no-change phrases may be "
            "paraphrased; counts, locations, dates and causes are discarded. Category agreement "
            "does not imply that the caption and highlighted regions are spatially consistent."
        )
        st.json(audit.get("answer_validation", {}))
        st.image(
            str(outputs["color_mask_with_legend"]),
            caption="Predicted class mask with embedded legend",
            width="stretch",
        )
        distribution = record["model_outputs"]["mask_class_distribution"]
        fraction_columns = st.columns(3)
        for column, label in zip(fraction_columns, (0, 1, 2)):
            item = distribution[str(label)]
            column.metric(
                f"Predicted pixel fraction — {CLASS_NAMES[label]}",
                f"{item['fraction']:.2%}",
            )
        st.caption(
            "These percentages are fractions of model-output pixels. They are not accuracy, "
            "confidence, real-world area, object counts, or reference-label counts."
        )
        st.markdown("**Raw model caption**")
        st.code(record["model_outputs"]["caption"]["text"] or "[empty]", language=None)
        st.caption(
            "The downloadable raw caption is preserved exactly. The main answer may remove "
            "repetition or conservatively paraphrase it, but the caption remains an unverified "
            "output from a separate model head and may disagree with the change categories."
        )
        download_columns = st.columns(3)
        with download_columns[0]:
            _download(outputs["color_mask_with_legend"], "Download coloured mask", "image/png", "construction-mask")
        with download_columns[1]:
            _download(outputs["overlay"], "Download overlay", "image/png", "construction-overlay")
        with download_columns[2]:
            _download(outputs["caption"], "Download raw caption", "text/plain", "construction-caption")
        record_columns = st.columns(3)
        with record_columns[0]:
            _download(
                Path(audit["outputs"]["answer_text"]),
                "Download answer",
                "text/markdown",
                "infrastructure-answer",
            )
        with record_columns[1]:
            _download(
                outputs["execution_record"],
                "Download MCI execution record",
                "application/json",
                "construction-json",
            )
        with record_columns[2]:
            _download(
                Path(audit["outputs"]["assistant_record"]),
                "Download assistant audit record",
                "application/json",
                "infrastructure-audit",
            )
        st.markdown("**Validated question interpretation**")
        st.json(audit["validated_interpretation"])
        st.markdown("**MCI execution record**")
        st.json(record)


def _legacy_construction_pair_mode() -> None:
    st.header("Buildings and infrastructure")
    st.write(
        "Ask a question and upload both a before image and an after image of the same location. "
        "A change comparison cannot be made from only one image."
    )
    question = st.text_area(
        "What would you like to know about the changes?",
        placeholder="Example: What building and road changes does the model predict?",
        key="infrastructure-question",
        max_chars=2_000,
    )
    date_columns = st.columns(2)
    before_date_text = date_columns[0].text_input(
        "Before acquisition date (optional)",
        placeholder="YYYY-MM-DD",
        key="infrastructure-before-date",
    )
    after_date_text = date_columns[1].text_input(
        "After acquisition date (optional)",
        placeholder="YYYY-MM-DD",
        key="infrastructure-after-date",
    )
    validated_dates: tuple[str | None, str | None] = (None, None)
    date_error: str | None = None
    try:
        validated_dates = validate_acquisition_dates(before_date_text, after_date_text)
    except LanguageModelError as exc:
        date_error = str(exc)
        st.error(date_error)

    provider_info = _cached_language_provider()
    if provider_info.available:
        st.caption(
            f"Question interpretation uses the local {provider_info.identifier} model. "
            "It cannot add findings beyond the MCI outputs."
        )
    else:
        st.warning(f"Local language model unavailable: {provider_info.reason}")
    st.warning(
        "Resizing changes pixel dimensions only. It does not align images, create missing detail, "
        "guarantee suitable ground resolution, or make two locations comparable."
    )

    before_upload, after_upload = st.columns(2)
    before_file = before_upload.file_uploader(
        "Before image", type=("png", "jpg", "jpeg"), key="construction-before"
    )
    after_file = after_upload.file_uploader(
        "After image", type=("png", "jpg", "jpeg"), key="construction-after"
    )
    missing_image = missing_image_message(before_file is not None, after_file is not None)
    if missing_image:
        st.info(missing_image)
    resize = st.checkbox(
        "Create explicit 256×256 prepared copies when dimensions differ (Lanczos resampling)",
        value=False,
    )

    prepared_before = None
    prepared_after = None
    errors: list[str] = []
    if before_file is not None:
        try:
            prepared_before = prepare_uploaded_image(before_file.getvalue(), "before", resize)
        except GeoLensInputError as exc:
            errors.append(str(exc))
    if after_file is not None:
        try:
            prepared_after = prepare_uploaded_image(after_file.getvalue(), "after", resize)
        except GeoLensInputError as exc:
            errors.append(str(exc))
    for message in errors:
        st.error(message)

    if prepared_before and prepared_after:
        preview_before, preview_after = st.columns(2)
        before_label = (
            f"Prepared before: {prepared_before.original_size[0]}×{prepared_before.original_size[1]} "
            "→ 256×256"
            if prepared_before.resized
            else "Before preview: unchanged 256×256 RGB"
        )
        after_label = (
            f"Prepared after: {prepared_after.original_size[0]}×{prepared_after.original_size[1]} "
            "→ 256×256"
            if prepared_after.resized
            else "After preview: unchanged 256×256 RGB"
        )
        preview_before.image(prepared_before.image, caption=before_label, width="stretch")
        preview_after.image(prepared_after.image, caption=after_label, width="stretch")

    aligned = st.checkbox(
        "I confirm these images depict the same location and are already spatially aligned.",
        value=False,
    )
    prerequisites = []
    if not CHECKPOINT.is_file():
        prerequisites.append(f"verified checkpoint is missing: {CHECKPOINT}")
    if not CHANGE_SOURCE.is_dir():
        prerequisites.append(f"prepared Change-Agent source is missing: {CHANGE_SOURCE}")
    for message in prerequisites:
        st.error(message)

    before_bytes = before_file.getvalue() if before_file is not None else None
    after_bytes = after_file.getvalue() if after_file is not None else None
    request_fingerprint = infrastructure_request_fingerprint(
        question,
        before_bytes,
        after_bytes,
        before_date_text,
        after_date_text,
        resize,
    )
    can_analyse = bool(
        question.strip()
        and prepared_before
        and prepared_after
        and aligned
        and not errors
        and not prerequisites
        and date_error is None
        and provider_info.available
    )
    if st.button("Analyse uploaded pair", type="primary", disabled=not can_analyse):
        try:
            with st.status("Interpreting the question and running verified CPU inference...", expanded=True) as status:
                interpretation = interpret_infrastructure_request(
                    OllamaProvider(provider_info),
                    question,
                )
                st.write("Question accepted; validating the uploaded pair and checkpoint")
                run_directory = create_unique_run_directory(
                    GEOLENS_OUTPUTS / "construction",
                    "run",
                )
                before_path, after_path = save_prepared_pair(
                    prepared_before,
                    prepared_after,
                    run_directory,
                )
                args = argparse.Namespace(
                    before=before_path,
                    after=after_path,
                    checkpoint=CHECKPOINT,
                    output_dir=run_directory,
                    device="cpu",
                    source_dir=CHANGE_SOURCE,
                )
                st.write("Validating inputs, checkpoint and upstream source")
                record = run_change_agent(args, model_loader=_streamlit_model_loader)
                audit = write_infrastructure_audit(
                    interpretation,
                    record,
                    validated_dates[0],
                    validated_dates[1],
                )
                status.update(
                    label="Buildings and infrastructure analysis complete",
                    state="complete",
                    expanded=False,
                )
            st.session_state["infrastructure_result"] = {
                "fingerprint": request_fingerprint,
                "analysis": record,
                "audit": audit,
            }
        except (
            FileNotFoundError,
            InputValidationError,
            LanguageModelError,
            ModelSafetyError,
            RuntimeError,
        ) as exc:
            st.error(f"Buildings and infrastructure analysis failed: {exc}")

    retained = st.session_state.get("infrastructure_result")
    if isinstance(retained, Mapping):
        if retained.get("fingerprint") == request_fingerprint:
            record = retained.get("analysis")
            audit = retained.get("audit")
            if isinstance(record, Mapping) and isinstance(audit, Mapping):
                _show_construction_result(record, audit)
        else:
            st.info(
                "The question, images, dates, or preparation setting changed. The previous run "
                "remains on disk but is not shown as a result for these inputs."
            )


def _show_infrastructure_timeline(record: Mapping[str, Any]) -> None:
    answer_path = Path(record["outputs"]["timeline_answer"])
    timeline_path = Path(record["outputs"]["timeline_record"])
    if not answer_path.is_file() or not timeline_path.is_file():
        st.error("The retained timeline is missing its answer or execution record.")
        return

    st.subheader("GeoLens timeline answer")
    st.markdown(answer_path.read_text(encoding="utf-8"))
    if record.get("status") == "partial":
        st.warning("Some comparisons failed. Completed intervals are preserved below.")
    elif record.get("status") == "failed":
        st.error("No timeline comparison completed successfully.")

    st.subheader("Chronological observations")
    observations = list(record["observations"])
    columns = st.columns(len(observations))
    for column, observation in zip(columns, observations, strict=True):
        prepared = observation["prepared_file"]
        original = observation["original_upload"]
        column.image(
            prepared["path"],
            caption=(
                f"{observation['acquisition_date']} · {original['filename']} · "
                f"prepared 256×256 RGB"
            ),
            width="stretch",
        )

    st.subheader("Explore a comparison")
    comparisons = list(record["comparisons"])
    comparison_by_id = {item["id"]: item for item in comparisons}
    selected_id = st.selectbox(
        "Comparison interval",
        [item["id"] for item in comparisons],
        format_func=lambda value: (
            ("Overall first-to-last" if comparison_by_id[value]["kind"] == "endpoint" else "Consecutive")
            + " · "
            + comparison_by_id[value]["before_date"]
            + " → "
            + comparison_by_id[value]["after_date"]
            + (" · unavailable" if comparison_by_id[value]["status"] != "success" else "")
        ),
        key=f"timeline-comparison-{Path(record['outputs']['timeline_record']).parent.name}",
    )
    selected_comparisons = [comparison_by_id[selected_id]]
    observation_by_id = {item["id"]: item for item in observations}
    for comparison in selected_comparisons:
        interval = f"{comparison['before_date']} → {comparison['after_date']}"
        if comparison["kind"] == "endpoint":
            st.markdown(f"### Overall endpoint comparison · {interval}")
            st.caption(
                "This prediction compares only the first and last observations. It is not a sum "
                "of the consecutive results and does not establish continuous change."
            )
        else:
            st.markdown(f"### Consecutive interval · {interval}")
        if comparison["status"] != "success":
            st.error(f"Comparison failed: {comparison.get('error', 'unknown error')}")
            continue

        before_observation = observation_by_id[comparison["before_observation"]]
        after_observation = observation_by_id[comparison["after_observation"]]
        try:
            before_path, after_path = comparison_input_paths(record, comparison["id"])
            divider = st.slider(
                "Before/after divider",
                min_value=0,
                max_value=100,
                value=50,
                format="%d%% before",
                key=f"comparison-divider-{record['request_fingerprint']}-{comparison['id']}",
                help="Left of the divider shows the before image; right shows the after image.",
            )
            reveal = compose_before_after_reveal(before_path, after_path, divider)
            st.image(
                reveal,
                caption=(
                    f"Before {comparison['before_date']} ← divider → "
                    f"After {comparison['after_date']}"
                ),
                width="stretch",
            )
            st.caption(
                "This display uses the exact prepared inputs for the selected interval. The slider "
                "does not establish or improve their alignment."
            )
        except PresentationIdentityError as exc:
            st.error(f"Comparison display identity check failed: {exc}")
            before_path = Path(before_observation["prepared_file"]["path"])
            after_path = Path(after_observation["prepared_file"]["path"])

        if st.toggle(
            "Show side-by-side fallback",
            value=False,
            key=f"comparison-side-by-side-{record['request_fingerprint']}-{comparison['id']}",
        ):
            input_columns = st.columns(2)
            input_columns[0].image(
                str(before_path),
                caption=f"Before · {comparison['before_date']}",
                width="stretch",
            )
            input_columns[1].image(
                str(after_path),
                caption=f"After · {comparison['after_date']}",
                width="stretch",
            )

        if st.toggle(
            "Show prediction overlay",
            value=True,
            key=f"comparison-overlay-{record['request_fingerprint']}-{comparison['id']}",
        ):
            st.image(
                comparison["overlay"],
                caption=f"Prediction overlay for {interval}",
                width="stretch",
            )
            st.markdown(
                "**Overlay legend:** Red: predicted building changes. Yellow: predicted road changes."
            )
            st.caption(
                "Each overlay is an independent model prediction for its labelled interval, not a "
                "verified inventory or a count of buildings or roads."
            )
        with st.expander(f"View evidence and technical details · {interval}"):
            execution_path = Path(comparison["execution_record"])
            assistant_path = Path(comparison["assistant_record"])
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            assistant = json.loads(assistant_path.read_text(encoding="utf-8"))
            st.markdown("**Raw model caption**")
            st.code(execution["model_outputs"]["caption"]["text"] or "[empty]", language=None)
            st.markdown("**Predicted mask classes and pixel populations**")
            st.json(execution["model_outputs"]["mask_class_distribution"])
            st.markdown("**Input identity and timings**")
            st.json(
                {
                    "input_identity_verified": comparison["input_identity_verified"],
                    "inputs": execution["inputs"],
                    "timing_seconds": execution["runtime"]["timing_seconds"],
                    "answer_validation": assistant.get("answer_validation"),
                }
            )
            download_columns = st.columns(3)
            key = comparison["id"]
            with download_columns[0]:
                _download(
                    Path(comparison["overlay"]),
                    "Download overlay",
                    "image/png",
                    f"{key}-overlay",
                )
            with download_columns[1]:
                _download(
                    Path(comparison["class_mask"]),
                    "Download coloured mask",
                    "image/png",
                    f"{key}-mask",
                )
            with download_columns[2]:
                _download(
                    Path(comparison["raw_caption"]),
                    "Download raw caption",
                    "text/plain",
                    f"{key}-caption",
                )
            record_columns = st.columns(3)
            with record_columns[0]:
                _download(
                    Path(comparison["answer_file"]),
                    "Download interval answer",
                    "text/markdown",
                    f"{key}-answer",
                )
            with record_columns[1]:
                _download(
                    execution_path,
                    "Download execution record",
                    "application/json",
                    f"{key}-execution",
                )
            with record_columns[2]:
                _download(
                    assistant_path,
                    "Download assistant record",
                    "application/json",
                    f"{key}-assistant",
                )

    download_columns = st.columns(2)
    with download_columns[0]:
        _download(
            answer_path,
            "Download timeline answer",
            "text/markdown",
            "timeline-answer",
        )
    with download_columns[1]:
        _download(
            timeline_path,
            "Download timeline execution record",
            "application/json",
            "timeline-record",
        )
    with st.expander("View timeline provenance and technical details"):
        st.json(record)

    _render_follow_up_chat(
        FollowUpContext(
            workflow="infrastructure_timeline",
            analysis_id=Path(record["outputs"]["timeline_record"]).parent.name,
            record=record,
            base_answer=answer_path.read_text(encoding="utf-8"),
            selected_comparison_id=selected_id,
        )
    )


def _construction_mode() -> None:
    st.header("Buildings and infrastructure")
    st.write(
        "Ask a question and upload 2–4 dated images of the same location. GeoLens compares each "
        "consecutive pair with the existing change model."
    )
    _step("1 · Select inputs")
    question = st.text_area(
        "What would you like to know about the changes?",
        placeholder="Example: What building and road changes does the model predict over time?",
        key="infrastructure-timeline-question",
        max_chars=2_000,
    )
    provider_info = _cached_language_provider()
    if provider_info.available:
        st.caption(
            f"Question interpretation uses the local {provider_info.identifier} model. "
            "It cannot add findings beyond the interval outputs."
        )
    else:
        st.warning(f"Local language model unavailable: {provider_info.reason}")

    files = st.file_uploader(
        "Timeline images",
        type=("png", "jpg", "jpeg"),
        accept_multiple_files=True,
        key="construction-timeline-images",
        help="Upload 2–4 observations. Dates entered below determine chronological order.",
    )
    files = list(files or [])
    if len(files) < 2:
        st.info(
            f"Add {2 - len(files)} more image{'s' if len(files) != 1 else ''} "
            "to run a timeline."
        )
    elif len(files) > 4:
        st.error("Upload no more than four images.")

    st.warning(
        "The model requires 256×256 RGB inputs. Resizing changes pixel dimensions only; it does "
        "not register images, establish alignment, create detail, or guarantee comparable ground "
        "resolution."
    )
    resize = st.checkbox(
        "Create explicit 256×256 prepared copies when dimensions differ (Lanczos resampling)",
        value=False,
        key="construction-timeline-resize",
    )
    date_values: list[str] = []
    for index, uploaded in enumerate(files, start=1):
        date_values.append(
            st.text_input(
                f"Acquisition date for image {index}: {uploaded.name}",
                placeholder="YYYY-MM-DD",
                key=f"construction-timeline-date-{index}",
            )
        )

    prepared_images = []
    preparation_errors: list[str] = []
    for index, uploaded in enumerate(files, start=1):
        try:
            prepared_images.append(
                prepare_uploaded_image(uploaded.getvalue(), f"image {index}", resize)
            )
        except GeoLensInputError as exc:
            preparation_errors.append(str(exc))
    for message in preparation_errors:
        st.error(message)

    observations = None
    observation_error: str | None = None
    if len(files) >= 2 and not preparation_errors:
        try:
            observations = validate_timeline_observations(
                [item.name for item in files], prepared_images, date_values
            )
        except InfrastructureTimelineError as exc:
            observation_error = str(exc)
            st.error(observation_error)

    if observations:
        _step("2 · Review chronological order")
        st.markdown("**Chronological order used for analysis**")
        preview_columns = st.columns(len(observations))
        for column, observation in zip(preview_columns, observations, strict=True):
            preparation = (
                f"{observation.prepared.original_size[0]}×"
                f"{observation.prepared.original_size[1]} → 256×256"
                if observation.prepared.resized
                else "unchanged 256×256 RGB"
            )
            column.image(
                observation.prepared.image,
                caption=(
                    f"{observation.acquisition_date.isoformat()} · {observation.filename} · "
                    f"{preparation}"
                ),
                width="stretch",
            )

    include_endpoint = False
    if observations and len(observations) > 2:
        include_endpoint = st.checkbox(
            "Also run a separate first-to-last endpoint comparison",
            value=False,
            key="construction-timeline-endpoint",
            help="This is an independent model run, not a sum of consecutive predictions.",
        )
    aligned = st.checkbox(
        "I confirm all uploaded images depict the same extent and are already spatially aligned.",
        value=False,
        key="construction-timeline-aligned",
    )

    prerequisites: list[str] = []
    if not CHECKPOINT.is_file():
        prerequisites.append(f"verified checkpoint is missing: {CHECKPOINT}")
    if not CHANGE_SOURCE.is_dir():
        prerequisites.append(f"prepared Change-Agent source is missing: {CHANGE_SOURCE}")
    for message in prerequisites:
        st.error(message)

    upload_bytes = [item.getvalue() for item in files]
    request_fingerprint = timeline_request_fingerprint(
        question,
        upload_bytes,
        date_values,
        resize,
        include_endpoint,
    )
    can_analyse = bool(
        question.strip()
        and observations
        and aligned
        and not preparation_errors
        and observation_error is None
        and not prerequisites
        and provider_info.available
    )
    _step("3 · Analyse")
    if st.button("Run timeline analysis", type="primary", disabled=not can_analyse):
        try:
            with st.status(
                "Interpreting the question and comparing the timeline...", expanded=True
            ) as status:
                interpretation = interpret_infrastructure_request(
                    OllamaProvider(provider_info), question
                )
                parent = create_unique_run_directory(
                    GEOLENS_OUTPUTS / "construction-timelines", "timeline"
                )
                comparison_number = 0

                def run_comparison(
                    before_path: Path, after_path: Path, output_directory: Path
                ) -> Mapping[str, Any]:
                    nonlocal comparison_number
                    comparison_number += 1
                    st.write(f"Running independent comparison {comparison_number}")
                    args = argparse.Namespace(
                        before=before_path,
                        after=after_path,
                        checkpoint=CHECKPOINT,
                        output_dir=output_directory,
                        device="cpu",
                        source_dir=CHANGE_SOURCE,
                    )
                    return run_change_agent(args, model_loader=_streamlit_model_loader)

                timeline = execute_infrastructure_timeline(
                    parent_directory=parent,
                    question=question.strip(),
                    fingerprint=request_fingerprint,
                    interpretation=interpretation,
                    observations=observations,
                    resize_requested=resize,
                    include_endpoint=include_endpoint,
                    comparison_runner=run_comparison,
                )
                status.update(
                    label=(
                        "Timeline analysis complete"
                        if timeline["status"] == "success"
                        else "Timeline analysis finished with comparison failures"
                    ),
                    state="complete" if timeline["status"] == "success" else "error",
                    expanded=False,
                )
            st.session_state["infrastructure_timeline_result"] = timeline
        except (InfrastructureTimelineError, LanguageModelError, RuntimeError) as exc:
            st.error(f"Buildings and infrastructure timeline failed: {exc}")

    retained = st.session_state.get("infrastructure_timeline_result")
    if isinstance(retained, Mapping):
        if retained.get("request_fingerprint") == request_fingerprint:
            _step("4 · Explore results")
            _show_infrastructure_timeline(retained)
        else:
            st.info(
                "The question, an image, a date, endpoint option, or preparation setting changed. "
                "The previous timeline remains on disk but is not shown for these inputs."
            )


def _prepared_vegetation_inputs() -> tuple[dict[str, Any] | None, list[Path]]:
    try:
        config, base = load_config(VEGETATION_CONFIG)
    except VegetationValidationError:
        return None, [VEGETATION_CONFIG]
    missing: list[Path] = []
    for role in ("before", "after"):
        item = config[role]
        specs = [item["red"], item["nir"], item["quality"]]
        true_color = item.get("true_color", {})
        specs.extend(value for value in true_color.values() if isinstance(value, Mapping))
        for spec in specs:
            path = Path(spec["path"])
            resolved = (base / path).resolve() if not path.is_absolute() else path.resolve()
            if not resolved.is_file():
                missing.append(resolved)
    return config, missing


def _show_provenance(record: Mapping[str, Any]) -> None:
    with st.expander("Acquisition provenance"):
        for role in ("before", "after"):
            item = record["inputs"][role]
            source = item["source"]
            st.markdown(f"**{role.title()} — {item['date']}**")
            st.write(
                {
                    "provider": source.get("provider"),
                    "collection": source.get("collection"),
                    "item_id": source.get("item_id"),
                    "product_uri": source.get("product_uri"),
                    "processing_baseline": source.get("processing_baseline"),
                    "tile": source.get("tile"),
                    "stac_item_url": source.get("stac_item_url"),
                }
            )


def _show_vegetation_result(record: Mapping[str, Any]) -> None:
    try:
        outputs = validate_result_record(
            record,
            ("comparison_png", "execution_record", "summary", "before_ndvi", "after_ndvi", "delta_ndvi"),
        )
    except GeoLensResultError as exc:
        st.error(str(exc))
        return

    st.subheader("Latest prepared-example result")
    date_columns = st.columns(2)
    date_columns[0].metric("Before acquisition", record["dates"]["before"])
    date_columns[1].metric("After acquisition", record["dates"]["after"])
    _show_provenance(record)

    st.image(
        str(outputs["comparison_png"]),
        caption="True colour (top); before NDVI, after NDVI and after-minus-before NDVI (bottom)",
        width="stretch",
    )
    st.markdown(
        "**Difference legend:** blue = NDVI increased; red = NDVI decreased; white = little "
        "difference; grey = excluded pixels. NDVI means Normalised Difference Vegetation Index."
    )
    st.caption(
        "True-colour stretching is display-only. The recorded reflectance scaling, NDVI arrays and "
        "statistics are unchanged."
    )

    validity = record["validity"]
    interpretation = record["interpretation_results"]
    categories = interpretation["change_categories"]
    coverage, threshold = st.columns(2)
    coverage.metric("Common valid coverage", f"{validity['valid_coverage_percent']:.2f}%")
    threshold.metric("Exploratory NDVI threshold", f"±{record['threshold']:.2f}")

    st.markdown(
        "### Persistent-vegetation statistics\n"
        "These statistics use only pixels classified as vegetation (SCL 4) on **BOTH dates**."
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric("Selected pixels", f"{interpretation['pixels']:,}")
    summary_columns[1].metric("Mean NDVI before", f"{interpretation['before_ndvi']['mean']:.3f}")
    summary_columns[2].metric("Mean NDVI after", f"{interpretation['after_ndvi']['mean']:.3f}")
    summary_columns[3].metric("Mean NDVI difference", f"{interpretation['delta_ndvi']['mean']:+.3f}")
    fraction_columns = st.columns(3)
    fraction_columns[0].metric("NDVI increased", f"{categories['increased']['fraction']:.2%}")
    fraction_columns[1].metric("NDVI decreased", f"{categories['decreased']['fraction']:.2%}")
    fraction_columns[2].metric("Approximately stable", f"{categories['approximately_stable']['fraction']:.2%}")
    st.warning(
        "These two-date NDVI statistics do not measure vegetation gain/loss, prove plant-health "
        "changes, establish damage, or identify the cause of a difference."
    )

    with st.expander("Downloads and technical details"):
        download_columns = st.columns(3)
        with download_columns[0]:
            _download(outputs["comparison_png"], "Download comparison PNG", "image/png", "vegetation-png")
        with download_columns[1]:
            _download(outputs["summary"], "Download summary", "text/plain", "vegetation-summary")
        with download_columns[2]:
            _download(outputs["execution_record"], "Download execution record", "application/json", "vegetation-json")
        raster_columns = st.columns(3)
        for column, name, label in zip(
            raster_columns,
            ("before_ndvi", "after_ndvi", "delta_ndvi"),
            ("Download before NDVI", "Download after NDVI", "Download NDVI difference"),
        ):
            with column:
                _download(outputs[name], label, "image/tiff", f"vegetation-{name}")
        st.json(record)


def _vegetation_mode() -> None:
    st.header("Vegetation comparison")
    st.write(
        "Run the fixed, corrected Sentinel-2 Level-2A example. Ordinary RGB uploads are not accepted "
        "for NDVI because they lack the required NIR, reflectance, geospatial and quality metadata."
    )
    config, missing = _prepared_vegetation_inputs()
    if config is None:
        st.error(f"Vegetation configuration is unavailable: {VEGETATION_CONFIG}")
    if missing:
        st.error("Prepared vegetation data is unavailable. Run the documented preparation command first.")
        with st.expander("Missing prepared files"):
            for path in missing:
                st.code(str(path))
    if config:
        st.info(
            f"Prepared acquisitions: {config['before']['date']} and {config['after']['date']}. "
            "The app does not download data on startup."
        )

    if st.button("Run prepared example", type="primary", disabled=bool(missing or config is None)):
        try:
            with st.status("Running corrected vegetation comparison...", expanded=True) as status:
                st.write("Validating bands, scale/offset metadata, grids and SCL masks")
                record = run_vegetation(VEGETATION_CONFIG, GEOLENS_OUTPUTS / "vegetation")
                status.update(label="Vegetation comparison complete", state="complete", expanded=False)
            st.session_state["vegetation_result"] = record
        except (VegetationValidationError, OSError, RuntimeError) as exc:
            st.error(f"Vegetation comparison failed: {exc}")

    record = st.session_state.get("vegetation_result")
    if isinstance(record, Mapping):
        _show_vegetation_result(record)


def _historical_map() -> None:
    draft_bounds = st.session_state.get("historical_draft_bounds")
    confirmed = st.session_state.get("historical_confirmed_aoi")
    confirmed_label = st.session_state.get("historical_aoi_label", "Confirmed map area")
    map_view = folium.Map(location=[9.4981, 76.3388], zoom_start=12, tiles="OpenStreetMap")
    if draft_bounds and not (
        isinstance(confirmed, ConfirmedAOI) and tuple(draft_bounds) == confirmed.bounds
    ):
        west, south, east, north = draft_bounds
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#2563eb",
            weight=3,
            fill=True,
            fill_opacity=0.12,
            tooltip="Current draft AOI — confirm below",
        ).add_to(map_view)
    if isinstance(confirmed, ConfirmedAOI):
        folium.Rectangle(
            bounds=[[confirmed.south, confirmed.west], [confirmed.north, confirmed.east]],
            color="#15803d",
            weight=3,
            fill=False,
            tooltip=str(confirmed_label),
            popup=str(confirmed_label),
        ).add_to(map_view)
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": {"shapeOptions": {"color": "#2563eb"}},
        },
        edit_options={"edit": False, "remove": False},
    ).add_to(map_view)
    map_state = st_folium(
        map_view,
        key="historical-aoi-map",
        height=500,
        width=None,
        returned_objects=["last_active_drawing", "all_drawings"],
    )
    drawing = map_state.get("last_active_drawing") if isinstance(map_state, Mapping) else None
    geometry = drawing.get("geometry") if isinstance(drawing, Mapping) else None
    if isinstance(geometry, Mapping):
        try:
            bounds = bounds_from_geojson(geometry)
            if update_aoi_state(st.session_state, bounds):
                st.rerun()
        except HistoricalValidationError as exc:
            st.error(str(exc))


def _show_historical_result(
    record: Mapping[str, Any],
    answer: Mapping[str, Any] | None = None,
    assistant_execution: Mapping[str, Any] | None = None,
) -> None:
    st.subheader("Historical vegetation timeline")
    if answer:
        st.markdown("## GeoLens answer")
        st.markdown(str(answer.get("text", "")))
        if answer.get("fallback_used"):
            st.caption(
                "A checked deterministic explanation is shown because the generated claim plan "
                "did not pass validation. The rejection reason is retained in technical details."
            )
        else:
            st.caption(
                "Generated from a validated claim plan; every displayed finding and date is tied "
                "to the execution record."
            )
        if assistant_execution:
            answer_path_value = assistant_execution.get("outputs", {}).get("answer_text")
            if answer_path_value:
                _download(
                    Path(answer_path_value),
                    "Download this answer",
                    "text/markdown",
                    "historical-answer",
                )

    status = record["status"]
    if status == "success":
        st.success("All requested annual observations are available.")
    elif status == "partial":
        st.warning(f"Partial result. Missing years: {record['missing_years']}")
    else:
        st.error("No requested year produced a suitable observation. Details are retained below.")

    available_observations = [
        item for item in record["annual_observations"] if item["status"] == "available"
    ]
    for observation in record["annual_observations"]:
        if observation["status"] != "available":
            st.warning(
                f"{observation['year']}: unavailable — "
                f"{observation.get('reason', 'no suitable observation')}"
            )
    if available_observations:
        observation_by_year = {item["year"]: item for item in available_observations}
        selected_year = st.selectbox(
            "Dated satellite observation",
            list(observation_by_year),
            format_func=lambda value: (
                f"{value} · acquired {observation_by_year[value]['acquisition_date']}"
            ),
            key=f"historical-observation-{Path(record['outputs']['execution_record']).parent.name}",
        )
        observation = observation_by_year[selected_year]
        image_columns = st.columns(2)
        image_columns[0].image(
            observation["outputs"]["true_color_png"],
            caption=f"{selected_year} true colour · display stretch only",
            width="stretch",
        )
        image_columns[1].image(
            observation["outputs"]["ndvi_png"],
            caption=f"{selected_year} NDVI · grey pixels excluded",
            width="stretch",
        )

    st.markdown("### Explore an NDVI comparison")
    st.caption(
        "Blue = NDVI increase; red = NDVI decrease; white = little difference; grey = excluded. "
        "Unavailable years are not bridged or relabelled as adjacent comparisons."
    )
    first_last = record["first_to_last_difference"]
    differences = list(record["adjacent_differences"]) + [
        {**first_last, "comparison_kind": "first_to_last"}
    ]
    difference_by_key = {
        f"{item.get('comparison_kind', 'adjacent')}:{item['label']}": item
        for item in differences
    }
    selected_difference_key = st.selectbox(
        "NDVI difference interval",
        list(difference_by_key),
        format_func=lambda value: (
            ("First-to-last" if value.startswith("first_to_last:") else "Adjacent")
            + " · "
            + difference_by_key[value]["label"].replace("-to-", " → ").replace("-first-last", "")
            + (" · unavailable" if difference_by_key[value]["status"] != "available" else "")
        ),
        key=f"historical-difference-{Path(record['outputs']['execution_record']).parent.name}",
    )
    selected_difference = difference_by_key[selected_difference_key]
    if selected_difference["status"] == "available":
        st.image(
            selected_difference["outputs"]["difference_png"],
            caption=(
                f"{selected_difference['before_date']} to "
                f"{selected_difference['after_date']} NDVI difference"
            ),
            width="stretch",
        )
    else:
        st.warning(
            f"{selected_difference['label']}: unavailable — "
            f"{selected_difference['reason']}"
        )

    summary_path = Path(record["outputs"]["summary"])
    _render_follow_up_chat(
        FollowUpContext(
            workflow="historical_vegetation",
            analysis_id=Path(record["outputs"]["execution_record"]).parent.name,
            record=record,
            base_answer=(
                str(answer.get("text", ""))
                if answer
                else summary_path.read_text(encoding="utf-8")
            ),
            selected_comparison_id=selected_difference["label"],
        )
    )
    with st.expander("View evidence and technical details"):
        if answer and answer.get("fallback_used"):
            st.warning(f"Fallback reason: {answer.get('fallback_reason')}")
        trend = record["common_valid_trend"]
        st.markdown("**Directly comparable trend metrics**")
        st.metric("Common-valid coverage", f"{trend['coverage']:.1%}")
        if trend["sufficient"]:
            chart_rows = trend["annual_statistics"]
            st.line_chart(
                {
                    "year": [item["year"] for item in chart_rows],
                    "mean NDVI": [item["mean"] for item in chart_rows],
                    "median NDVI": [item["median"] for item in chart_rows],
                },
                x="year",
                y=["mean NDVI", "median NDVI"],
            )
        else:
            st.warning(f"Comparable trend withheld: {trend['reason']}")
        cache = record["cache"]
        cache_columns = st.columns(2)
        cache_columns[0].metric(
            "Newly retrieved selected windows", cache["selected_asset_windows_retrieved"]
        )
        cache_columns[1].metric(
            "Selected windows loaded from cache", cache["selected_asset_windows_loaded_from_cache"]
        )
        st.caption(
            "Cache matching is tied to the exact confirmed AOI; another area's cache is never substituted."
        )
        st.markdown("**Year-valid observation metrics**")
        st.json([
            {
                "year": observation["year"],
                "status": observation["status"],
                "acquisition_date": observation.get("acquisition_date"),
                "valid_coverage": observation.get("valid_coverage"),
                "per_year_ndvi": observation.get("per_year_ndvi"),
                "population": observation.get("per_year_population"),
                "reason": observation.get("reason"),
            }
            for observation in record["annual_observations"]
        ])
        st.caption(
            "Year-valid statistics use each observation's own valid pixels. Common-pixel and "
            "pairwise statistics use the separately identified populations in the record."
        )
        st.markdown("**Pairwise comparison metrics**")
        st.json(record["adjacent_differences"])
        st.markdown("**Deterministic source summary**")
        st.code(summary_path.read_text(encoding="utf-8"), language=None)
        st.write(record["provider"])
        st.write({"confirmed_boundary": record["request"]["aoi"], "grid": record["grid"]})
        downloads = st.columns(2)
        with downloads[0]:
            _download(summary_path, "Download summary", "text/plain", "historical-summary")
        with downloads[1]:
            _download(
                Path(record["outputs"]["execution_record"]),
                "Download result JSON",
                "application/json",
                "historical-json",
            )
        if assistant_execution:
            assistant_path = Path(assistant_execution["outputs"]["assistant_record"])
            _download(
                assistant_path,
                "Download assistant audit record",
                "application/json",
                "historical-assistant-json",
            )
        st.markdown("**Annual visual outputs**")
        for observation in record["annual_observations"]:
            if observation["status"] != "available":
                continue
            visual_columns = st.columns(2)
            with visual_columns[0]:
                _download(
                    Path(observation["outputs"]["true_color_png"]),
                    f"Download {observation['year']} true colour",
                    "image/png",
                    f"historical-{observation['year']}-rgb",
                )
            with visual_columns[1]:
                _download(
                    Path(observation["outputs"]["ndvi_png"]),
                    f"Download {observation['year']} NDVI",
                    "image/png",
                    f"historical-{observation['year']}-ndvi",
                )
        st.markdown("**Difference visual outputs**")
        for difference in record["adjacent_differences"]:
            if difference["status"] != "available":
                continue
            _download(
                Path(difference["outputs"]["difference_png"]),
                f"Download {difference['label']} difference",
                "image/png",
                f"historical-{difference['label']}-difference",
            )
        if first_last["status"] == "available":
            _download(
                Path(first_last["outputs"]["difference_png"]),
                "Download first-to-last difference",
                "image/png",
                "historical-first-last-difference",
            )
        st.json(record)


def _manual_historical_request(confirmed: ConfirmedAOI) -> None:
    """Render the validated non-LLM fallback without displacing the chat workflow."""
    st.caption("This fallback uses the same AOI, request validation, and historical tool as chat.")
    control_columns = st.columns(3)
    analysis_label = control_columns[0].selectbox(
        "Requested analysis",
        ("Vegetation NDVI timeline", "Construction/roads from historical Sentinel-2"),
    )
    current_year = datetime.now(timezone.utc).year
    start_year = int(control_columns[1].number_input(
        "Start year",
        min_value=SUPPORTED_FIRST_YEAR,
        max_value=current_year,
        step=1,
        key="historical_manual_start_year",
    ))
    end_year = int(control_columns[2].number_input(
        "End year",
        min_value=SUPPORTED_FIRST_YEAR,
        max_value=current_year,
        step=1,
        key="historical_manual_end_year",
    ))
    analysis = "vegetation_ndvi" if analysis_label == "Vegetation NDVI timeline" else "construction"
    control_signature = (confirmed.cache_key, start_year, end_year, analysis)
    if st.session_state.get("historical_controls_signature") != control_signature:
        st.session_state["historical_controls_signature"] = control_signature
        st.session_state.pop("historical_request", None)
        st.session_state.pop("historical_result", None)
        st.session_state.pop("assistant_execution", None)
        st.session_state.pop("historical_answer", None)

    observation_count = end_year - start_year + 1
    st.write(
        f"Requested annual observations: {observation_count}; adjacent comparisons: "
        f"{max(0, observation_count - 1)}. Maximum allowed observations: {MAX_ANNUAL_OBSERVATIONS}."
    )
    st.caption(
        "Season policy: January 1–March 31 for every year. Candidate ranking: lowest catalogue cloud "
        "first, then require at least 80% valid SCL coverage inside the confirmed AOI."
    )
    if st.button("Validate structured request"):
        try:
            validated = build_request(confirmed, start_year, end_year, analysis=analysis)
            st.session_state["historical_request"] = validated
            st.session_state.pop("historical_result", None)
            st.session_state.pop("assistant_execution", None)
            st.session_state.pop("historical_answer", None)
        except HistoricalValidationError as exc:
            st.error(str(exc))

    validated_request = st.session_state.get("historical_request")
    if isinstance(validated_request, HistoricalRequest):
        st.success("Request validated. Analysis will use the exact confirmed AOI and fixed seasonal policy.")
        with st.expander("Validated request"):
            st.json(validated_request.record())
        if st.button("Run historical analysis", type="primary"):
            try:
                with st.status("Searching and processing annual observations...", expanded=True) as status:
                    st.write("Using bounded Earth Search queries and cached AOI windows where available")
                    result = run_historical(
                        validated_request,
                        GEOLENS_OUTPUTS / "historical",
                        HISTORICAL_CACHE,
                    )
                    status.update(label="Historical analysis complete", state="complete", expanded=False)
                st.session_state["historical_result"] = result
            except (HistoricalValidationError, HistoricalNetworkError, OSError, RuntimeError) as exc:
                st.error(f"Historical analysis failed: {exc}")


def _historical_mode() -> None:
    st.session_state.setdefault("historical_manual_start_year", 2021)
    st.session_state.setdefault("historical_manual_end_year", 2025)
    if not st.session_state.get("historical_area_initialised"):
        if not isinstance(st.session_state.get("historical_confirmed_aoi"), ConfirmedAOI):
            activate_confirmed_aoi(
                st.session_state,
                DEMO_BOUNDS,
                source="demo_preset",
                label=DEMO_AOI_LABEL,
            )
        st.session_state["historical_area_initialised"] = True

    st.header("Vegetation history")
    st.write(
        "Ask a question about vegetation-related greenness inside the confirmed map area. "
        "GeoLens validates the area and request before any satellite analysis runs."
    )
    _step("1 · Confirm area")

    confirmed = st.session_state.get("historical_confirmed_aoi")
    with st.container(border=True):
        st.markdown("### Active analysis area")
        if isinstance(confirmed, ConfirmedAOI):
            confirmed_label = st.session_state.get("historical_aoi_label", "Confirmed map area")
            st.success(
                f"Using the {confirmed_label} · approximately "
                f"{confirmed.area_square_km:.3f} km²"
            )
            st.caption(f"Confirmed WGS84 bounds: {confirmed.bounds}")
        else:
            st.warning("No area is confirmed. Confirm a custom rectangle or restore the demo area.")
        if st.button("Load Alappuzha demo area", width="stretch"):
            activate_confirmed_aoi(
                st.session_state,
                DEMO_BOUNDS,
                source="demo_preset",
                label=DEMO_AOI_LABEL,
            )
            st.session_state["historical_area_initialised"] = True
            st.session_state["historical_manual_start_year"] = 2021
            st.session_state["historical_manual_end_year"] = 2025
            st.rerun()
        st.caption(
            "The tested demo preset uses WGS84 bounds [76.34, 9.48, 76.36, 9.50] and "
            "the same area validator as a custom selection."
        )

    _historical_map()

    with st.expander("Choose or confirm a custom area"):
        st.caption(
            f"Draw a rectangle on the map, inspect it here, and confirm it explicitly. "
            f"The current limit is {DEFAULT_AREA_LIMIT_SQUARE_KM:.0f} km². Changing the boundary "
            "invalidates plans and results for the previous area."
        )
        if st.button("Reset boundary and result"):
            update_aoi_state(st.session_state, None)
            st.rerun()

        draft = st.session_state.get("historical_draft_bounds")
        draft_aoi = None
        if draft:
            try:
                draft_aoi = confirm_aoi(draft)
                st.write({
                    "draft_bounds_wgs84": [round(value, 7) for value in draft_aoi.bounds],
                    "approximate_area_square_km": round(draft_aoi.area_square_km, 3),
                })
            except HistoricalValidationError as exc:
                st.error(str(exc))
        if st.button("Confirm this boundary", disabled=draft_aoi is None, type="primary"):
            activate_confirmed_aoi(
                st.session_state,
                draft_aoi.bounds,
                source="custom_map_selection",
                label="Custom map area",
            )
            st.success("Boundary confirmed. Request controls are now enabled.")
            st.rerun()

    confirmed = st.session_state.get("historical_confirmed_aoi")
    if not isinstance(confirmed, ConfirmedAOI):
        st.warning("Confirm an analysis area before asking GeoLens to make a plan.")

    _step("2 · Ask and review")
    st.markdown("### Ask GeoLens")
    st.caption(
        "Try: “Compare vegetation from 2021 to 2025.” Requests for construction, buildings, "
        "or roads are preserved, but are not silently replaced with an NDVI analysis."
    )
    provider_info = _cached_language_provider()
    if provider_info.available:
        st.success(
            f"Local language model ready: {provider_info.identifier}. Requests stay on this machine."
        )
    else:
        st.warning(f"Local language model unavailable: {provider_info.reason}")
        st.code("ollama serve\nollama pull qwen3:8b\n.venv/bin/python scripts/check_llm.py")
    prompt = st.chat_input(
        "Example: Compare vegetation here from 2021 to 2025",
        disabled=not provider_info.available,
    )
    if prompt:
        for key in (
            "assistant_plan",
            "assistant_execution",
            "historical_answer",
            "historical_request",
            "historical_result",
        ):
            st.session_state.pop(key, None)
        st.session_state["historical_chat"] = [{"role": "user", "content": prompt}]
        try:
            with st.status("Interpreting and validating the request...", expanded=True) as status:
                provider = OllamaProvider(provider_info)
                interpreted = interpret_request(
                    provider,
                    prompt,
                )
                plan = build_tool_plan(
                    interpreted,
                    confirmed if isinstance(confirmed, ConfirmedAOI) else None,
                    current_year=datetime.now(timezone.utc).year,
                    active_boundary_source=st.session_state.get("historical_aoi_source"),
                    active_boundary_label=st.session_state.get("historical_aoi_label"),
                )
                status.update(label="Request interpretation complete", state="complete", expanded=False)
            st.session_state["assistant_plan"] = plan
            st.session_state["historical_chat"].append({
                "role": "assistant",
                "content": (
                    f"Interpreted analyses: {', '.join(interpreted.analyses) or 'none'}. "
                    f"Plan status: {plan.status}. Review the validated plan before running it."
                ),
            })
        except LanguageModelError as exc:
            st.session_state["historical_chat"].append({
                "role": "assistant",
                "content": f"The local model response was not accepted: {exc}",
            })
    for message in st.session_state.get("historical_chat", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    assistant_plan = st.session_state.get("assistant_plan")
    if isinstance(assistant_plan, ToolPlan):
        st.markdown("#### Validated interpretation and tool plan")
        if assistant_plan.boundary_resolution.label:
            st.info(f"Using the {assistant_plan.boundary_resolution.label}")
        st.markdown(summarize_tool_plan(assistant_plan))
        years: Any = (
            list(assistant_plan.historical_request.years)
            if assistant_plan.historical_request is not None
            else None
        )
        if years is None and assistant_plan.interpretation.start_year is not None:
            years = (
                f"{assistant_plan.interpretation.start_year}–"
                f"{assistant_plan.interpretation.end_year}"
                if assistant_plan.interpretation.end_year is not None
                else str(assistant_plan.interpretation.start_year)
            )
        st.write(
            f"Plan status: **{assistant_plan.status.replace('_', ' ')}** · "
            f"Requested: {', '.join(assistant_plan.interpretation.analyses) or 'none'} · "
            f"Years: {years or 'not resolved'}"
        )
        with st.expander("View plan evidence and technical details"):
            st.json(assistant_plan.record())
        if assistant_plan.boundary_resolution.missing_information:
            st.warning(
                "Clarification required: "
                + ", ".join(assistant_plan.boundary_resolution.missing_information)
                + ". Submit a complete follow-up request."
            )
        for tool in assistant_plan.tools:
            if tool.get("status") == "unsupported_in_historical_mode":
                st.warning(str(tool.get("reason")))
        if assistant_plan.runnable:
            _step("3 · Analyse")
        if assistant_plan.runnable and st.button("Run interpreted request", type="primary"):
            with st.status("Executing validated GeoLens tools...", expanded=True) as status:
                execution, tool_result = execute_tool_plan(
                    assistant_plan,
                    GEOLENS_OUTPUTS / "assistant",
                    GEOLENS_OUTPUTS / "historical",
                    HISTORICAL_CACHE,
                )
                if tool_result is not None:
                    provider = OllamaProvider(provider_info)
                    answer = generate_supported_answer(provider, tool_result, execution)
                    execution = attach_final_answer(execution, answer)
                    st.session_state["historical_result"] = tool_result
                    st.session_state["historical_answer"] = answer
                st.session_state["assistant_execution"] = execution
                status.update(
                    label="Validated tool execution complete",
                    state="complete" if execution["status"] != "failed" else "error",
                    expanded=False,
                )
            if execution["status"] == "failed":
                st.error("The validated tool failed. The error is retained in the assistant audit record.")

    if not isinstance(confirmed, ConfirmedAOI):
        return

    with st.expander("Manual structured request (fallback)"):
        _manual_historical_request(confirmed)

    result = st.session_state.get("historical_result")
    if isinstance(result, Mapping):
        _step("4 · Explore results")
        answer = st.session_state.get("historical_answer")
        assistant_execution = st.session_state.get("assistant_execution")
        _show_historical_result(
            result,
            answer if isinstance(answer, Mapping) else None,
            assistant_execution if isinstance(assistant_execution, Mapping) else None,
        )


def main() -> None:
    st.set_page_config(page_title="GeoLens", page_icon="🌍", layout="wide")
    _inject_geolens_theme()
    st.markdown(
        """
        <div class="geolens-hero">
          <div class="geolens-mark">G</div>
          <div>
            <h1>GeoLens</h1>
            <p>Local, evidence-linked comparison of uploaded imagery and historical vegetation greenness.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("## Choose an analysis")
    mode = st.sidebar.radio(
        "Choose a workflow",
        ("Buildings and infrastructure", "Vegetation history"),
        index=1,
    )
    st.sidebar.caption(
        "Buildings and infrastructure requires 2–4 dated, aligned RGB images. "
        "Vegetation history uses a confirmed map area, annual satellite observations and corrected "
        "NDVI processing. The inputs and results are not interchangeable."
    )
    st.sidebar.caption(
        f"Local only · upload limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per image · no database"
    )
    if mode == "Buildings and infrastructure":
        _construction_mode()
    else:
        _historical_mode()


if __name__ == "__main__":
    main()
