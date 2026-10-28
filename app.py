#!/usr/bin/env python3
"""Local Streamlit interface for SatQuery's existing analysis tools."""

from __future__ import annotations

import argparse
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
from satquery.historical import (  # noqa: E402
    DEFAULT_AREA_LIMIT_SQUARE_KM,
    DEMO_BOUNDS,
    MAX_ANNUAL_OBSERVATIONS,
    SUPPORTED_FIRST_YEAR,
    ConfirmedAOI,
    HistoricalNetworkError,
    HistoricalRequest,
    HistoricalValidationError,
    bounds_from_geojson,
    build_request,
    confirm_aoi,
    run_historical,
    update_aoi_state,
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


def _show_construction_result(record: Mapping[str, Any]) -> None:
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

    st.subheader("Latest construction analysis")
    before_path = Path(record["inputs"]["before"]["path"])
    after_path = Path(record["inputs"]["after"]["path"])
    before_column, after_column = st.columns(2)
    before_column.image(str(before_path), caption="Before — prepared model input", width="stretch")
    after_column.image(str(after_path), caption="After — prepared model input", width="stretch")

    mask_column, overlay_column = st.columns(2)
    mask_column.image(
        str(outputs["color_mask_with_legend"]),
        caption="Predicted class mask",
        width="stretch",
    )
    overlay_column.image(str(outputs["overlay"]), caption="Prediction overlay on after image", width="stretch")

    st.markdown(
        "**Mask colours:** black = background; yellow = predicted road change; "
        "red = predicted building change."
    )
    distribution = record["model_outputs"]["mask_class_distribution"]
    fraction_columns = st.columns(3)
    for column, label in zip(fraction_columns, (0, 1, 2)):
        item = distribution[str(label)]
        column.metric(f"Predicted pixel fraction — {CLASS_NAMES[label]}", f"{item['fraction']:.2%}")
    st.caption(
        "These percentages are fractions of model-output pixels. They are not accuracy, confidence, "
        "real-world area, or reference-label counts."
    )

    st.markdown("**Raw model caption**")
    st.code(record["model_outputs"]["caption"]["text"] or "[empty]", language=None)
    st.caption(
        "Caption repetitions are preserved exactly. The caption is an unverified output from a separate "
        "model head and is not derived from the mask."
    )

    with st.expander("Downloads and technical details"):
        download_columns = st.columns(3)
        with download_columns[0]:
            _download(outputs["color_mask_with_legend"], "Download coloured mask", "image/png", "construction-mask")
        with download_columns[1]:
            _download(outputs["overlay"], "Download overlay", "image/png", "construction-overlay")
        with download_columns[2]:
            _download(outputs["caption"], "Download raw caption", "text/plain", "construction-caption")
        _download(outputs["execution_record"], "Download execution record", "application/json", "construction-json")
        st.json(record)


def _construction_mode() -> None:
    st.header("Construction changes")
    st.write(
        "Upload before and after PNG/JPEG images of the same location. The verified model requires "
        "already aligned, 256×256, three-channel RGB inputs."
    )
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

    can_analyse = bool(prepared_before and prepared_after and aligned and not errors and not prerequisites)
    if st.button("Analyse construction changes", type="primary", disabled=not can_analyse):
        run_directory = create_unique_run_directory(GEOLENS_OUTPUTS / "construction", "run")
        try:
            before_path, after_path = save_prepared_pair(prepared_before, prepared_after, run_directory)
            args = argparse.Namespace(
                before=before_path,
                after=after_path,
                checkpoint=CHECKPOINT,
                output_dir=run_directory,
                device="cpu",
                source_dir=CHANGE_SOURCE,
            )
            with st.status("Running verified CPU inference...", expanded=True) as status:
                st.write("Validating inputs, checkpoint and upstream source")
                record = run_change_agent(args, model_loader=_streamlit_model_loader)
                status.update(label="Construction analysis complete", state="complete", expanded=False)
            st.session_state["construction_result"] = record
        except (FileNotFoundError, InputValidationError, ModelSafetyError, RuntimeError) as exc:
            st.error(f"Construction analysis failed: {exc}")

    record = st.session_state.get("construction_result")
    if isinstance(record, Mapping):
        _show_construction_result(record)


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
    map_view = folium.Map(location=[9.4981, 76.3388], zoom_start=12, tiles="OpenStreetMap")
    if draft_bounds:
        west, south, east, north = draft_bounds
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#2563eb",
            weight=3,
            fill=True,
            fill_opacity=0.12,
            tooltip="Current draft AOI — confirm below",
        ).add_to(map_view)
    confirmed = st.session_state.get("historical_confirmed_aoi")
    if isinstance(confirmed, ConfirmedAOI):
        folium.Rectangle(
            bounds=[[confirmed.south, confirmed.west], [confirmed.north, confirmed.east]],
            color="#15803d",
            weight=3,
            fill=False,
            tooltip="Confirmed AOI",
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
    status = record["status"]
    if status == "success":
        st.success("All requested annual observations are available.")
    elif status == "partial":
        st.warning(f"Partial result. Missing years: {record['missing_years']}")
    else:
        st.error("No requested year produced a suitable observation. Details are retained below.")

    if answer:
        st.markdown("### GeoLens assistant answer")
        if answer.get("fallback_used"):
            st.warning(f"Deterministic fallback used: {answer.get('fallback_reason')}")
        else:
            st.caption("Model prose accepted; every displayed numerical fact was inserted from tool results.")
        st.markdown(str(answer.get("text", "")))

    cache = record["cache"]
    cache_columns = st.columns(2)
    cache_columns[0].metric("Newly retrieved selected windows", cache["selected_asset_windows_retrieved"])
    cache_columns[1].metric(
        "Selected windows loaded from cache", cache["selected_asset_windows_loaded_from_cache"]
    )
    st.caption("Cache matching is tied to the exact confirmed AOI; another area's cache is never substituted.")

    for observation in record["annual_observations"]:
        year = observation["year"]
        if observation["status"] != "available":
            st.warning(f"{year}: unavailable — {observation.get('reason', 'no suitable observation')}")
            continue
        st.markdown(f"### {year} · acquired {observation['acquisition_date']}")
        image_columns = st.columns(2)
        image_columns[0].image(
            observation["outputs"]["true_color_png"],
            caption=f"{year} true colour · display stretch only",
            width="stretch",
        )
        image_columns[1].image(
            observation["outputs"]["ndvi_png"],
            caption=f"{year} NDVI · grey pixels excluded",
            width="stretch",
        )
        metrics = st.columns(4)
        metrics[0].metric("Valid coverage", f"{observation['valid_coverage']:.1%}")
        metrics[1].metric("Mean NDVI (year-valid)", f"{observation['per_year_ndvi']['mean']:.3f}")
        metrics[2].metric("Median NDVI (year-valid)", f"{observation['per_year_ndvi']['median']:.3f}")
        cloud = observation.get("catalogue_cloud_cover_percent")
        metrics[3].metric("Catalogue scene cloud", f"{float(cloud):.2f}%" if cloud is not None else "unknown")
        st.caption(
            "The year-valid statistic uses this observation's own valid pixels and is not directly "
            "comparable unless the all-observation common population below is sufficient."
        )

    trend = record["common_valid_trend"]
    st.markdown("### Directly comparable trend")
    st.metric("Pixels valid across every available observation", f"{trend['coverage']:.1%}")
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

    st.markdown("### Adjacent-observation NDVI differences")
    st.caption(
        "Blue = NDVI increase; red = NDVI decrease; white = little difference; grey = excluded. "
        "Unavailable years are not bridged or relabelled as adjacent comparisons."
    )
    for difference in record["adjacent_differences"]:
        if difference["status"] != "available":
            st.warning(f"{difference['label']}: unavailable — {difference['reason']}")
            continue
        columns = st.columns((2, 1))
        columns[0].image(
            difference["outputs"]["difference_png"],
            caption=f"{difference['before_date']} to {difference['after_date']} NDVI difference",
            width="stretch",
        )
        categories = difference["categories"]
        columns[1].metric("Valid pair coverage", f"{difference['valid_coverage']:.1%}")
        columns[1].metric("NDVI increase", f"{categories['increased']['fraction']:.1%}")
        columns[1].metric("NDVI decrease", f"{categories['decreased']['fraction']:.1%}")
        columns[1].metric("Approximately stable", f"{categories['approximately_stable']['fraction']:.1%}")

    first_last = record["first_to_last_difference"]
    st.markdown("### First-to-last NDVI difference")
    if first_last["status"] == "available":
        st.image(
            first_last["outputs"]["difference_png"],
            caption=f"{first_last['before_date']} to {first_last['after_date']}",
            width="stretch",
        )
    else:
        st.warning(first_last["reason"])

    st.warning(
        "NDVI increase/decrease categories are not statistically significant change, confirmed "
        "vegetation gain/loss, deforestation, improved plant health, damage, or a causal explanation."
    )
    summary_path = Path(record["outputs"]["summary"])
    st.markdown("### Deterministic summary")
    st.code(summary_path.read_text(encoding="utf-8"), language=None)

    with st.expander("Data sources, downloads and execution details"):
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


def _historical_mode() -> None:
    st.header("Historical vegetation analysis")
    st.write(
        "Draw and confirm a small rectangle, then request a bounded Sentinel-2 annual NDVI timeline. "
        "The initial map view is centred on Alappuzha but is not a confirmed analysis area."
    )
    st.info(
        f"Demo AOI limit: {DEFAULT_AREA_LIMIT_SQUARE_KM:.0f} km². The app retrieves only small "
        "AOI raster windows and never applies the fine-resolution construction model to Sentinel-2."
    )
    map_actions = st.columns(2)
    if map_actions[0].button("Load documented Alappuzha demo rectangle"):
        update_aoi_state(st.session_state, DEMO_BOUNDS)
        st.rerun()
    if map_actions[1].button("Reset boundary and result"):
        update_aoi_state(st.session_state, None)
        st.rerun()
    _historical_map()

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
        st.session_state["historical_confirmed_aoi"] = draft_aoi
        for key in (
            "historical_request",
            "historical_result",
            "assistant_plan",
            "assistant_execution",
            "historical_answer",
            "historical_chat",
        ):
            st.session_state.pop(key, None)
        st.success("Boundary confirmed. Request controls are now enabled.")

    confirmed = st.session_state.get("historical_confirmed_aoi")
    if isinstance(confirmed, ConfirmedAOI):
        st.success(
            f"Confirmed AOI: {confirmed.bounds}; approximately {confirmed.area_square_km:.3f} km²."
        )
    else:
        st.warning("No area is confirmed. Drawing or loading a rectangle does not confirm it automatically.")

    st.markdown("### Ask GeoLens")
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
                    aoi_confirmed=isinstance(confirmed, ConfirmedAOI),
                )
                plan = build_tool_plan(
                    interpreted,
                    confirmed if isinstance(confirmed, ConfirmedAOI) else None,
                    current_year=datetime.now(timezone.utc).year,
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
        st.json(assistant_plan.record())
        if assistant_plan.interpretation.missing_information:
            st.warning(
                "Clarification required: "
                + ", ".join(assistant_plan.interpretation.missing_information)
                + ". Submit a complete follow-up request."
            )
        for tool in assistant_plan.tools:
            if tool.get("status") == "unsupported_in_historical_mode":
                st.warning(str(tool.get("reason")))
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
                    answer = generate_supported_answer(provider, tool_result)
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

    st.markdown("### Manual structured request")
    st.caption("This remains available as a non-LLM fallback and uses the same validators and tool.")
    control_columns = st.columns(3)
    analysis_label = control_columns[0].selectbox(
        "Requested analysis",
        ("Vegetation NDVI timeline", "Construction/roads from historical Sentinel-2"),
    )
    current_year = datetime.now(timezone.utc).year
    start_year = int(control_columns[1].number_input(
        "Start year", min_value=SUPPORTED_FIRST_YEAR, max_value=current_year, value=2021, step=1
    ))
    end_year = int(control_columns[2].number_input(
        "End year", min_value=SUPPORTED_FIRST_YEAR, max_value=current_year, value=2025, step=1
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

    result = st.session_state.get("historical_result")
    if isinstance(result, Mapping):
        answer = st.session_state.get("historical_answer")
        assistant_execution = st.session_state.get("assistant_execution")
        _show_historical_result(
            result,
            answer if isinstance(answer, Mapping) else None,
            assistant_execution if isinstance(assistant_execution, Mapping) else None,
        )


def main() -> None:
    st.set_page_config(page_title="GeoLens", page_icon="🌍", layout="wide")
    st.title("GeoLens")
    st.caption("Local satellite change analysis using SatQuery's existing verified tools")
    st.sidebar.markdown("## Analysis mode")
    mode = st.sidebar.radio(
        "Choose a workflow",
        ("Construction changes", "Vegetation comparison", "Historical vegetation"),
    )
    st.sidebar.caption(
        f"Local only · upload limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per image · no database"
    )
    if mode == "Construction changes":
        _construction_mode()
    elif mode == "Vegetation comparison":
        _vegetation_mode()
    else:
        _historical_mode()


if __name__ == "__main__":
    main()
