#!/usr/bin/env python3
"""Local Streamlit interface for SatQuery's existing analysis tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import streamlit as st


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
from satquery.vegetation import (  # noqa: E402
    VegetationValidationError,
    load_config,
    run as run_vegetation,
)


CHECKPOINT = PROJECT_ROOT / "weights" / "MCI_model.pth"
CHANGE_SOURCE = PROJECT_ROOT / ".cache" / "upstream" / "change-agent"
VEGETATION_CONFIG = PROJECT_ROOT / "configs" / "vegetation.real-example.json"
GEOLENS_OUTPUTS = PROJECT_ROOT / "outputs" / "geolens"


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


def main() -> None:
    st.set_page_config(page_title="GeoLens", page_icon="🌍", layout="wide")
    st.title("GeoLens")
    st.caption("Local satellite change analysis using SatQuery's existing verified tools")
    st.sidebar.markdown("## Analysis mode")
    mode = st.sidebar.radio("Choose a workflow", ("Construction changes", "Vegetation comparison"))
    st.sidebar.caption(
        f"Local only · upload limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per image · no database"
    )
    if mode == "Construction changes":
        _construction_mode()
    else:
        _vegetation_mode()


if __name__ == "__main__":
    main()
