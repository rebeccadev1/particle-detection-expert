"""Streamlit UI for per-tile particle detection on structured wafer surfaces."""

from __future__ import annotations

import contextlib
import io
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture.capture_ui import keep_capture_inputs, keep_stage_inputs, render_capture_tab, suspend_live_view  # noqa: E402
from src.overall.config import (  # noqa: E402
    apply_nsew_settings,
    beta_preset_rows,
    load_config,
    preset_for_beta,
    resolve_output_dir,
)
from src.overall.io.results_writer import excel_bytes, write_csv, write_xlsx  # noqa: E402
from src.overall.io.tile_loader import input_folder_labels  # noqa: E402
from src.verification.ui import (  # noqa: E402
    render_collection_tab,
    render_label_tab,
    render_labeled_tiles_tab,
    render_tile_inspect_tab,
)
from src.verification.queue import load_last_pipeline, load_particles_csv, write_last_pipeline  # noqa: E402
from src.main_pipeline.ml.infer import next_versioned_model_path  # noqa: E402
from src.training.train import default_label_path, main as train_main  # noqa: E402
from src.main_pipeline.job import claim_pipeline_result, pipeline_status, start_pipeline, stop_pipeline  # noqa: E402
from src.redirector.redirector import render_redirector_tab  # noqa: E402
from src.main_pipeline.report.report_generator import (  # noqa: E402
    encode_overlay_jpeg,
    overlay_downsample,
    overlay_markers,
    particles_at_least,
    summary_stats,
)
from src.shadow.shadow import keep_shadow_inputs, render_shadow_tab  # noqa: E402

st.set_page_config(page_title="Particle Detection Software", layout="wide")
st.title("Particle Detection Software")

NM_PER_UM = 1000.0
UI_SCHEMA = "mixed_layout_v22"


def _nm_to_um(nm: float) -> float:
    return float(nm) / NM_PER_UM


def _um_to_nm(um: float) -> float:
    return float(um) * NM_PER_UM


def _base_config() -> dict:
    return load_config(ROOT / "config.yaml")


def _widget_defaults(base: dict) -> dict:
    """Map config.yaml fields onto sidebar widget session-state keys."""
    pipe = base.get("pipeline") or {}
    pre = base.get("preprocessing") or {}
    det = base.get("detection") or {}
    meas = base.get("measurement") or {}
    report = base.get("report") or {}
    method = str(det.get("method", "fft"))
    if method not in ("fft", "tophat"):
        method = "fft"
    fft_mask = str(det.get("fft_mask", "per_tile"))
    if fft_mask not in ("per_tile", "shared"):
        fft_mask = "per_tile"
    size_agg = str(meas.get("size_aggregation", "max"))
    if size_agg not in ("max", "mean"):
        size_agg = "max"
    return {
        "ui_input_dir": str(base.get("input_dir") or ""),
        "ui_output_dir": str(base.get("output_dir") or "../Outputs"),
        "ui_filename_pattern": str(base.get("filename_pattern")),
        "ui_overlap_fraction": float(base.get("overlap_fraction", 0.0)),
        "ui_pixel_size_um": _nm_to_um(float(base.get("pixel_size_nm", 960.0))),
        "ui_workers": int(pipe.get("workers", 0) or 0),
        "ui_denoise": bool(pre.get("denoise", True)),
        "ui_denoise_sigma": float(pre.get("denoise_sigma", 1.0)),
        "ui_flatten_illumination": bool(pre.get("flatten_illumination", True)),
        "ui_flatten_sigma": float(pre.get("flatten_sigma", 160.0)),
        "ui_contrast_stretch": bool(pre.get("contrast_stretch", True)),
        "ui_method": method,
        "ui_particles_bright": bool(det.get("particles_bright", True)),
        "ui_fft_peak_threshold": float(det.get("fft_peak_threshold", 0.35)),
        "ui_fft_mask": fft_mask,
        "ui_tophat_radius": int(det.get("tophat_radius", 50)),
        "ui_min_size_um": _nm_to_um(float(det.get("min_size_nm", 10000.0))),
        "ui_max_size_um": _nm_to_um(float(det.get("max_size_nm", 100000.0))),
        "ui_blob_threshold": float(det.get("blob_threshold", 0.08)),
        "ui_blob_min_sigma": float(det.get("blob_min_sigma", 3.7)),
        "ui_min_prominence": float(det.get("min_prominence", 0.30)),
        "ui_min_confidence_pct": int(round(float(det.get("min_confidence", 0.50)) * 100.0)),
        "ui_blob_max_sigma": float(det.get("blob_max_sigma", 36.0)),
        "ui_local_snr_sigma": float(det.get("local_snr_sigma", 0.0)),
        "ui_edge_soften_sigma": float(det.get("edge_soften_sigma", 12.0)),
        "ui_edge_soften_strength": float(det.get("edge_soften_strength", 2.0)),
        "ui_edge_exclude_px": float(det.get("edge_exclude_px", 12.0)),
        "ui_min_circularity": float(det.get("min_circularity", 0.0)),
        "ui_structure_neighbor_px": float(det.get("structure_neighbor_px", 48.0)),
        "ui_ml_enabled": bool((base.get("ml") or {}).get("enabled", False)),
        "ui_ml_threshold": float((base.get("ml") or {}).get("threshold", 0.25)),
        "ui_merge_radius_px": float(meas.get("merge_radius_px", 8.0)),
        "ui_size_aggregation": size_agg,
        "ui_target_mb": float(report.get("target_mb", 20.0)),
        "ui_downsample": int(report.get("downsample", 0) or 0),
        "ui_f_beta": preset_for_beta(
            beta_preset_rows(base),
            1.0 if report.get("f_beta") is None else report.get("f_beta"),
        )["beta"],
    }


def _hold_key(key: str) -> str:
    return f"_held_{key}"


def _epoch() -> int:
    return int(st.session_state.get("ui_settings_epoch") or 0)


def _live_key(key: str) -> str:
    """Widget key. A new epoch remounts the control so the browser shows the new value."""
    epoch = _epoch()
    if epoch <= 0:
        return key
    return f"{key}__e{epoch}"


def _apply_widget_defaults(base: dict) -> None:
    defaults = _widget_defaults(base)
    for key, value in defaults.items():
        st.session_state[_live_key(key)] = value
        st.session_state[key] = value
        st.session_state[_hold_key(key)] = value


def _backfill_held_widgets(base: dict) -> None:
    """Keep values for sidebar widgets that are not drawn this run.

    Streamlit drops a widget key when that control is not rendered. The held
    copy is what the pipeline reads for a closed parameter group.
    """
    for key, value in _widget_defaults(base).items():
        held = _hold_key(key)
        if held not in st.session_state:
            st.session_state[held] = st.session_state.get(key, value)


def _restore_widgets(keys: tuple[str, ...], *, force: bool) -> None:
    """Put held values back before the widgets are drawn.

    A closed group drops its widget, and Streamlit can leave the old key
    behind as an empty value. Restoring only when the key is missing keeps
    that empty value and throws away the edit. ``force`` is for the run that
    opens the group. Later edits in that group must not be overwritten.
    """
    for key in keys:
        held = _hold_key(key)
        live = _live_key(key)
        if held not in st.session_state:
            continue
        if force or live not in st.session_state:
            st.session_state[live] = st.session_state[held]
            st.session_state[key] = st.session_state[held]


def _capture_widget_values(keys: tuple[str, ...]) -> None:
    """Copy widgets that are still on screen. Called before they unmount."""
    for key in keys:
        live = _live_key(key)
        if live in st.session_state:
            st.session_state[_hold_key(key)] = st.session_state[live]
            st.session_state[key] = st.session_state[live]


_SIDEBAR_SECTIONS: dict[str, tuple[str, ...]] = {
    "Loading": (
        "ui_input_dir",
        "ui_filename_pattern",
        "ui_overlap_fraction",
        "ui_pixel_size_um",
        "ui_workers",
    ),
    "Preprocessing": (
        "ui_denoise",
        "ui_denoise_sigma",
        "ui_flatten_illumination",
        "ui_flatten_sigma",
        "ui_contrast_stretch",
    ),
    "Detection": (
        "ui_method",
        "ui_particles_bright",
        "ui_fft_peak_threshold",
        "ui_fft_mask",
        "ui_tophat_radius",
        "ui_min_size_um",
        "ui_max_size_um",
        "ui_blob_threshold",
        "ui_min_prominence",
        "ui_min_confidence_pct",
        "ui_blob_min_sigma",
        "ui_blob_max_sigma",
        "ui_local_snr_sigma",
        "ui_edge_soften_sigma",
        "ui_edge_soften_strength",
        "ui_edge_exclude_px",
        "ui_min_circularity",
        "ui_structure_neighbor_px",
        "ui_ml_enabled",
        "ui_ml_threshold",
    ),
    "Output": (
        "ui_output_dir",
        "ui_merge_radius_px",
        "ui_size_aggregation",
        "ui_target_mb",
        "ui_downsample",
        "ui_f_beta",
    ),
}


def _selected_profile(base: dict) -> dict:
    if st.session_state.get("ui_settings_nsew"):
        return _nsew_config(base)
    return base


def _assign_widget(key: str, value, *, previous_epoch: int | None = None) -> None:
    """Write a sidebar value onto the current epoch key and the held copy.

    The bare key is the widget key when the epoch is still 0. Writing it
    after that widget exists raises, so callers that just bumped the epoch
    pass the epoch from before the bump and skip the bare key in that case.
    """
    st.session_state[_live_key(key)] = value
    st.session_state[_hold_key(key)] = value
    if previous_epoch is None or previous_epoch > 0:
        st.session_state[key] = value


def _apply_beta_preset(base: dict, *, previous_epoch: int | None = None, clear_trained: bool = False) -> None:
    """Point the ML model and threshold slider at the selected beta's preset."""
    beta = float(
        st.session_state.get(_live_key("ui_f_beta"), st.session_state.get("ui_f_beta", 1.0))
    )
    preset = preset_for_beta(beta_preset_rows(base), beta)
    _assign_widget("ui_ml_threshold", float(preset["threshold"]), previous_epoch=previous_epoch)
    st.session_state["ui_beta_model_path"] = str(preset["model_path"])
    if clear_trained:
        st.session_state.pop("ui_trained_model", None)


def _on_beta_change(base: dict) -> None:
    """Keep every other sidebar edit, then load this beta's model and threshold."""
    previous_epoch = _epoch()
    captured: dict[str, object] = {}
    for keys in _SIDEBAR_SECTIONS.values():
        for key in keys:
            live = _live_key(key)
            if live in st.session_state:
                captured[key] = st.session_state[live]
            elif key in st.session_state:
                captured[key] = st.session_state[key]
            elif _hold_key(key) in st.session_state:
                captured[key] = st.session_state[_hold_key(key)]
    st.session_state["ui_settings_epoch"] = previous_epoch + 1
    for key, value in captured.items():
        _assign_widget(key, value, previous_epoch=previous_epoch)
    _apply_beta_preset(base, previous_epoch=previous_epoch, clear_trained=True)


def _apply_standard_settings(
    base: dict, folder: str | None = None, *, clear_trained: bool = False
) -> None:
    """Load Uniform or NSEW standard values. Keep the tile folder just chosen."""
    previous = _live_key("ui_input_dir")
    chosen = folder if folder is not None else str(
        st.session_state.get(previous) or st.session_state.get("ui_input_dir") or ""
    )
    previous_epoch = _epoch()
    st.session_state["ui_settings_epoch"] = previous_epoch + 1
    _apply_widget_defaults(_selected_profile(base))
    live = _live_key("ui_input_dir")
    st.session_state[live] = chosen
    if previous_epoch > 0:
        st.session_state["ui_input_dir"] = chosen
    st.session_state[_hold_key("ui_input_dir")] = chosen
    _apply_beta_preset(base, previous_epoch=previous_epoch, clear_trained=clear_trained)


def _on_input_folder(base: dict) -> None:
    chosen = str(st.session_state.get(_live_key("ui_input_dir")) or "")
    previous = str(st.session_state.get(_hold_key("ui_input_dir")) or "")
    if chosen == previous:
        return
    _apply_standard_settings(base, chosen)


def _keep_filename_pattern(base: dict) -> None:
    """A cleared pattern matches every file and then crashes. Restore the selected profile."""
    live = _live_key("ui_filename_pattern")
    current = str(st.session_state.get(live) or st.session_state.get("ui_filename_pattern") or "")
    held = str(st.session_state.get(_hold_key("ui_filename_pattern")) or "")
    if current.strip() or held.strip():
        return
    _apply_standard_settings(base)


def _widget_value(key: str):
    live = _live_key(key)
    if live in st.session_state:
        st.session_state[_hold_key(key)] = st.session_state[live]
        st.session_state[key] = st.session_state[live]
        return st.session_state[live]
    if key in st.session_state:
        st.session_state[_hold_key(key)] = st.session_state[key]
        return st.session_state[key]
    return st.session_state[_hold_key(key)]


def _nsew_config(base: dict) -> dict:
    return apply_nsew_settings(base)


def _on_reset_standard(base: dict) -> None:
    _apply_standard_settings(base, clear_trained=True)


def _on_settings_profile(base: dict) -> None:
    _apply_standard_settings(base, clear_trained=True)


def _sidebar(base: dict) -> dict:
    config = deepcopy(base)
    if st.session_state.get("ui_schema") != UI_SCHEMA:
        if not isinstance(st.session_state.get("ui_settings_nsew"), bool):
            st.session_state["ui_settings_nsew"] = False
        _apply_standard_settings(base, clear_trained=True)
        st.session_state["ui_schema"] = UI_SCHEMA
        st.session_state["ui_initialized"] = True
        st.session_state["ui_sidebar_frontend_seeded"] = True
    _backfill_held_widgets(base)
    _keep_filename_pattern(base)

    st.sidebar.button(
        "Reset to standard values",
        on_click=_on_reset_standard,
        args=(base,),
        help="Restore sidebar settings from config.yaml (Uniform) or "
        "nsew_config.yaml (NSEW), matching the toggle above.",
        width="stretch",
    )
    if not isinstance(st.session_state.get("ui_settings_nsew"), bool):
        st.session_state["ui_settings_nsew"] = False
    uniform_col, nsew_col = st.sidebar.columns([1, 1.4], vertical_alignment="center")
    uniform_col.markdown(
        """
<style>
div[data-testid="stHorizontalBlock"]:has(#settings-uniform) [data-testid="stColumn"]:first-child,
div[data-testid="stHorizontalBlock"]:has(#settings-uniform) [data-testid="stColumn"]:first-child [data-testid="stElementContainer"],
div[data-testid="stHorizontalBlock"]:has(#settings-uniform) [data-testid="stColumn"]:first-child [data-testid="stMarkdown"],
div[data-testid="stHorizontalBlock"]:has(#settings-uniform) [data-testid="stColumn"]:first-child [data-testid="stMarkdownContainer"] {
  margin: 0 !important;
  padding: 0 !important;
  height: 24px !important;
  min-height: 24px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: flex-end !important;
}
#settings-uniform {
  margin: 0;
  font-size: 14px;
  line-height: 21px;
  font-weight: 400;
}
</style>
<div id="settings-uniform">Uniform</div>
""",
        unsafe_allow_html=True,
    )
    nsew_col.toggle(
        "NSEW",
        key="ui_settings_nsew",
        on_change=_on_settings_profile,
        args=(base,),
        help="Off is Uniform (config.yaml, v4 threshold 0.23). "
        "On loads NSEW pixel size, preprocess, detection, and ML threshold "
        "from nsew_config.yaml.",
    )

    def _select_sidebar_group(name: str) -> None:
        current = str(st.session_state.get("ui_sidebar_group") or "")
        _capture_widget_values(_SIDEBAR_SECTIONS.get(current, ()))
        st.session_state["ui_sidebar_group"] = name

    for name in ["Loading", "Preprocessing", "Detection", "Output"]:
        st.sidebar.button(
            name,
            key=f"ui_sidebar_tab_{name}",
            on_click=_select_sidebar_group,
            args=(name,),
            type="primary" if st.session_state.get("ui_sidebar_group") == name else "secondary",
            width="stretch",
        )
    section = st.session_state.get("ui_sidebar_group") or "Loading"
    # First time these controls are on the page, write the saved value into
    # session state in this run. Otherwise the browser draws the widget
    # minimum, and the next click stores that minimum over the real setting.
    seed_frontend = not st.session_state.get("ui_sidebar_frontend_seeded")
    # Every group stays mounted. Hiding a group used to drop its widgets, and
    # the next click then saved the empty leftover instead of the edit.
    hidden = [
        name
        for name in ("Loading", "Preprocessing", "Detection", "Output")
        if name != section
    ]
    if hidden:
        selectors = ",".join(
            f'[data-testid="stSidebar"] .st-key-sidebar_sec_{name}' for name in hidden
        )
        st.sidebar.markdown(
            f"<style>{selectors}{{display:none !important;}}</style>",
            unsafe_allow_html=True,
        )

    with st.sidebar.container(key="sidebar_sec_Loading"):
        _restore_widgets(_SIDEBAR_SECTIONS["Loading"], force=seed_frontend)
        folder_choices = input_folder_labels()
        current_folder = str(
            st.session_state.get(_live_key("ui_input_dir"))
            or st.session_state.get("ui_input_dir")
            or ""
        )
        if current_folder and current_folder not in folder_choices:
            folder_choices = [current_folder, *folder_choices]
        folder_help = (
            "Folder under ASML SE/Inputs. Suggestions appear as you type, "
            "for example Groundup/Groundup v7 (06-10) or R3 04-08. "
            "Choosing a folder loads the standard Uniform or NSEW settings, "
            "matching the switch above."
        )
        if folder_choices:
            st.selectbox(
                "Tile folder",
                folder_choices,
                placeholder="Start typing a folder name",
                accept_new_options=True,
                filter_mode="fuzzy",
                key=_live_key("ui_input_dir"),
                on_change=_on_input_folder,
                args=(base,),
                help=folder_help,
            )
        else:
            st.text_input(
                "Tile folder",
                key=_live_key("ui_input_dir"),
                on_change=_on_input_folder,
                args=(base,),
                help=folder_help,
            )
        st.text_input(
            "Filename pattern (regex)",
            key=_live_key("ui_filename_pattern"),
            help="Default: R{run}_{row}_{col}_{mag}X.tif or .bmp, e.g. R3_5_32_5X.tif. "
            "Leave this as is for Groundup quad and all-lights shots. "
            "Names that do not match still run detection; they are left out of the mosaic.",
        )
        st.slider(
            "Tile overlap fraction",
            min_value=0.0,
            max_value=0.5,
            step=0.01,
            key=_live_key("ui_overlap_fraction"),
        )
        st.number_input(
            "Pixel size (µm)",
            min_value=0.001,
            step=0.1,
            format="%.3f",
            key=_live_key("ui_pixel_size_um"),
            help="These 5X tiles are 1 pixel = 0.96 µm (960 nm).",
        )
        st.number_input(
            "Parallel workers",
            min_value=0,
            max_value=32,
            key=_live_key("ui_workers"),
            help="Tiles processed in parallel with processes (not threads). 0 uses all CPU cores.",
        )
    with st.sidebar.container(key="sidebar_sec_Preprocessing"):
        _restore_widgets(_SIDEBAR_SECTIONS["Preprocessing"], force=seed_frontend)
        st.checkbox("Denoise", key=_live_key("ui_denoise"))
        st.number_input(
            "Denoise sigma", min_value=0.0, key=_live_key("ui_denoise_sigma")
        )
        st.checkbox("Flatten illumination", key=_live_key("ui_flatten_illumination"))
        st.number_input(
            "Flatten sigma",
            min_value=1.0,
            key=_live_key("ui_flatten_sigma"),
            help="Large values flatten only illumination. Small values halo region edges.",
        )
        st.checkbox("Contrast stretch", key=_live_key("ui_contrast_stretch"))
    with st.sidebar.container(key="sidebar_sec_Detection"):
        _restore_widgets(_SIDEBAR_SECTIONS["Detection"], force=seed_frontend)
        st.selectbox(
            "Background suppression",
            options=["fft", "tophat"],
            key=_live_key("ui_method"),
        )
        st.checkbox("Particles are bright", key=_live_key("ui_particles_bright"))
        st.slider(
            "FFT peak threshold", 0.05, 0.95, key=_live_key("ui_fft_peak_threshold")
        )
        st.selectbox(
            "FFT notch source",
            options=["per_tile", "shared"],
            key=_live_key("ui_fft_mask"),
            help="per_tile: notch each tile from its own spectrum. "
            "shared: reuse the first tile (uniform lattice only).",
        )
        st.number_input(
            "Top-hat radius (px)",
            min_value=1,
            key=_live_key("ui_tophat_radius"),
            help="Must be larger than the biggest particle. Default 50 (≈100 µm flakes). "
            "A small radius erases large debris.",
        )
        st.number_input(
            "Min size (µm)",
            min_value=0.1,
            step=0.5,
            format="%.2f",
            key=_live_key("ui_min_size_um"),
            help="Keep particles larger than this. Default 10 µm (≈10.4 px at 0.96 µm/px). "
            "Raise Blob min sigma with this so the detector does not hunt for smaller specks.",
        )
        st.number_input(
            "Max size (µm)",
            min_value=0.1,
            step=0.5,
            format="%.2f",
            key=_live_key("ui_max_size_um"),
            help="Upper size cap. Default 100 µm. Blob max sigma 36 is about 100 µm "
            "at 0.96 µm/px. Raise Top-hat radius with this so large flakes are not erased.",
        )
        st.slider(
            "Blob threshold", 0.001, 0.5, key=_live_key("ui_blob_threshold")
        )
        st.slider(
            "Min prominence",
            min_value=0.0,
            max_value=0.8,
            key=_live_key("ui_min_prominence"),
            help="Ignore residual weaker than this (0–1). Raises this to drop grid/texture; "
            "lowers it to pick up dimmer specks. 0 disables.",
        )
        st.slider(
            "Confidence score (%)",
            min_value=0,
            max_value=100,
            step=1,
            key=_live_key("ui_min_confidence_pct"),
            help="Drop blobs whose residual at the peak is below this score. 0 disables.",
        )
        st.number_input(
            "Blob min sigma", min_value=0.5, key=_live_key("ui_blob_min_sigma")
        )
        st.number_input(
            "Blob max sigma", min_value=1.0, key=_live_key("ui_blob_max_sigma")
        )
        st.number_input(
            "Local SNR sigma (0 = auto)",
            min_value=0.0,
            key=_live_key("ui_local_snr_sigma"),
            help="Envelope for local energy. 0 auto-enables on FFT residuals "
            "(3.5 × blob max sigma) and stays off for top-hat.",
        )
        st.number_input(
            "Edge soften sigma",
            min_value=0.0,
            key=_live_key("ui_edge_soften_sigma"),
            help="Coarse blur before the region-border gradient. 0 disables.",
        )
        st.number_input(
            "Edge soften strength",
            min_value=0.0,
            key=_live_key("ui_edge_soften_strength"),
            help="How hard to attenuate residual on area borders. 0 disables.",
        )
        st.number_input(
            "Edge exclude (px)",
            min_value=0.0,
            key=_live_key("ui_edge_exclude_px"),
            help="Ignore blobs this close to a region border. "
            "Pad corners and box rims are always dropped.",
        )
        st.slider(
            "Min circularity",
            min_value=0.0,
            max_value=1.0,
            key=_live_key("ui_min_circularity"),
            help="Drop very elongated hits. Keep low: real debris is irregular, not round. 0 keeps every blob.",
        )
        st.number_input(
            "Structure neighbor radius (px)",
            min_value=0.0,
            key=_live_key("ui_structure_neighbor_px"),
            help="Drop blobs that have 2+ other blobs this close (chains and grids). 0 disables.",
        )
        st.checkbox(
            "Apply ML filter",
            key=_live_key("ui_ml_enabled"),
            help="Score each DoG proposal with the trained patch model. "
            "Leave off until you have trained (`python -m src.training`). "
            "Detection stays classical; the model only drops false positives.",
        )
        ml_min, ml_max = 0.0, 1.0
        live_threshold = _live_key("ui_ml_threshold")
        stored = float(
            st.session_state.get(live_threshold, st.session_state.get("ui_ml_threshold", 0.0))
        )
        clamped = min(ml_max, max(ml_min, stored))
        if stored != clamped:
            st.session_state[live_threshold] = clamped
            st.session_state["ui_ml_threshold"] = clamped
        st.slider(
            "ML P(particle) threshold",
            min_value=ml_min,
            max_value=ml_max,
            step=0.01,
            key=_live_key("ui_ml_threshold"),
            help="Keep blobs with at least this probability. 0 keeps every proposal.",
        )
    with st.sidebar.container(key="sidebar_sec_Output"):
        _restore_widgets(_SIDEBAR_SECTIONS["Output"], force=seed_frontend)
        st.text_input(
            "Output folder",
            key=_live_key("ui_output_dir"),
            help="Default is ASML SE/Outputs. A name like Single v7 goes in Outputs/Single v7.",
        )
        st.number_input(
            "Merge radius (px)", min_value=0.0, key=_live_key("ui_merge_radius_px")
        )
        st.selectbox(
            "Cluster size aggregation",
            options=["max", "mean"],
            key=_live_key("ui_size_aggregation"),
        )
        st.number_input(
            "Stitched mosaic size (MB)",
            min_value=1.0,
            max_value=200.0,
            step=1.0,
            key=_live_key("ui_target_mb"),
            help="Each tile is downsampled before stitching so the full mosaic is about this large.",
        )
        st.number_input(
            "Overlay downsample (0 = auto)",
            min_value=0,
            key=_live_key("ui_downsample"),
            help="0 chooses a factor from the target size. A value > 0 forces that integer factor.",
        )
        beta_rows = beta_preset_rows(base)
        beta_options = [float(row["beta"]) for row in beta_rows]
        st.selectbox(
            "Beta",
            options=beta_options,
            format_func=lambda value: f"{float(value):.2f}",
            key=_live_key("ui_f_beta"),
            on_change=_on_beta_change,
            args=(base,),
            help="Chooses the Fβ score and the ML model and starting threshold "
            "for the next detection run. 1 is F1. A later threshold edit stays "
            "until you pick another beta.",
        )
        selected = preset_for_beta(beta_rows, float(_widget_value("ui_f_beta")))
        st.caption(
            f"{Path(selected['model_path']).name} · threshold {float(selected['threshold']):.2f}"
        )
    st.session_state["ui_sidebar_frontend_seeded"] = True

    config["input_dir"] = _widget_value("ui_input_dir")
    config["output_dir"] = _widget_value("ui_output_dir")
    config["filename_pattern"] = _widget_value("ui_filename_pattern")
    config["overlap_fraction"] = float(_widget_value("ui_overlap_fraction"))
    config["pixel_size_nm"] = _um_to_nm(float(_widget_value("ui_pixel_size_um")))
    pipe = config.setdefault("pipeline", {})
    pipe["workers"] = int(_widget_value("ui_workers"))

    pre = config.setdefault("preprocessing", {})
    pre["denoise"] = bool(_widget_value("ui_denoise"))
    pre["denoise_sigma"] = float(_widget_value("ui_denoise_sigma"))
    pre["flatten_illumination"] = bool(_widget_value("ui_flatten_illumination"))
    pre["flatten_sigma"] = float(_widget_value("ui_flatten_sigma"))
    pre["contrast_stretch"] = bool(_widget_value("ui_contrast_stretch"))

    det = config.setdefault("detection", {})
    det["method"] = _widget_value("ui_method")
    det["particles_bright"] = bool(_widget_value("ui_particles_bright"))
    det["fft_peak_threshold"] = float(_widget_value("ui_fft_peak_threshold"))
    det["fft_mask"] = _widget_value("ui_fft_mask")
    det["tophat_radius"] = int(_widget_value("ui_tophat_radius"))
    det["min_size_nm"] = _um_to_nm(float(_widget_value("ui_min_size_um")))
    det["max_size_nm"] = _um_to_nm(float(_widget_value("ui_max_size_um")))
    det["blob_threshold"] = float(_widget_value("ui_blob_threshold"))
    det["min_prominence"] = float(_widget_value("ui_min_prominence"))
    det["min_confidence"] = float(_widget_value("ui_min_confidence_pct")) / 100.0
    det["blob_min_sigma"] = float(_widget_value("ui_blob_min_sigma"))
    det["blob_max_sigma"] = float(_widget_value("ui_blob_max_sigma"))
    det["local_snr_sigma"] = float(_widget_value("ui_local_snr_sigma"))
    det["edge_soften_sigma"] = float(_widget_value("ui_edge_soften_sigma"))
    det["edge_soften_strength"] = float(_widget_value("ui_edge_soften_strength"))
    det["edge_exclude_px"] = float(_widget_value("ui_edge_exclude_px"))
    det["min_circularity"] = float(_widget_value("ui_min_circularity"))
    det["structure_neighbor_px"] = float(_widget_value("ui_structure_neighbor_px"))

    ml = config.setdefault("ml", {})
    ml["enabled"] = bool(_widget_value("ui_ml_enabled"))
    ml_min, ml_max = 0.0, 1.0
    ml["threshold"] = min(ml_max, max(ml_min, float(_widget_value("ui_ml_threshold"))))
    beta_choice = preset_for_beta(beta_preset_rows(base), float(_widget_value("ui_f_beta")))
    if not st.session_state.get("ui_beta_model_path"):
        st.session_state["ui_beta_model_path"] = str(beta_choice["model_path"])
    trained_model = st.session_state.get("ui_trained_model")
    if trained_model:
        ml["model_path"] = str(trained_model)
    else:
        ml["model_path"] = str(
            st.session_state.get("ui_beta_model_path") or beta_choice["model_path"]
        )

    meas = config.setdefault("measurement", {})
    meas["merge_radius_px"] = float(_widget_value("ui_merge_radius_px"))
    meas["size_aggregation"] = _widget_value("ui_size_aggregation")
    report = config.setdefault("report", {})
    report["target_mb"] = float(_widget_value("ui_target_mb"))
    report["downsample"] = int(_widget_value("ui_downsample"))
    report["f_beta"] = float(_widget_value("ui_f_beta"))
    return config


def _apply_finished_pipeline() -> None:
    result = claim_pipeline_result()
    if result is None:
        return
    if result["stopped"]:
        st.warning("Pipeline stopped. Results from this run were not saved.")
        return
    if result["trace"]:
        st.error(result["error"] or "Pipeline failed.")
        st.code(result["trace"])
        return
    if result["error"]:
        st.error(result["error"])
        return

    table = result["table"]
    mosaic = result["mosaic"]
    config = result["config"]
    st.session_state["table"] = table
    st.session_state["mosaic"] = mosaic
    st.session_state["config"] = config
    st.session_state.pop("overlay", None)
    st.session_state.pop("overlay_key", None)
    out_dir = resolve_output_dir(config["output_dir"])
    csv_path = out_dir / "particles.csv"
    xlsx_path = out_dir / "particles.xlsx"
    write_csv(table, csv_path)
    write_xlsx(table, xlsx_path, config)
    write_last_pipeline(config, csv_path)
    st.session_state["last_pipeline_csv"] = str(csv_path)
    st.session_state["ui_show_last_run"] = True
    st.success(f"Wrote {csv_path} and {xlsx_path} ({len(table)} particles).")


def _pipeline_buttons(config: dict, *, running: bool) -> None:
    run_col, stop_col = st.columns(2)
    if run_col.button(
        "Run pipeline",
        type="primary",
        disabled=running,
        key="ui_run_pipeline",
        width="stretch",
    ):
        try:
            start_pipeline(config)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        else:
            st.rerun(scope="app")
    if stop_col.button(
        "Stop pipeline",
        disabled=not running,
        key="ui_stop_pipeline",
        width="stretch",
    ):
        stop_pipeline()
        st.rerun(scope="fragment")


@st.fragment(run_every=0.5)
def _pipeline_running(config: dict) -> None:
    status = pipeline_status()
    if not status["running"]:
        st.session_state["_pipeline_was_running"] = False
        st.rerun(scope="app")
        return
    st.session_state["_pipeline_was_running"] = True
    total = max(status["total"], 1)
    fraction = min(status["current"], total) / total if status["total"] else 0.0
    label = status["name"] or "Starting…"
    if status["total"]:
        label = f"{status['current']}/{status['total']} {label}"
    if status["stopping"]:
        label = f"Stopping after the current tile. {label}"
    st.progress(fraction, text=label)
    _pipeline_buttons(config, running=True)


def _pipeline_controls(config: dict) -> None:
    if pipeline_status()["running"] or st.session_state.get("_pipeline_was_running"):
        _pipeline_running(config)
        return
    _pipeline_buttons(config, running=False)


def _render_training(config: dict) -> None:
    """Run ``python -m src.training`` from the app."""
    st.caption(
        "Trains particle vs not-particle from labels. "
        "Patch mode reads unmarked tiles. Residual mode uses a particles.csv. "
        "Circled label JPEGs are not used."
    )
    if "ui_train_labels" not in st.session_state:
        st.session_state["ui_train_labels"] = str(default_label_path())
    if "ui_train_input_dir" not in st.session_state:
        st.session_state["ui_train_input_dir"] = str(config.get("input_dir") or "")

    mode = st.selectbox(
        "Mode",
        options=["patch", "residual"],
        key="ui_train_mode",
        help="patch: HOG ExtraTrees on unmarked crops. residual: ExtraTrees on detector columns.",
    )
    labels = st.text_input("Labels CSV", key="ui_train_labels")
    output = st.text_input(
        "Output model",
        key="ui_train_output",
        placeholder="Blank writes the next models/particle_clf_vN.joblib",
    )
    detections = st.text_input(
        "Detections CSV",
        key="ui_train_detections",
        placeholder="particles.csv — required for residual unless a default exists",
    )
    input_dir = st.text_input(
        "Tile folder",
        key="ui_train_input_dir",
        help="Patch mode reads tiles from here. Blank uses config.yaml.",
    )
    with st.expander("More training settings"):
        config_path = st.text_input(
            "Config YAML",
            key="ui_train_config",
            placeholder=str(ROOT / "config.yaml"),
        )
        pattern = st.text_input(
            "Filename pattern",
            key="ui_train_pattern",
            placeholder="Blank keeps the config pattern",
        )
        min_size = st.number_input(
            "Min size (nm)",
            min_value=0.0,
            step=1000.0,
            key="ui_train_min_size_nm",
        )
        manifest = st.text_input("Manifest CSV", key="ui_train_manifest")
        v4_model = st.text_input(
            "Previous model for hard negatives",
            key="ui_train_v4_model",
            help="Optional. Upweights false positives that this model still scores highly.",
        )
        if "ui_train_hard_weight" not in st.session_state:
            st.session_state["ui_train_hard_weight"] = 3.0
        if "ui_train_hard_cutoff" not in st.session_state:
            st.session_state["ui_train_hard_cutoff"] = 0.20
        hard_weight = st.number_input(
            "Hard-negative weight",
            min_value=1.0,
            step=0.5,
            key="ui_train_hard_weight",
        )
        hard_cutoff = st.number_input(
            "Hard-negative cutoff",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="ui_train_hard_cutoff",
        )

    if st.button("Run training", type="primary", key="ui_train_run"):
        destination = Path(output.strip()) if output.strip() else next_versioned_model_path()
        argv = ["--mode", mode, "--labels", labels.strip(), "--output", str(destination)]
        if detections.strip():
            argv.extend(["--detections", detections.strip()])
        if config_path.strip():
            argv.extend(["--config", config_path.strip()])
        if input_dir.strip():
            argv.extend(["--input-dir", input_dir.strip()])
        if pattern.strip():
            argv.extend(["--filename-pattern", pattern.strip()])
        if float(min_size) > 0:
            argv.extend(["--min-size-nm", str(float(min_size))])
        if manifest.strip():
            argv.extend(["--manifest", manifest.strip()])
        if v4_model.strip():
            argv.extend(
                [
                    "--v4-model",
                    v4_model.strip(),
                    "--hard-neg-weight",
                    str(float(hard_weight)),
                    "--hard-neg-cutoff",
                    str(float(hard_cutoff)),
                ]
            )
        log = io.StringIO()
        try:
            with st.spinner("Training… this can take several minutes."):
                with contextlib.redirect_stdout(log):
                    code = train_main(argv)
        except SystemExit as exc:
            st.error(str(exc.code) if exc.code else "Training stopped.")
        except Exception as exc:
            st.exception(exc)
        else:
            if code == 0:
                st.session_state["ui_trained_model"] = str(destination)
                st.success(
                    f"Wrote {destination}. Detection will use this model until you change beta or reset standard values."
                )
            else:
                st.error(f"Training exited with code {code}.")
        text = log.getvalue().strip()
        if text:
            st.code(text)

    active = st.session_state.get("ui_trained_model")
    if active:
        st.caption(f"Detection model: `{active}`")


def _saved_mosaic_path(active_config: dict) -> Path | None:
    """JPEG written for this output folder, only if it belongs to the current CSV.

    A run with no row/column names does not stitch. An older ``mosaic_overlay.jpg``
    left in the same folder is from a previous run and must not be shown.
    """
    output_dir = active_config.get("output_dir")
    if not output_dir:
        return None
    folder = resolve_output_dir(output_dir)
    path = folder / "mosaic_overlay.jpg"
    if not path.is_file():
        return None
    csv_path = folder / "particles.csv"
    if csv_path.is_file() and path.stat().st_mtime + 2 < csv_path.stat().st_mtime:
        return None
    return path


def _finished_run_state(config: dict) -> tuple[pd.DataFrame | None, object, dict]:
    """Session results from the run that just finished, else the saved last run."""
    table = st.session_state.get("table")
    mosaic = st.session_state.get("mosaic")
    active_config = deepcopy(st.session_state.get("config", config))
    active_config["report"] = config.get("report", active_config.get("report", {}))
    if isinstance(table, pd.DataFrame):
        return table, mosaic, active_config
    last_csv, run_config = load_last_pipeline(config)
    if last_csv is None or not last_csv.is_file():
        return None, None, active_config
    loaded = deepcopy(config)
    for key, value in run_config.items():
        if value not in (None, ""):
            loaded[key] = value
    loaded["report"] = config.get("report", loaded.get("report", {}))
    return load_particles_csv(last_csv), None, loaded


def _render_size_threshold_table(table: pd.DataFrame) -> None:
    """Detected particles with equivalent diameter at least 10, 20, 25, 50, 75 µm."""
    st.caption("Detected particles at least this large (equivalent circular diameter).")
    st.dataframe(particles_at_least(table), width="content", hide_index=True)


def _render_size_bins(table: pd.DataFrame) -> None:
    st.subheader("Size distribution (µm)")
    if table.empty or "size" not in table.columns:
        st.info("This run has no particle sizes to bin.")
        return
    sizes_um = table["size"].to_numpy(dtype=float) / NM_PER_UM
    n_bins = min(20, max(int(table["size"].nunique()), 1))
    counts, bin_edges = np.histogram(sizes_um, bins=n_bins)
    hist_df = pd.DataFrame(
        {"count": counts},
        index=[f"{bin_edges[i]:.2f}–{bin_edges[i + 1]:.2f}" for i in range(len(counts))],
    )
    st.bar_chart(hist_df)


def _render_finished_run(config: dict) -> None:
    """Table, size bins, and mosaic from the latest finished pipeline run."""
    table, mosaic, active_config = _finished_run_state(config)
    if table is None:
        st.info("Run the pipeline once. Last run keeps that table, size bins, and mosaic.")
        return

    stats = summary_stats(table)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Particles", stats["particle_count"])
    c2.metric("Mean size (µm)", f"{_nm_to_um(stats['size_mean']):.2f}")
    c3.metric("Min size (µm)", f"{_nm_to_um(stats['size_min']):.2f}")
    c4.metric("Max size (µm)", f"{_nm_to_um(stats['size_max']):.2f}")
    _render_size_threshold_table(table)

    st.subheader("Results")
    display = table.copy()
    if not display.empty and "size" in display.columns:
        display.insert(display.columns.get_loc("size") + 1, "size_um", display["size"] / NM_PER_UM)
    preview_rows = 5_000
    if len(display) > preview_rows:
        st.caption(
            f"Showing the first {preview_rows:,} of {len(display):,} rows. "
            "Download the CSV or Excel file for the full table."
        )
        st.dataframe(display.head(preview_rows), width="stretch", hide_index=True)
    else:
        st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "CSV `size` is equivalent circular diameter in nm "
        "(from the connected footprint on the photo, not the FFT/top-hat residual "
        "and not the DoG scale bin). `size_um` is the same value in µm."
    )

    _render_size_bins(table)

    dl_csv, dl_xlsx = st.columns(2)
    with dl_csv:
        st.download_button(
            "Download CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="particles.csv",
            mime="text/csv",
            key="last_run_download_csv",
        )
    with dl_xlsx:
        st.download_button(
            "Download Excel",
            data=excel_bytes(table, active_config),
            file_name="particles.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="last_run_download_xlsx",
        )

    placements = getattr(mosaic, "placements", None)
    if not placements:
        saved = _saved_mosaic_path(active_config)
        st.subheader("Stitched mosaic")
        if saved is not None:
            st.image(str(saved), caption="Saved mosaic from the last run", width="stretch")
            st.download_button(
                "Download stitched mosaic",
                data=saved.read_bytes(),
                file_name="mosaic_overlay.jpg",
                mime="image/jpeg",
                key="last_run_download_saved_mosaic",
            )
            return
        st.info(
            "No tiles had row/column (or x/y) in the filename, so nothing was "
            "stitched. Detection results and the size bins are above. Open "
            "Verification → Pointer to inspect each image."
        )
        return

    placed_names = {placement.name for placement in mosaic.placements}
    if "source_tile" in table.columns:
        table_tiles = {
            str(name).replace("\\", "/").rsplit("/", 1)[-1]
            for name in table["source_tile"].astype(str)
        }
        leftover = table_tiles - placed_names
        if leftover:
            st.caption(
                f"{len(leftover)} file(s) did not match the filename pattern and were "
                "detected but not stitched into this mosaic."
            )

    st.subheader("Stitched mosaic")
    max_h = max(int(mosaic.full_height), 1)
    max_w = max(int(mosaic.full_width), 1)
    factor = overlay_downsample(mosaic, active_config, crop=None)
    out_h = max(int(np.ceil(mosaic.full_height / factor)), 0)
    out_w = max(int(np.ceil(mosaic.full_width / factor)), 0)
    st.caption(
        f"Full mosaic is {mosaic.full_height}×{mosaic.full_width} px. "
        f"Tiles are downsampled ×{factor} before stitching → {out_h}×{out_w} "
        f"(~{out_h * out_w * 3 / (1024 * 1024):.1f} MB uncompressed RGB)."
    )

    overlay_key = (
        "full",
        float(active_config.get("report", {}).get("target_mb", 20.0)),
        int(active_config.get("report", {}).get("downsample", 0) or 0),
        len(table),
    )
    if st.session_state.get("overlay_key") != overlay_key:
        try:
            st.session_state["overlay"] = overlay_markers(
                mosaic,
                table,
                active_config,
                crop=None,
            )
        except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
            st.error(f"Mosaic stitching failed: {exc}")
        else:
            st.session_state["overlay_key"] = overlay_key
            mosaic_path = resolve_output_dir(active_config.get("output_dir")) / "mosaic_overlay.jpg"
            mosaic_path.parent.mkdir(parents=True, exist_ok=True)
            mosaic_path.write_bytes(encode_overlay_jpeg(st.session_state["overlay"]))
    overlay = st.session_state.get("overlay")
    if overlay is not None:
        st.image(overlay, caption="All tiles, stitched at reduced resolution", width="stretch")
        st.download_button(
            "Download stitched mosaic",
            data=encode_overlay_jpeg(overlay),
            file_name="mosaic_overlay.jpg",
            mime="image/jpeg",
            key="last_run_download_mosaic",
        )

    with st.expander("Inspect a crop (full-resolution mosaic pixels)"):
        col_a, col_b, col_c, col_d = st.columns(4)
        crop_y = col_a.number_input("Crop y", min_value=0, max_value=max_h, value=0)
        crop_x = col_b.number_input("Crop x", min_value=0, max_value=max_w, value=0)
        crop_h = col_c.number_input(
            "Crop height", min_value=1, max_value=max_h, value=min(max_h, 2048)
        )
        crop_w = col_d.number_input(
            "Crop width", min_value=1, max_value=max_w, value=min(max_w, 2048)
        )
        if st.button("Render crop"):
            crop = (int(crop_y), int(crop_x), int(crop_h), int(crop_w))
            cropped = overlay_markers(mosaic, table, active_config, crop=crop)
            st.image(cropped, caption="Cropped region with detections", width="stretch")


def _render_detection(config: dict) -> None:
    st.caption(
        "A folder with N, S, E, and W is combined first. "
        "Detection runs on those shots and on the combi, plus the uniform all shot when the folder has one."
    )
    _apply_finished_pipeline()
    _pipeline_controls(config)


config = _sidebar(_base_config())

keep_shadow_inputs()
keep_capture_inputs()
keep_stage_inputs()

pipeline_tab, verification_tab, training_tab, shadow_tab, redirector_tab, capture_tab = st.tabs(
    ["Main Pipeline", "Verification", "Training", "Shadow", "Redirector", "Capture"],
    on_change="rerun",
    key="main_tabs",
)

if pipeline_tab.open:
    with pipeline_tab:
        _render_detection(config)

        def _toggle_last_run() -> None:
            st.session_state["ui_show_last_run"] = not bool(
                st.session_state.get("ui_show_last_run")
            )

        def _toggle_last_summary() -> None:
            st.session_state["ui_show_last_summary"] = not bool(
                st.session_state.get("ui_show_last_summary")
            )

        show_last_run = bool(st.session_state.get("ui_show_last_run"))
        show_summary = bool(st.session_state.get("ui_show_last_summary"))
        last_col, summary_col = st.columns(2)
        last_col.button(
            "Hide last run" if show_last_run else "Last run",
            key="ui_toggle_last_run",
            on_click=_toggle_last_run,
            width="stretch",
        )
        summary_col.button(
            "Hide summary of last run" if show_summary else "Summary of last run",
            key="ui_toggle_last_summary",
            on_click=_toggle_last_summary,
            width="stretch",
        )
        if show_last_run:
            st.subheader("Last run")
            _render_finished_run(config)
        if show_summary:
            st.subheader("Summary of last run")
            render_labeled_tiles_tab(
                config,
                ROOT,
                st.session_state.get("mosaic"),
                st.session_state.get("table"),
            )

if verification_tab.open:
    with verification_tab:
        tinder_tab, pointer_tab, database_tab = st.tabs(
            ["Tinder", "Pointer", "Database"],
            on_change="rerun",
            key="verification_tabs",
        )
        if tinder_tab.open:
            with tinder_tab:
                render_label_tab(
                    config,
                    ROOT,
                    st.session_state.get("table"),
                    st.session_state.get("mosaic"),
                )
        if pointer_tab.open:
            with pointer_tab:
                render_tile_inspect_tab(
                    config,
                    ROOT,
                    st.session_state.get("table"),
                    st.session_state.get("mosaic"),
                )
        if database_tab.open:
            with database_tab:
                show_area_line = st.toggle(
                    "Red area line",
                    value=True,
                    key="database_red_area_line",
                    help="Outline the measured particle area on the particle crops.",
                )
                groundup_tab, glovebox_tab = st.tabs(
                    ["Groundup", "Glovebox"],
                    on_change="rerun",
                    key="database_campaign_tabs",
                )
                for campaign_tab, campaign in (
                    (groundup_tab, "groundup"),
                    (glovebox_tab, "glovebox"),
                ):
                    if not campaign_tab.open:
                        continue
                    with campaign_tab:
                        particle_col, not_particle_col = st.columns(2)
                        with particle_col:
                            render_collection_tab(
                                ROOT,
                                "particle",
                                "Particles",
                                n_columns=2,
                                campaign=campaign,
                                config=config,
                                show_area_line=show_area_line,
                            )
                        with not_particle_col:
                            render_collection_tab(
                                ROOT,
                                "not_particle",
                                "Not particles",
                                n_columns=2,
                                campaign=campaign,
                            )

if training_tab.open:
    with training_tab:
        _render_training(config)

if shadow_tab.open:
    with shadow_tab:
        render_shadow_tab(config)

if redirector_tab.open:
    with redirector_tab:
        render_redirector_tab()

if capture_tab.open:
    with capture_tab:
        render_capture_tab()
else:
    suspend_live_view()

