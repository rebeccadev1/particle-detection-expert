from __future__ import annotations

import contextlib
import importlib
import io
import sys
from copy import deepcopy
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.capture.camera_automation as _camera
import src.overall.io.tile_loader as _tile_loader

importlib.reload(_camera)
importlib.reload(_tile_loader)
latest_run_folder = _camera.latest_run_folder
input_folder_labels = _tile_loader.input_folder_labels

from src.capture.capture_ui import keep_capture_inputs, keep_stage_inputs, render_capture_tab, suspend_live_view
from src.overall.config import (
    DEFAULT_INPUT_DIR,
    apply_nsew_settings,
    beta_preset_rows,
    load_config,
    preset_for_beta,
    advance_versioned_output_name,
    next_versioned_output_name,
    output_choice_label,
    output_folder_choices,
    resolve_output_dir,
)
from src.overall.io.results_writer import excel_bytes, result_table, write_csv, write_xlsx
import src.verification.inspect as _inspect
import src.verification.ui as _verification_ui
import src.main_pipeline.report.report_generator as _report

importlib.reload(_inspect)
importlib.reload(_verification_ui)
importlib.reload(_report)
label_colors_for_hits = _inspect.label_colors_for_hits

from src.verification.ui import (
    labels_store,
    render_collection_tab,
    render_label_tab,
    render_labeled_tiles_tab,
    render_tile_inspect_tab,
    request_last_run,
)
from src.verification.queue import load_last_pipeline, load_particles_csv, write_last_pipeline
from src.main_pipeline.ml.infer import next_versioned_model_path
from src.training.train import default_label_path, main as train_main
from src.main_pipeline.job import claim_pipeline_result, pipeline_status, start_pipeline, stop_pipeline
from src.redirector.redirector import render_redirector_tab
from src.main_pipeline.report.report_generator import (
    encode_overlay_jpeg,
    overlay_downsample,
    overlay_rgb,
    particles_at_least,
    size_distribution,
    summary_pdf_name,
    summary_stats,
    unstitched_tile_paths,
    unstitched_tile_rgb,
)
from src.main_pipeline.runner import load_report_mosaic

draw_overlay_markers = _report.draw_overlay_markers
from src.shadow.shadow import keep_shadow_inputs, render_shadow_tab

st.set_page_config(
    page_title="Particle Detection Software",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    [data-testid="stVerticalBlock"][class*="st-key-block-"] {
        background-color: rgba(15, 35, 140, 0.04);
        border: none;
        box-shadow: none;
        border-radius: 0.6rem;
        padding: 0.35rem 0.9rem 0.7rem;
        margin-bottom: 0.75rem;
    }
    [class*="st-key-block-"] [data-testid="stNumberInputContainer"],
    [class*="st-key-block-"] [data-testid="stSelectbox"] [role="group"],
    [class*="st-key-block-"] [data-baseweb="input"],
    [class*="st-key-block-"] [data-baseweb="textarea"] {
        background-color: #e2e6f2;
    }
    /* Body is 16px + 4pt. Each step below is smaller and lighter.
       h1 app title, h2 section, h3 subsection, then figures, then body. */
    html {
        font-size: calc(16px + 4pt) !important;
    }
    h1, h2, h3 {
        line-height: 1.25 !important;
        margin: 0 !important;
    }
    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        padding: 0.35rem 0 0.5rem !important;
    }
    h2 {
        font-size: 1.5rem !important;
        font-weight: 650 !important;
        padding: 0.15rem 0 0.4rem !important;
    }
    h3 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        padding: 0.1rem 0 0.3rem !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 1.05rem !important;
        font-weight: 650 !important;
        padding: 0.1rem 0 0.35rem !important;
    }
    .summary-stats {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-start;
        gap: 0.85rem 2.25rem;
        margin: 0.15rem 0 0.35rem;
    }
    .summary-stat-label {
        font-size: 0.95rem;
        font-weight: 500;
        line-height: 1.6;
        margin: 0 0 0.25rem;
    }
    .summary-stat-value {
        font-size: 1rem;
        font-weight: 400;
        line-height: 1.6;
    }
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span {
        font-size: 0.8rem !important;
        font-weight: 400 !important;
    }
    [data-testid="stWidgetLabel"] p {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMarkdownContainer"] p {
        font-weight: 400;
    }
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 550 !important;
    }
    .stButton button, .stDownloadButton button {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stTable"] th {
        font-size: 0.8rem !important;
        font-weight: 650 !important;
    }
    [data-testid="stTable"] td {
        font-size: 0.9rem !important;
        font-weight: 400 !important;
    }
    [class*="st-key-block-last-run"] [data-testid="stVegaLiteChart"],
    [class*="st-key-block-last-run"] .vega-embed {
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 3.75rem !important;
    }
    [data-testid="stElementContainer"]:has(#dot-decimals),
    [data-testid="stElementContainer"]:has(#sidebar-sec-hide) {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.html(
    """
    <div id="dot-decimals"></div>
    <script>
    (function () {
      if (window.__dotDecimals) return;
      window.__dotDecimals = true;
      var typeDesc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "type");
      Object.defineProperty(HTMLInputElement.prototype, "type", {
        configurable: true,
        enumerable: typeDesc.enumerable,
        get: function () { return typeDesc.get.call(this); },
        set: function (value) {
          if (value === "number") {
            typeDesc.set.call(this, "text");
            this.setAttribute("inputmode", "decimal");
            this.setAttribute("lang", "en");
            return;
          }
          typeDesc.set.call(this, value);
        }
      });
      var nativeSetAttribute = Element.prototype.setAttribute;
      Element.prototype.setAttribute = function (name, value) {
        if (this instanceof HTMLInputElement && name === "type" && value === "number") {
          nativeSetAttribute.call(this, "type", "text");
          nativeSetAttribute.call(this, "inputmode", "decimal");
          nativeSetAttribute.call(this, "lang", "en");
          return;
        }
        nativeSetAttribute.call(this, name, value);
      };
      document.querySelectorAll('input[type="number"]').forEach(function (input) {
        input.type = "text";
      });
      document.addEventListener("beforeinput", function (event) {
        var target = event.target;
        if (!(target instanceof HTMLInputElement)) return;
        if (!target.closest('[data-testid="stNumberInput"]')) return;
        if (!event.data || event.data.indexOf(",") === -1) return;
        event.preventDefault();
        var start = target.selectionStart == null ? target.value.length : target.selectionStart;
        var end = target.selectionEnd == null ? start : target.selectionEnd;
        var insert = event.data.replace(/,/g, ".");
        var next = target.value.slice(0, start) + insert + target.value.slice(end);
        var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
        setter.call(target, next);
        var caret = start + insert.length;
        target.setSelectionRange(caret, caret);
        target.dispatchEvent(new Event("input", { bubbles: true }));
      }, true);
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)
st.title("Particle Detection Software")

NM_PER_UM = 1000.0
UI_SCHEMA = "mixed_layout_v28"

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
        "ui_output_dir": "",
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
        "ui_edge_exclude_px": float(det["edge_exclude_px"]),
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
    for key, value in _widget_defaults(base).items():
        held = _hold_key(key)
        if held not in st.session_state:
            st.session_state[held] = st.session_state.get(key, value)

def _restore_widgets(keys: tuple[str, ...], *, force: bool) -> None:
    for key in keys:
        held = _hold_key(key)
        live = _live_key(key)
        if held not in st.session_state:
            continue
        if force or live not in st.session_state:
            st.session_state[live] = st.session_state[held]
            st.session_state[key] = st.session_state[held]

def _capture_widget_values(keys: tuple[str, ...]) -> None:
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

    st.session_state[_live_key(key)] = value
    st.session_state[_hold_key(key)] = value
    if previous_epoch is None or previous_epoch > 0:
        st.session_state[key] = value

def _apply_beta_preset(base: dict, *, previous_epoch: int | None = None, clear_trained: bool = False) -> None:
    """Point the ML model and threshold slider at this mode's beta preset."""
    beta = float(
        st.session_state.get(_live_key("ui_f_beta"), st.session_state.get("ui_f_beta", 1.0))
    )
    preset = preset_for_beta(beta_preset_rows(_selected_profile(base)), beta)
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
    """Load Uniform or NSEW standard values. Keep the tile folder just chosen.

    The output folder is ``{tile folder} vN``. An empty tile folder leaves the
    output folder empty so the control shows its placeholder.
    """
    previous = _live_key("ui_input_dir")
    profile = _selected_profile(base)
    if folder is None:
        chosen = str(
            st.session_state.get(previous) or st.session_state.get("ui_input_dir") or ""
        )
        if not chosen:
            chosen = str(profile.get("input_dir") or "")
    else:
        chosen = folder
    previous_epoch = _epoch()
    st.session_state["ui_settings_epoch"] = previous_epoch + 1
    _apply_widget_defaults(profile)
    live = _live_key("ui_input_dir")
    st.session_state[live] = chosen
    if previous_epoch > 0:
        st.session_state["ui_input_dir"] = chosen
    st.session_state[_hold_key("ui_input_dir")] = chosen
    _assign_widget(
        "ui_output_dir",
        next_versioned_output_name(chosen),
        previous_epoch=previous_epoch,
    )
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

def _on_profile_choice(base: dict) -> None:
    st.session_state["ui_settings_nsew"] = st.session_state.get("ui_settings_profile") == "NSEW"
    _on_settings_profile(base)

def _prepare_ui_state(base: dict) -> None:
    if st.session_state.get("ui_schema") != UI_SCHEMA:
        if not isinstance(st.session_state.get("ui_settings_nsew"), bool):
            st.session_state["ui_settings_nsew"] = False
        st.session_state["ui_settings_profile"] = (
            "NSEW" if st.session_state.get("ui_settings_nsew") else "Uniform"
        )
        _apply_standard_settings(base, "", clear_trained=True)
        st.session_state["ui_schema"] = UI_SCHEMA
        st.session_state["ui_initialized"] = True
        st.session_state["ui_param_mount"] = "session"
    elif st.session_state.get("ui_param_mount") != "session":
        _apply_standard_settings(base, clear_trained=False)
        st.session_state["ui_param_mount"] = "session"
        st.session_state["ui_sidebar_frontend_seeded"] = False
    _backfill_held_widgets(base)
    _keep_filename_pattern(base)
    _apply_capture_suggestion()
    if not isinstance(st.session_state.get("ui_settings_nsew"), bool):
        st.session_state["ui_settings_nsew"] = False
    if st.session_state.get("ui_settings_profile") not in ("Uniform", "NSEW"):
        st.session_state["ui_settings_profile"] = (
            "NSEW" if st.session_state.get("ui_settings_nsew") else "Uniform"
        )


def _capture_folder_label(folder: Path) -> str:
    try:
        return folder.resolve().relative_to(DEFAULT_INPUT_DIR.resolve()).as_posix()
    except ValueError:
        return folder.name


def _apply_capture_suggestion() -> None:
    """Remember the newest capture folder so Tile folder can offer it first.

    The control itself stays empty until a folder is chosen, so the placeholder
    stays visible.
    """
    latest_path = latest_run_folder()
    if latest_path is None:
        return
    st.session_state["ui_suggested_capture"] = _capture_folder_label(latest_path)


def _render_adjust_parameters(base: dict) -> None:
    st.radio(
        "Settings profile",
        ("Uniform", "NSEW"),
        horizontal=True,
        key="ui_settings_profile",
        persist_state="session",
        on_change=_on_profile_choice,
        args=(base,),
        help="Uniform or NSEW standard settings.",
    )
    st.button(
        "Reset to standard values",
        on_click=_on_reset_standard,
        args=(base,),
        help="Restore settings for the profile (Uniform or NSEW) selected above.",
        width="stretch",
    )

    def _select_sidebar_group(name: str) -> None:
        current = str(st.session_state.get("ui_sidebar_group") or "")
        _capture_widget_values(_SIDEBAR_SECTIONS.get(current, ()))
        st.session_state["ui_sidebar_group"] = "" if current == name else name

    for column, name in zip(
        st.columns(4),
        ["Loading", "Preprocessing", "Detection", "Output"],
    ):
        with column:
            st.button(
                name,
                key=f"ui_sidebar_tab_{name}",
                on_click=_select_sidebar_group,
                args=(name,),
                type="primary" if st.session_state.get("ui_sidebar_group") == name else "secondary",
                width="stretch",
            )
    section = st.session_state.get("ui_sidebar_group") or ""

    seed_frontend = not st.session_state.get("ui_sidebar_frontend_seeded")
   
    hidden = [
        name
        for name in ("Loading", "Preprocessing", "Detection", "Output")
        if name != section
    ]
    if hidden:
        selectors = ",".join(
            f'[data-testid="stLayoutWrapper"]:has(> .st-key-sidebar_sec_{name})'
            for name in hidden
        )
        st.markdown(
            f'<div id="sidebar-sec-hide"></div>'
            f"<style>{selectors}{{display:none !important;}}</style>",
            unsafe_allow_html=True,
        )

    with st.container(key="sidebar_sec_Loading"):
        _restore_widgets(_SIDEBAR_SECTIONS["Loading"], force=seed_frontend)
        folder_choices = input_folder_labels()
        suggestion = st.session_state.get("ui_suggested_capture")
        if isinstance(suggestion, str) and suggestion in folder_choices:
            folder_choices = [suggestion, *(name for name in folder_choices if name != suggestion)]
        input_key = _live_key("ui_input_dir")
        current_folder = str(
            st.session_state.get(input_key)
            or st.session_state.get("ui_input_dir")
            or ""
        )
        if current_folder and current_folder not in folder_choices:
            folder_choices = [current_folder, *folder_choices]
        elif not current_folder:
            st.session_state.pop(input_key, None)
            st.session_state["ui_input_dir"] = ""
            st.session_state[_hold_key("ui_input_dir")] = ""
        folder_help = (
            "Folder under ASML SE/Inputs. Suggestions appear as you type."
        )
        if folder_choices:
            input_kwargs = {}
            if input_key not in st.session_state:
                input_kwargs["index"] = None
            st.selectbox(
                "Tile folder",
                folder_choices,
                placeholder="Start typing a folder name",
                accept_new_options=True,
                filter_mode="fuzzy",
                persist_state="session",
                key=input_key,
                on_change=_on_input_folder,
                args=(base,),
                help=folder_help,
                **input_kwargs,
            )
        else:
            st.text_input(
                "Tile folder",
                persist_state="session",
                key=input_key,
                placeholder="Start typing a folder name",
                on_change=_on_input_folder,
                args=(base,),
                help=folder_help,
            )
        st.slider(
            "Tile overlap fraction",
            min_value=0.0,
            max_value=0.5,
            step=0.01,
            persist_state="session",
            key=_live_key("ui_overlap_fraction"),
            help="The fraction of the tile that is overlapped by the next tile.",
        )
        st.number_input(
            "Pixel size (µm)",
            min_value=0.001,
            step=0.1,
            format="%.3f",
            persist_state="session",
            key=_live_key("ui_pixel_size_um"),
            help="Select the physical pixel size.",
        )
        st.number_input(
            "Parallel workers",
            min_value=1,
            max_value=32,
            persist_state="session",
            key=_live_key("ui_workers"),
            help="How many tiles to process at once. The minimum is 1.",
        )
        with st.expander("Filename pattern"):
            st.text_input(
                "Filename pattern (regex)",
                persist_state="session",
                key=_live_key("ui_filename_pattern"),
                help="Names that do not match still run detection; they are left out of the mosaic.",
            )
    with st.container(key="sidebar_sec_Preprocessing"):
        _restore_widgets(_SIDEBAR_SECTIONS["Preprocessing"], force=seed_frontend)
        st.checkbox(
            "Denoise",
            persist_state="session",
            key=_live_key("ui_denoise"),
            help="Small blur so sensor speckle is not detected as particles.",
        )
        st.number_input(
            "Denoise sigma",
            min_value=0.0,
            persist_state="session",
            key=_live_key("ui_denoise_sigma"),
            help="Blur radius in pixels; larger values also round off real particles.",
        )
        st.checkbox(
            "Flatten illumination",
            persist_state="session",
            key=_live_key("ui_flatten_illumination"),
            help="Divide each tile by a wide blur of itself to remove vignetting.",
        )
        st.number_input(
            "Flatten sigma",
            min_value=1.0,
            persist_state="session",
            key=_live_key("ui_flatten_sigma"),
            help="Large values flatten only illumination; small values halo region edges.",
        )
        st.checkbox(
            "Contrast stretch",
            persist_state="session",
            key=_live_key("ui_contrast_stretch"),
            help="Map each tile's 1st–99th percentiles onto 0–1 so contrast matches before detection.",
        )
    with st.container(key="sidebar_sec_Detection"):
        _restore_widgets(_SIDEBAR_SECTIONS["Detection"], force=seed_frontend)
        st.selectbox(
            "Background suppression",
            options=["fft", "tophat"],
            persist_state="session",
            key=_live_key("ui_method"),
            help="fft notches the repeating lattice and falls back to top-hat when a tile has no peaks.",
        )
        st.checkbox(
            "Particles are bright",
            persist_state="session",
            key=_live_key("ui_particles_bright"),
            help="Off inverts the residual so dark debris is found the same way.",
        )
        st.slider(
            "FFT peak threshold",
            0.05,
            0.95,
            persist_state="session",
            key=_live_key("ui_fft_peak_threshold"),
            help="Lower notches more of the lattice and can also remove real structure.",
        )
        st.selectbox(
            "FFT notch source",
            options=["per_tile", "shared"],
            persist_state="session",
            key=_live_key("ui_fft_mask"),
            help="per_tile notches each tile from its own spectrum; shared reuses the first tile.",
        )
        st.number_input(
            "Top-hat radius (px)",
            min_value=1,
            persist_state="session",
            key=_live_key("ui_tophat_radius"),
            help="Must be larger than the biggest particle, or large debris is erased.",
        )
        st.number_input(
            "Min size (µm)",
            min_value=0.1,
            step=0.5,
            format="%.2f",
            persist_state="session",
            key=_live_key("ui_min_size_um"),
            help="Keep particles larger than this.",
        )
        st.number_input(
            "Max size (µm)",
            min_value=0.1,
            step=0.5,
            format="%.2f",
            persist_state="session",
            key=_live_key("ui_max_size_um"),
            help="Drop particles larger than this.",
        )
        st.slider(
            "Blob threshold",
            0.001,
            0.5,
            persist_state="session",
            key=_live_key("ui_blob_threshold"),
            help="DoG peak height after 0–1 scaling; lower keeps dimmer candidates.",
        )
        st.slider(
            "Min prominence",
            min_value=0.0,
            max_value=0.8,
            persist_state="session",
            key=_live_key("ui_min_prominence"),
            help="Zero residual weaker than this before detection; 0 disables.",
        )
        st.slider(
            "Confidence score (%)",
            min_value=0,
            max_value=100,
            step=1,
            persist_state="session",
            key=_live_key("ui_min_confidence_pct"),
            help="Drop blobs whose residual at the peak is below this; 0 disables.",
        )
        st.number_input(
            "Blob min sigma",
            min_value=0.5,
            persist_state="session",
            key=_live_key("ui_blob_min_sigma"),
            help="Smallest scale searched; diameter is about 2.8 times sigma.",
        )
        st.number_input(
            "Blob max sigma",
            min_value=1.0,
            persist_state="session",
            key=_live_key("ui_blob_max_sigma"),
            help="Largest scale searched; raise it with Max size so large flakes are found.",
        )
        st.number_input(
            "Local SNR sigma (0 = auto)",
            min_value=0.0,
            persist_state="session",
            key=_live_key("ui_local_snr_sigma"),
            help="Envelope for local energy; 0 turns this on for FFT residuals and off for top-hat.",
        )
        st.number_input(
            "Edge soften sigma",
            min_value=0.0,
            persist_state="session",
            key=_live_key("ui_edge_soften_sigma"),
            help="Coarse blur before the region-border gradient; 0 disables.",
        )
        st.number_input(
            "Edge soften strength",
            min_value=0.0,
            persist_state="session",
            key=_live_key("ui_edge_soften_strength"),
            help="How hard to attenuate residual on area borders; 0 disables.",
        )
        st.number_input(
            "Edge exclude (px)",
            min_value=0.0,
            persist_state="session",
            key=_live_key("ui_edge_exclude_px"),
            help="Ignore blobs this close to a region border, including pad corners.",
        )
        st.slider(
            "Min circularity",
            min_value=0.0,
            max_value=1.0,
            persist_state="session",
            key=_live_key("ui_min_circularity"),
            help="Drop elongated hits; 0 keeps every blob.",
        )
        st.number_input(
            "Structure neighbor radius (px)",
            min_value=0.0,
            persist_state="session",
            key=_live_key("ui_structure_neighbor_px"),
            help="Drop blobs with at least this profile's minimum neighbours this close. Uniform uses 3, NSEW uses 2. 0 disables.",
        )
        st.checkbox(
            "Apply ML filter",
            persist_state="session",
            key=_live_key("ui_ml_enabled"),
            help="Score each proposal with the trained model and drop false positives.",
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
            persist_state="session",
            key=_live_key("ui_ml_threshold"),
            help="Keep blobs with at least this probability; 0 keeps every proposal.",
        )
    with st.container(key="sidebar_sec_Output"):
        _restore_widgets(_SIDEBAR_SECTIONS["Output"], force=seed_frontend)
        output_key = _live_key("ui_output_dir")
        stored_output = st.session_state.get(output_key, st.session_state.get("ui_output_dir"))
        current_output = output_choice_label("" if stored_output is None else str(stored_output))
        if current_output and str(stored_output or "") != current_output:
            st.session_state[output_key] = current_output
            st.session_state["ui_output_dir"] = current_output
            st.session_state[_hold_key("ui_output_dir")] = current_output
        elif not current_output:
            st.session_state.pop(output_key, None)
            st.session_state["ui_output_dir"] = ""
            st.session_state[_hold_key("ui_output_dir")] = ""
        tile_folder = str(
            st.session_state.get(_live_key("ui_input_dir"))
            or st.session_state.get("ui_input_dir")
            or ""
        )
        standard_output = next_versioned_output_name(tile_folder)
        output_choices = output_folder_choices(current_output, standard_output)
        output_help = (
            "Folder under ASML SE/Outputs."
        )
        if output_choices:
            output_kwargs = {}
            if output_key not in st.session_state:
                output_kwargs["index"] = None
            st.selectbox(
                "Output folder",
                output_choices,
                placeholder="Start typing a folder name",
                accept_new_options=True,
                filter_mode="fuzzy",
                persist_state="session",
                key=output_key,
                help=output_help,
                **output_kwargs,
            )
        else:
            st.text_input(
                "Output folder",
                persist_state="session",
                key=output_key,
                placeholder="Start typing a folder name",
                help=output_help,
            )
        st.number_input(
            "Merge radius (px)",
            min_value=0.0,
            persist_state="session",
            key=_live_key("ui_merge_radius_px"),
            help="Hits closer than this on overlapping tiles count as one particle; 0 skips the merge.",
        )
        st.selectbox(
            "Cluster size aggregation",
            options=["max", "mean"],
            persist_state="session",
            key=_live_key("ui_size_aggregation"),
            help="max keeps the largest diameter in a merged cluster; mean averages them.",
        )
        st.number_input(
            "Stitched mosaic size (MB)",
            min_value=1.0,
            max_value=200.0,
            step=1.0,
            persist_state="session",
            key=_live_key("ui_target_mb"),
            help="Each tile is downsampled before stitching so the full mosaic is about this large.",
        )
        st.number_input(
            "Overlay downsample (0 = auto)",
            min_value=0,
            persist_state="session",
            key=_live_key("ui_downsample"),
            help="0 picks a factor from the target size; a value above 0 forces that factor.",
        )
        beta_rows = beta_preset_rows(base)
        beta_options = [float(row["beta"]) for row in beta_rows]
        st.selectbox(
            "Beta",
            options=beta_options,
            format_func=lambda value: f"{float(value):.2f}",
            persist_state="session",
            key=_live_key("ui_f_beta"),
            on_change=_on_beta_change,
            args=(base,),
            help="Chooses the Fβ score and the ML model and starting threshold for the next run.",
        )
    st.session_state["ui_sidebar_frontend_seeded"] = True


def _config_from_state(base: dict) -> dict:
    config = deepcopy(base)
    chosen_input = _widget_value("ui_input_dir")
    config["input_dir"] = "" if chosen_input is None else str(chosen_input)
    chosen_output = _widget_value("ui_output_dir")
    chosen_output = "" if chosen_output is None else str(chosen_output).strip()
    if not chosen_output:
        chosen_output = next_versioned_output_name(str(config.get("input_dir") or ""))
    config["output_dir"] = chosen_output
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
    beta_choice = preset_for_beta(
        beta_preset_rows(_selected_profile(base)), float(_widget_value("ui_f_beta"))
    )
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
    if result["trace"] or result["error"]:
        st.error("The pipeline stopped before saving this run. Fix the cause below, then run it again.")
        with st.expander("Error detail"):
            if result["error"]:
                st.write(result["error"])
            if result["trace"]:
                st.code(result["trace"])
        return

    table = result["table"]
    mosaic = result["mosaic"]
    config = result["config"]
    st.session_state["table"] = table
    st.session_state["mosaic"] = mosaic
    st.session_state["config"] = config
    st.session_state.pop("overlay", None)
    st.session_state.pop("overlay_plain", None)
    st.session_state.pop("overlay_key", None)
    st.session_state.pop("unstitched_images", None)
    st.session_state.pop("unstitched_key", None)
    out_dir = resolve_output_dir(config["output_dir"])
    csv_path = out_dir / "particles.csv"
    xlsx_path = out_dir / "particles.xlsx"
    write_csv(table, csv_path)
    write_xlsx(table, xlsx_path, config)
    for report_name in (
        "last_pipeline.json",
        summary_pdf_name(config),
        "mosaic_overlay.png",
        "mosaic_plain.png",
    ):
        report_path = out_dir / report_name
        if report_path.is_file():
            report_path.touch()
    write_last_pipeline(config, csv_path)
    st.session_state["last_pipeline_csv"] = str(csv_path)
    st.session_state["ui_show_last_run"] = True
    st.session_state["load_last_run"] = True
    st.session_state["load_labeled_tiles"] = False
    stats = summary_stats(table)
    st.success(
        f"{stats['particle_count']} particles. Mean size {_nm_to_um(stats['size_mean']):.2f} µm."
    )
    st.caption(f"Saved to {out_dir}.")


def _advance_output_folder(config: dict) -> None:
    """Point Output folder at the next ``{tile folder} vN`` after this run."""
    nxt = advance_versioned_output_name(
        config.get("input_dir"),
        config.get("output_dir"),
    )
    if not nxt or nxt == output_choice_label(config.get("output_dir")):
        return
    live = _live_key("ui_output_dir")
    st.session_state[live] = nxt
    st.session_state["ui_output_dir"] = nxt
    st.session_state[_hold_key("ui_output_dir")] = nxt


def _pipeline_buttons(config: dict, *, running: bool) -> None:
    with st.container(horizontal=True):
        if st.button(
            "Run pipeline",
            type="primary",
            disabled=running,
            key="ui_run_pipeline",
        ):
            try:
                start_pipeline(config)
            except (ValueError, RuntimeError) as exc:
                detail = str(exc).strip() or type(exc).__name__
                st.error(f"The pipeline did not start. {detail.splitlines()[0]}")
                with st.expander("Error detail"):
                    st.code(detail)
            else:
                _advance_output_folder(config)
                if pipeline_status()["pending"]:
                    st.rerun(scope="app")
                else:
                    st.rerun(scope="fragment")
        if st.button(
            "Stop pipeline",
            disabled=not running,
            key="ui_stop_pipeline",
        ):
            stop_pipeline()
            st.rerun(scope="fragment")


@st.fragment(run_every=0.5)
def _pipeline_controls(config: dict) -> None:
    """Progress and the run/stop buttons stay in this fragment.

    Moving the buttons out while the pipeline is idle, then back in on the
    next run, leaves the previous row on screen for a moment.
    """
    status = pipeline_status()
    if status["running"]:
        total = max(status["total"], 1)
        fraction = min(status["current"], total) / total if status["total"] else 0.0
        label = status["name"] or "Starting…"
        if status["total"]:
            label = f"{status['current']}/{status['total']} {label}"
        if status["stopping"]:
            label = f"Stopping after the current tile. {label}"
        st.progress(fraction, text=label)
        _pipeline_buttons(config, running=True)
        return
    if status["pending"]:
        st.caption("Updating results…")
        st.rerun(scope="app")
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


def _write_mosaic_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Failed to write mosaic image to {path}")


def _saved_plain_mosaic(saved: Path, config: dict) -> np.ndarray:
    """Unlabeled mosaic for a saved run. Stitches once, then reuses the file."""
    plain_path = saved.parent / "mosaic_plain.png"
    cached = st.session_state.get("saved_plain_rgb")
    key = str(plain_path.resolve())
    if (
        st.session_state.get("saved_plain_key") == key
        and isinstance(cached, np.ndarray)
        and plain_path.is_file()
    ):
        return cached
    if plain_path.is_file():
        bgr = cv2.imread(str(plain_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Could not read {plain_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        import importlib

        import src.main_pipeline.runner as pipeline_runner

        if not hasattr(pipeline_runner, "load_report_mosaic"):
            pipeline_runner = importlib.reload(pipeline_runner)
        mosaic = pipeline_runner.load_report_mosaic(config)
        if not mosaic.placements:
            raise FileNotFoundError("No tiles could be placed for this mosaic.")
        rgb = overlay_rgb(mosaic, config, crop=None)
        _write_mosaic_png(rgb, plain_path)
    st.session_state["saved_plain_rgb"] = rgb
    st.session_state["saved_plain_key"] = key
    return rgb


def _labels_stamp() -> int:
    """Change when Tinder writes a mark, so the mosaic is drawn again."""
    path = ROOT / "labels" / "all_labels.csv"
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _mosaic_labels_on() -> bool:
    """Checkbox for detection circles on the stitched mosaic. On by default."""
    return st.checkbox(
        "Labels",
        value=True,
        key="mosaic_show_labels",
        help="Circles on detected particles. Green is a particle you marked, "
        "red is not a particle, blue is not sure, and orange is not marked yet.",
    )


def _mosaic_with_review_labels(
    saved: Path,
    table: pd.DataFrame,
    config: dict,
) -> np.ndarray | None:
    """Draw this run's Tinder colours on the plain mosaic.

    The file saved at the end of the pipeline is always orange, because the
    marks did not exist yet. This redraws those circles from the label table.
    """
    plain_path = saved.parent / "mosaic_plain.png"
    if not plain_path.is_file():
        return None
    try:
        stamp = (plain_path.stat().st_mtime_ns, _labels_stamp(), int(len(table)))
    except OSError:
        return None
    cache_key = ("review_mosaic", str(plain_path), stamp)
    if st.session_state.get("review_mosaic_key") == cache_key:
        cached = st.session_state.get("review_mosaic_rgb")
        if isinstance(cached, np.ndarray):
            return cached
    bgr = cv2.imread(str(plain_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        mosaic = load_report_mosaic(config)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not getattr(mosaic, "placements", None) or int(mosaic.full_width) < 1:
        return None
    scale = rgb.shape[1] / float(mosaic.full_width)
    colors = label_colors_for_hits(
        table,
        labels_store(ROOT).load(),
        run_dir=str(config.get("input_dir") or "") or None,
    )
    marked = draw_overlay_markers(
        rgb, mosaic, table, config, colors=colors, scale=scale
    )
    st.session_state["review_mosaic_key"] = cache_key
    st.session_state["review_mosaic_rgb"] = marked
    return marked


def _saved_mosaic_path(active_config: dict) -> Path | None:
    """Mosaic written for this output folder, only if it belongs to the current CSV.

    A run with no row/column names does not stitch. An older mosaic left in the
    same folder is from a previous run and must not be shown.
    """
    output_dir = active_config.get("output_dir")
    if not output_dir:
        return None
    folder = resolve_output_dir(output_dir)
    path = next(
        (
            candidate
            for name in ("mosaic_overlay.png", "mosaic_overlay.jpg")
            if (candidate := folder / name).is_file()
        ),
        None,
    )
    if path is None:
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
    if not st.session_state.get("load_last_run"):
        return None, None, active_config
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
    st.dataframe(
        particles_at_least(table),
        width=250,
        height=300,
        hide_index=True,
    )


def _render_size_bins(table: pd.DataFrame, min_size_um: float) -> None:
    st.subheader("Size distribution (µm)")
    if table.empty or "size" not in table.columns:
        st.info("This run has no particle sizes to bin.")
        return
    frame = size_distribution(table, min_size_um).rename_axis("size").reset_index()
    chart = (
        alt.Chart(frame)
        .mark_bar(color="#0F238C")
        .encode(
            x=alt.X(
                "size:N",
                sort=None,
                title="Size (µm)",
                axis=alt.Axis(labelAngle=-40, labelPadding=4, titlePadding=8, labelLimit=90),
            ),
            y=alt.Y(
                "count:Q",
                title="Amount of particles",
                axis=alt.Axis(labelPadding=8, titlePadding=12),
            ),
        )
        .properties(height=320, padding={"left": 10, "right": 10, "top": 6, "bottom": 6})
        .configure_view(stroke=None)
    )
    st.altair_chart(chart, width="stretch", theme=None)


def _render_summary_stats(stats: dict) -> None:
    """Label above value, same arrangement as the Settings profile control."""
    items = (
        ("Particles", f"{int(stats['particle_count'])}"),
        ("Mean size (µm)", f"{_nm_to_um(stats['size_mean']):.2f}"),
        ("Min size (µm)", f"{_nm_to_um(stats['size_min']):.2f}"),
        ("Max size (µm)", f"{_nm_to_um(stats['size_max']):.2f}"),
    )
    cells = "".join(
        "<div class='summary-stat'>"
        f"<div class='summary-stat-label'>{label}</div>"
        f"<div class='summary-stat-value'>{value}</div>"
        "</div>"
        for label, value in items
    )
    st.markdown(f"<div class='summary-stats'>{cells}</div>", unsafe_allow_html=True)


def _summary_heading(config: dict) -> str:
    """``Summary of {input folder name}``, or ``Last run`` when no folder is known."""
    raw = str(config.get("input_dir") or "").strip().strip('"').strip("'").rstrip("/\\")
    name = Path(raw).name if raw else ""
    if not name:
        return "Last run"
    return f"Summary of {name}"


def _request_labeled_tiles() -> None:
    st.session_state["load_labeled_tiles"] = True


def _render_unstitched_images(
    paths: list[Path],
    table: pd.DataFrame,
    config: dict,
    show_labels: bool,
) -> None:
    """Show each image that has no row/column pattern, without stitching."""
    if not paths:
        return
    stamps: list[str] = []
    for path in paths:
        try:
            stamps.append(f"{path}:{path.stat().st_mtime_ns}")
        except OSError:
            stamps.append(str(path))
    report = config.get("report") or {}
    key = (
        tuple(stamps),
        bool(show_labels),
        int(len(table)),
        float(report.get("target_mb", 20.0)),
        int(report.get("downsample", 0) or 0),
    )
    cached = st.session_state.get("unstitched_images")
    if st.session_state.get("unstitched_key") != key or not isinstance(cached, list):
        built: list[tuple[str, np.ndarray | None, str | None]] = []
        with st.spinner("Loading images…"):
            for path in paths:
                try:
                    rgb = unstitched_tile_rgb(path, table, config, labels=show_labels)
                except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                    built.append((path.name, None, str(exc)))
                else:
                    built.append((path.name, rgb, None))
        st.session_state["unstitched_images"] = built
        st.session_state["unstitched_key"] = key
        cached = built
    for name, rgb, error in cached:
        if error or rgb is None:
            st.error(f"{name} could not be shown.")
            with st.expander(f"Error detail for {name}"):
                st.code(error or "")
            continue
        st.image(rgb, caption=name, width="stretch")


def _render_finished_run(config: dict) -> bool:
    """Table, size bins, and mosaic from the latest finished pipeline run."""
    table, mosaic, active_config = _finished_run_state(config)
    if table is None:
        with st.container(border=True, key="block-last-run"):
            saved_csv, saved_config = load_last_pipeline(config)
            st.header(_summary_heading(saved_config))
            st.button(
                "Load last run",
                key="run_load_last_run",
                on_click=request_last_run,
                disabled=saved_csv is None or not Path(saved_csv).is_file(),
            )
            st.caption("Last run stays unloaded until you click Load last run.")
        return False

    with st.container(border=True, key="block-last-run"):
        st.header(_summary_heading(active_config))
        stats = summary_stats(table)
        _render_summary_stats(stats)
        _render_size_threshold_table(table)
        detect_min_nm = float((active_config.get("detection") or {}).get("min_size_nm") or 0.0)
        _render_size_bins(table, _nm_to_um(detect_min_nm))

    with st.container(border=True, key="block-results"):
        st.header("All particles")
        display = result_table(table)
        preview_rows = 100
        if len(display) > preview_rows:
            st.caption(
                f"Showing the first {preview_rows:,} of {len(display):,} rows. "
                "Download the CSV or Excel file for the full table."
            )
            st.dataframe(display.head(preview_rows), width="stretch", hide_index=True)
        else:
            st.dataframe(display, width="stretch", hide_index=True)
    

        dl_csv, dl_xlsx = st.columns(2)
        with dl_csv:
            st.download_button(
                "Download CSV",
                data=result_table(table).to_csv(index=False).encode("utf-8"),
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

    with st.container(border=True, key="block-mosaic"):
        placements = getattr(mosaic, "placements", None)
        loose_paths = unstitched_tile_paths(active_config)
        saved = None if placements else _saved_mosaic_path(active_config)
        has_mosaic = bool(placements) or saved is not None
        show_labels = True
        if not has_mosaic and not loose_paths:
            st.header("Images")
            st.info("No images were found for this run.")
        else:
            st.header("Stitched mosaic" if has_mosaic else "Images")
            show_labels = _mosaic_labels_on()
            if saved is not None:
                if show_labels:
                    review = _mosaic_with_review_labels(saved, table, active_config)
                    if review is None:
                        st.image(
                            str(saved),
                            caption="Saved mosaic with labels",
                            width="stretch",
                        )
                        download_name = saved.name
                        download_bytes = saved.read_bytes()
                        download_mime = (
                            "image/png" if saved.suffix.lower() == ".png" else "image/jpeg"
                        )
                    else:
                        st.image(
                            review,
                            caption="Saved mosaic with labels",
                            width="stretch",
                        )
                        download_name = "mosaic_overlay.jpg"
                        download_bytes = encode_overlay_jpeg(review)
                        download_mime = "image/jpeg"
                else:
                    try:
                        if not (saved.parent / "mosaic_plain.png").is_file():
                            with st.spinner("Stitching the mosaic without labels…"):
                                plain_rgb = _saved_plain_mosaic(saved, active_config)
                        else:
                            plain_rgb = _saved_plain_mosaic(saved, active_config)
                    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                        st.error("The mosaic without labels could not be built.")
                        with st.expander("Error detail"):
                            st.code(str(exc))
                        plain_rgb = None
                    if plain_rgb is None:
                        download_bytes = None
                    else:
                        st.image(
                            plain_rgb,
                            caption="Saved mosaic without labels",
                            width="stretch",
                        )
                        download_name = "mosaic_plain.jpg"
                        download_bytes = encode_overlay_jpeg(plain_rgb)
                        download_mime = "image/jpeg"
                if download_bytes is not None:
                    st.download_button(
                        "Download stitched mosaic",
                        data=download_bytes,
                        file_name=download_name,
                        mime=download_mime,
                        key="last_run_download_saved_mosaic",
                    )
        if placements:
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
                _labels_stamp(),
            )
            if st.session_state.get("overlay_key") != overlay_key:
                try:
                    plain = overlay_rgb(mosaic, active_config, crop=None)
                    st.session_state["overlay_plain"] = plain
                    review_colors = label_colors_for_hits(
                        table,
                        labels_store(ROOT).load(),
                        run_dir=str(active_config.get("input_dir") or "") or None,
                    )
                    st.session_state["overlay"] = draw_overlay_markers(
                        plain,
                        mosaic,
                        table,
                        active_config,
                        crop=None,
                        colors=review_colors,
                    )
                except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                    st.error("The mosaic was not stitched. The particle table below is still available.")
                    with st.expander("Error detail"):
                        st.code(str(exc))
                else:
                    st.session_state["overlay_key"] = overlay_key
                    out_dir = resolve_output_dir(active_config.get("output_dir"))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for name, rgb in (
                        ("mosaic_overlay.png", st.session_state["overlay"]),
                        ("mosaic_plain.png", st.session_state["overlay_plain"]),
                    ):
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        if not cv2.imwrite(str(out_dir / name), bgr):
                            st.error(f"Failed to write mosaic image to {out_dir / name}")
            labeled = st.session_state.get("overlay")
            plain = st.session_state.get("overlay_plain")
            if not show_labels and plain is None and labeled is not None:
                try:
                    with st.spinner("Stitching the mosaic without labels…"):
                        plain = overlay_rgb(mosaic, active_config, crop=None)
                    st.session_state["overlay_plain"] = plain
                    plain_path = (
                        resolve_output_dir(active_config.get("output_dir")) / "mosaic_plain.png"
                    )
                    _write_mosaic_png(plain, plain_path)
                except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                    st.error("The mosaic without labels could not be built.")
                    with st.expander("Error detail"):
                        st.code(str(exc))
            overlay = labeled if show_labels else plain
            if overlay is not None:
                caption = (
                    "All tiles, stitched at reduced resolution, with labels"
                    if show_labels
                    else "All tiles, stitched at reduced resolution, without labels"
                )
                st.image(overlay, caption=caption, width="stretch")
                st.download_button(
                    "Download stitched mosaic",
                    data=encode_overlay_jpeg(overlay),
                    file_name="mosaic_overlay.jpg" if show_labels else "mosaic_plain.jpg",
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
                    cropped = overlay_rgb(mosaic, active_config, crop=crop)
                    if show_labels:
                        cropped = draw_overlay_markers(
                            cropped,
                            mosaic,
                            table,
                            active_config,
                            crop=crop,
                            colors=label_colors_for_hits(
                                table,
                                labels_store(ROOT).load(),
                                run_dir=str(active_config.get("input_dir") or "") or None,
                            ),
                        )
                    caption = (
                        "Cropped region with labels"
                        if show_labels
                        else "Cropped region without labels"
                    )
                    st.image(cropped, caption=caption, width="stretch")
        if loose_paths:
            if has_mosaic:
                st.subheader("Separate images")
                st.caption(
                    "These filenames have no row/column pattern, so they are not stitched."
                )
            _render_unstitched_images(loose_paths, table, active_config, show_labels)
    return True


def _render_detection(config: dict) -> None:
    _apply_finished_pipeline()
    _pipeline_controls(config)


base_config = _base_config()
_prepare_ui_state(base_config)

keep_shadow_inputs()
keep_capture_inputs()
keep_stage_inputs()

_MAIN_PAGES = ["Capture", "Analyse", "Verify", "Additional tools"]


def _select_main_page(name: str) -> None:
    st.session_state["main_toolbar"] = name


if st.session_state.get("main_toolbar") not in _MAIN_PAGES:
    st.session_state["main_toolbar"] = "Capture"

with st.sidebar:
    st.subheader("Modules")
    for name in _MAIN_PAGES:
        st.button(
            name,
            key=f"main_toolbar_{name}",
            on_click=_select_main_page,
            args=(name,),
            type="primary" if st.session_state["main_toolbar"] == name else "secondary",
            width="stretch",
        )

page = st.session_state["main_toolbar"]

if page == "Analyse":
    with st.container(border=True, key="block-parameters"):
        st.header("Adjust parameters")
        _render_adjust_parameters(base_config)
    config = _config_from_state(base_config)
    with st.container(border=True, key="block-run"):
        st.header("Run/stop pipeline")
        _render_detection(config)
    run_loaded = _render_finished_run(config)
    if run_loaded:
        with st.container(border=True, key="block-review"):
            if not st.session_state.get("load_labeled_tiles"):
                st.header("Labeled tiles")
                st.button(
                    "Load labeled tiles",
                    key="run_load_labeled_tiles",
                    on_click=_request_labeled_tiles,
                )
                st.caption("Labeled tiles stay unloaded until you click Load labeled tiles.")
            else:
                render_labeled_tiles_tab(
                    config,
                    ROOT,
                    st.session_state.get("mosaic"),
                    st.session_state.get("table"),
                )
else:
    config = _config_from_state(base_config)

if page == "Verify":
    tinder_tab, pointer_tab, database_tab = st.tabs(
        ["Label", "Inspect", "Database"],
        on_change="rerun",
        key="verification_tabs",
    )
    if tinder_tab.open:
        with tinder_tab:
            with st.container(border=True, key="block-label"):
                render_label_tab(
                    config,
                    ROOT,
                    st.session_state.get("table"),
                    st.session_state.get("mosaic"),
                )
    if pointer_tab.open:
        with pointer_tab:
            with st.container(border=True, key="block-inspect"):
                render_tile_inspect_tab(
                    config,
                    ROOT,
                    st.session_state.get("table"),
                    st.session_state.get("mosaic"),
                )
    if database_tab.open:
        with database_tab:
            show_area_line = st.toggle(
                "Area line",
                value=True,
                key="database_area_line",
                help="Outline the measured particle area on the particle crops.",
            )
            st.caption(
                "Angle labels named 3of6 stay in all_labels only, until they are placed."
            )
            uniform_tab, nsew_tab, glovebox_tab = st.tabs(
                ["Uniform", "NSEW", "Glovebox"],
                on_change="rerun",
                key="database_campaign_tabs",
            )
            for campaign_tab, campaign in (
                (uniform_tab, "uniform"),
                (nsew_tab, "nsew"),
                (glovebox_tab, "glovebox"),
            ):
                if not campaign_tab.open:
                    continue
                with campaign_tab:
                    particle_col, not_particle_col = st.columns(2)
                    with particle_col:
                        with st.container(border=True, key="block-particles"):
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
                        with st.container(border=True, key="block-not-particles"):
                            render_collection_tab(
                                ROOT,
                                "not_particle",
                                "Not particles",
                                n_columns=2,
                                campaign=campaign,
                            )

if page == "Additional tools":
    training_tab, shadow_tab, redirector_tab = st.tabs(
        ["Training", "Shadow", "Redirector"],
        on_change="rerun",
        key="tools_tabs",
    )
    if training_tab.open:
        with training_tab:
            with st.container(border=True, key="block-training"):
                _render_training(config)
    if shadow_tab.open:
        with shadow_tab:
            render_shadow_tab(config)
    if redirector_tab.open:
        with redirector_tab:
            render_redirector_tab()

if page == "Capture":
    render_capture_tab()
else:
    suspend_live_view()

