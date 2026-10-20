from __future__ import annotations

import contextlib
import io
import sys
from copy import deepcopy
from pathlib import Path

import cv2
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
    next_versioned_output_name,
    output_choice_label,
    output_folder_choices,
    resolve_output_dir,
)
from src.overall.io.results_writer import excel_bytes, result_table, write_csv, write_xlsx  # noqa: E402
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
    size_distribution,
    summary_stats,
)
from src.shadow.shadow import keep_shadow_inputs, render_shadow_tab  # noqa: E402

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
    html {
        font-size: calc(16px + 8pt) !important;
    }
    h1 {
        font-size: calc(2.75 * 16px + 8pt) !important;
    }
    h2 {
        font-size: calc(2.25 * 16px + 8pt) !important;
    }
    h3 {
        font-size: calc(1.75 * 16px + 8pt) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: calc(2.25 * 16px + 8pt) !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 3.75rem !important;
    }
    [data-testid="stMain"] h1 {
        padding-top: 0.75rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Particle Detection Software")

NM_PER_UM = 1000.0
UI_SCHEMA = "mixed_layout_v24"

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
        st.session_state["ui_sidebar_frontend_seeded"] = True
    _backfill_held_widgets(base)
    _keep_filename_pattern(base)
    if not isinstance(st.session_state.get("ui_settings_nsew"), bool):
        st.session_state["ui_settings_nsew"] = False
    if st.session_state.get("ui_settings_profile") not in ("Uniform", "NSEW"):
        st.session_state["ui_settings_profile"] = (
            "NSEW" if st.session_state.get("ui_settings_nsew") else "Uniform"
        )


def _render_adjust_parameters(base: dict) -> None:
    st.radio(
        "Settings profile",
        ("Uniform", "NSEW"),
        horizontal=True,
        key="ui_settings_profile",
        on_change=_on_profile_choice,
        args=(base,),
        help="Uniform or NSEW standard settings.",
    )
    st.button(
        "Reset to standard values",
        on_click=_on_reset_standard,
        args=(base,),
        help="Restore settings for the profile selected above.",
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
            f'.st-key-sidebar_sec_{name}' for name in hidden
        )
        st.markdown(
            f"<style>{selectors}{{display:none !important;}}</style>",
            unsafe_allow_html=True,
        )

    with st.container(key="sidebar_sec_Loading"):
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
            "Folder under ASML SE/Inputs. Suggestions appear as you type."
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
        st.slider(
            "Tile overlap fraction",
            min_value=0.0,
            max_value=0.5,
            step=0.01,
            key=_live_key("ui_overlap_fraction"),
            help="The fraction of the tile that is overlapped by the next tile.",
        )
        st.number_input(
            "Pixel size (µm)",
            min_value=0.001,
            step=0.1,
            format="%.3f",
            key=_live_key("ui_pixel_size_um"),
            help="Select the physical pixel size.",
        )
        st.number_input(
            "Parallel workers",
            min_value=1,
            max_value=32,
            key=_live_key("ui_workers"),
            help="How many tiles to process at once. The minimum is 1.",
        )
        with st.expander("Filename pattern"):
            st.text_input(
                "Filename pattern (regex)",
                key=_live_key("ui_filename_pattern"),
                help="Names that do not match still run detection; they are left out of the mosaic.",
            )
    with st.container(key="sidebar_sec_Preprocessing"):
        _restore_widgets(_SIDEBAR_SECTIONS["Preprocessing"], force=seed_frontend)
        st.checkbox(
            "Denoise",
            key=_live_key("ui_denoise"),
            help="Small blur so sensor speckle is not detected as particles.",
        )
        st.number_input(
            "Denoise sigma",
            min_value=0.0,
            key=_live_key("ui_denoise_sigma"),
            help="Blur radius in pixels; larger values also round off real particles.",
        )
        st.checkbox(
            "Flatten illumination",
            key=_live_key("ui_flatten_illumination"),
            help="Divide each tile by a wide blur of itself to remove vignetting.",
        )
        st.number_input(
            "Flatten sigma",
            min_value=1.0,
            key=_live_key("ui_flatten_sigma"),
            help="Large values flatten only illumination; small values halo region edges.",
        )
        st.checkbox(
            "Contrast stretch",
            key=_live_key("ui_contrast_stretch"),
            help="Map each tile's 1st–99th percentiles onto 0–1 so contrast matches before detection.",
        )
    with st.container(key="sidebar_sec_Detection"):
        _restore_widgets(_SIDEBAR_SECTIONS["Detection"], force=seed_frontend)
        st.selectbox(
            "Background suppression",
            options=["fft", "tophat"],
            key=_live_key("ui_method"),
            help="fft notches the repeating lattice and falls back to top-hat when a tile has no peaks.",
        )
        st.checkbox(
            "Particles are bright",
            key=_live_key("ui_particles_bright"),
            help="Off inverts the residual so dark debris is found the same way.",
        )
        st.slider(
            "FFT peak threshold",
            0.05,
            0.95,
            key=_live_key("ui_fft_peak_threshold"),
            help="Lower notches more of the lattice and can also remove real structure.",
        )
        st.selectbox(
            "FFT notch source",
            options=["per_tile", "shared"],
            key=_live_key("ui_fft_mask"),
            help="per_tile notches each tile from its own spectrum; shared reuses the first tile.",
        )
        st.number_input(
            "Top-hat radius (px)",
            min_value=1,
            key=_live_key("ui_tophat_radius"),
            help="Must be larger than the biggest particle, or large debris is erased.",
        )
        st.number_input(
            "Min size (µm)",
            min_value=0.1,
            step=0.5,
            format="%.2f",
            key=_live_key("ui_min_size_um"),
            help="Keep particles larger than this.",
        )
        st.number_input(
            "Max size (µm)",
            min_value=0.1,
            step=0.5,
            format="%.2f",
            key=_live_key("ui_max_size_um"),
            help="Drop particles larger than this.",
        )
        st.slider(
            "Blob threshold",
            0.001,
            0.5,
            key=_live_key("ui_blob_threshold"),
            help="DoG peak height after 0–1 scaling; lower keeps dimmer candidates.",
        )
        st.slider(
            "Min prominence",
            min_value=0.0,
            max_value=0.8,
            key=_live_key("ui_min_prominence"),
            help="Zero residual weaker than this before detection; 0 disables.",
        )
        st.slider(
            "Confidence score (%)",
            min_value=0,
            max_value=100,
            step=1,
            key=_live_key("ui_min_confidence_pct"),
            help="Drop blobs whose residual at the peak is below this; 0 disables.",
        )
        st.number_input(
            "Blob min sigma",
            min_value=0.5,
            key=_live_key("ui_blob_min_sigma"),
            help="Smallest scale searched; diameter is about 2.8 times sigma.",
        )
        st.number_input(
            "Blob max sigma",
            min_value=1.0,
            key=_live_key("ui_blob_max_sigma"),
            help="Largest scale searched; raise it with Max size so large flakes are found.",
        )
        st.number_input(
            "Local SNR sigma (0 = auto)",
            min_value=0.0,
            key=_live_key("ui_local_snr_sigma"),
            help="Envelope for local energy; 0 turns this on for FFT residuals and off for top-hat.",
        )
        st.number_input(
            "Edge soften sigma",
            min_value=0.0,
            key=_live_key("ui_edge_soften_sigma"),
            help="Coarse blur before the region-border gradient; 0 disables.",
        )
        st.number_input(
            "Edge soften strength",
            min_value=0.0,
            key=_live_key("ui_edge_soften_strength"),
            help="How hard to attenuate residual on area borders; 0 disables.",
        )
        st.number_input(
            "Edge exclude (px)",
            min_value=0.0,
            key=_live_key("ui_edge_exclude_px"),
            help="Ignore blobs this close to a region border, including pad corners.",
        )
        st.slider(
            "Min circularity",
            min_value=0.0,
            max_value=1.0,
            key=_live_key("ui_min_circularity"),
            help="Drop elongated hits; 0 keeps every blob.",
        )
        st.number_input(
            "Structure neighbor radius (px)",
            min_value=0.0,
            key=_live_key("ui_structure_neighbor_px"),
            help="Drop blobs that have two or more neighbours this close; 0 disables.",
        )
        st.checkbox(
            "Apply ML filter",
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
                key=output_key,
                help=output_help,
                **output_kwargs,
            )
        else:
            st.text_input(
                "Output folder",
                key=output_key,
                placeholder="Start typing a folder name",
                help=output_help,
            )
        st.number_input(
            "Merge radius (px)",
            min_value=0.0,
            key=_live_key("ui_merge_radius_px"),
            help="Hits closer than this on overlapping tiles count as one particle; 0 skips the merge.",
        )
        st.selectbox(
            "Cluster size aggregation",
            options=["max", "mean"],
            key=_live_key("ui_size_aggregation"),
            help="max keeps the largest diameter in a merged cluster; mean averages them.",
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
            help="0 picks a factor from the target size; a value above 0 forces that factor.",
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
            help="Chooses the Fβ score and the ML model and starting threshold for the next run.",
        )
    st.session_state["ui_sidebar_frontend_seeded"] = True


def _config_from_state(base: dict) -> dict:
    config = deepcopy(base)
    config["input_dir"] = _widget_value("ui_input_dir")
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
    st.session_state.pop("overlay_key", None)
    out_dir = resolve_output_dir(config["output_dir"])
    csv_path = out_dir / "particles.csv"
    xlsx_path = out_dir / "particles.xlsx"
    write_csv(table, csv_path)
    write_xlsx(table, xlsx_path, config)
    for report_name in ("last_pipeline.json", "last_run.pdf", "mosaic_overlay.png"):
        report_path = out_dir / report_name
        if report_path.is_file():
            report_path.touch()
    write_last_pipeline(config, csv_path)
    st.session_state["last_pipeline_csv"] = str(csv_path)
    st.session_state["ui_show_last_run"] = True
    stats = summary_stats(table)
    st.success(
        f"{stats['particle_count']} particles. Mean size {_nm_to_um(stats['size_mean']):.2f} µm."
    )
    st.caption(f"Saved to {out_dir}.")


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
                st.rerun(scope="app")
        if st.button(
            "Stop pipeline",
            disabled=not running,
            key="ui_stop_pipeline",
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


def _render_size_bins(table: pd.DataFrame, min_size_um: float) -> None:
    st.subheader("Size distribution (µm)")
    if table.empty or "size" not in table.columns:
        st.info("This run has no particle sizes to bin.")
        return
    st.bar_chart(
        size_distribution(table, min_size_um),
        x_label="Size (µm)",
        y_label="Amount of particles",
        sort=False,
    )


def _render_finished_run(config: dict) -> None:
    """Table, size bins, and mosaic from the latest finished pipeline run."""
    table, mosaic, active_config = _finished_run_state(config)
    if table is None:
        with st.container(border=True, key="block-last-run"):
            st.subheader("Last run")
            st.info("Run the pipeline once. Last run keeps that table, size bins, and mosaic.")
        return

    with st.container(border=True, key="block-last-run"):
        st.subheader("Last run")
        stats = summary_stats(table)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Particles", stats["particle_count"])
        c2.metric("Mean size (µm)", f"{_nm_to_um(stats['size_mean']):.2f}")
        c3.metric("Min size (µm)", f"{_nm_to_um(stats['size_min']):.2f}")
        c4.metric("Max size (µm)", f"{_nm_to_um(stats['size_max']):.2f}")
        _render_size_threshold_table(table)

    with st.container(border=True, key="block-mosaic"):
        placements = getattr(mosaic, "placements", None)
        if not placements:
            saved = _saved_mosaic_path(active_config)
            st.subheader("Stitched mosaic")
            if saved is not None:
                st.image(str(saved), caption="Saved mosaic from the last run", width="stretch")
                st.download_button(
                    "Download stitched mosaic",
                    data=saved.read_bytes(),
                    file_name=saved.name,
                    mime="image/png" if saved.suffix.lower() == ".png" else "image/jpeg",
                    key="last_run_download_saved_mosaic",
                )
            else:
                st.info(
                    "No tiles had row/column (or x/y) in the filename, so nothing was "
                    "stitched. The table is below. Open Verification, then Inspect, for each image."
                )
        else:
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
                    st.error("The mosaic was not stitched. The particle table below is still available.")
                    with st.expander("Error detail"):
                        st.code(str(exc))
                else:
                    st.session_state["overlay_key"] = overlay_key
                    mosaic_path = resolve_output_dir(active_config.get("output_dir")) / "mosaic_overlay.png"
                    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
                    bgr = cv2.cvtColor(st.session_state["overlay"], cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(mosaic_path), bgr):
                        st.error(f"Failed to write mosaic image to {mosaic_path}")
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

    with st.container(border=True, key="block-results"):
        st.subheader("Results")
        display = result_table(table)
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
            "`x_global_um` and `y_global_um` are microns from the top-left of the mosaic "
            "(x right, y down). `size_um` is equivalent circular diameter in µm."
        )

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

    min_size_nm = float((active_config.get("detection") or {}).get("min_size_nm") or 0.0)
    with st.container(border=True, key="block-size"):
        _render_size_bins(table, _nm_to_um(min_size_nm))


def _render_detection(config: dict) -> None:
    _apply_finished_pipeline()
    _pipeline_controls(config)


base_config = _base_config()
_prepare_ui_state(base_config)

keep_shadow_inputs()
keep_capture_inputs()
keep_stage_inputs()

_MAIN_PAGES = ["Capture", "Run", "Verify", "Additional tools"]


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

if page == "Run":
    with st.container(border=True, key="block-parameters"):
        st.subheader("Adjust parameters")
        _render_adjust_parameters(base_config)
    config = _config_from_state(base_config)
    with st.container(border=True, key="block-run"):
        st.subheader("Run/stop pipeline")
        _render_detection(config)
    _render_finished_run(config)
    with st.container(border=True, key="block-review"):
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

