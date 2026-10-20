"""Streamlit tabs for labeling crops and browsing collections."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

def _failure(action: str, next_step: str, exc: BaseException) -> None:
    detail = str(exc).strip() or type(exc).__name__
    first = detail.splitlines()[0]
    if len(first) > 160:
        st.error(f"{action} {next_step}")
    else:
        st.error(f"{action} {first}")
    with st.expander("Error detail"):
        st.code(detail)


from src.overall.config import cfg_get, resolve_output_dir
from src.overall.io.tile_loader import DEFAULT_FILENAME_PATTERN, matching_tile_paths
from src.verification.crops import (
    TileImageCache,
    brightest_direction_crop,
    brightest_prominence_direction_row,
    combi_or_uniform_crop,
    compass_direction,
    config_for_hit_placement,
    crop_particle,
    nsew_direction_crops,
    find_tile_path,
    local_px_to_global_nm,
    placement_for_tile,
    ParticleCrop,
)
from src.verification.audit_misses import input_tile_names
from src.verification.inspect import (
    CELL_NAMES,
    DEFAULT_MISS_SIZE_UM,
    GRID_N,
    INSPECT_DISPLAY_WIDTH,
    LABEL_CIRCLE_COLORS,
    MISSED_PARTICLE_COLOR,
    OVERVIEW_MAX_SIDE,
    ZOOM_MAX_SIDE,
    candidate_placements,
    cell_bounds,
    circles_from_rows,
    detections_on_review_picture,
    display_xy_to_local,
    expand_direction_tile_names,
    expand_submap_lighting_names,
    label_color_for_key,
    labeled_particle_recovery,
    label_on_run_tiles,
    crop_metric_cells,
    f_beta_score,
    std_of_crop_cells,
    last_run_class_counts,
    last_run_class_counts_by_tile,
    unhappiness_score,
    local_xy_on_tile,
    missed_particle_record,
    footprint_grow_mass,
    outline_particle_crop_file,
    photo_particle_edge,
    verify_missed_photo_sizes,
    overlay_view,
    pointer_miss_rows,
    pointer_direction_tiles,
    pointer_set_hits,
    preview_click_crop,
    related_scene_tile_names,
    tile_names_for_hits,
    undetected_size_floor_nm,
)
from src.verification.queue import (
    clear_label_queue,
    expand_labeled_keys,
    label_queue_is_cleared,
    last_run_csv,
    load_last_pipeline,
    restore_label_queue,
    use_last_pipeline_queue,
    load_particles_csv,
    label_map_with_aliases,
    merge_detection_tables,
    tag_table,
    nsew_key_aliases,
    share_exposure_keys,
    unlabeled_queue,
    collapse_submap_review_detections,
    snap_detection_keys_to_labels,
)
from src.main_pipeline.measurement.measurer import (
    DEFAULT_NSEW_MERGE_RADIUS_PX,
    DEFAULT_NSEW_SIZE_MATCH_FRACTION,
    is_combi_tile,
    is_nsew_family_tile,
    is_submap_lighting_tile,
    lighting_role,
    lighting_source_name,
    lighting_tile_id,
    submap_id,
)
from src.verification.store import LabelStore, crop_source_folder, rows_for_campaign
from src.main_pipeline.report.report_generator import particles_at_least
from src.main_pipeline.stitching.stitcher import LazyMosaic

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # pragma: no cover - UI-only optional
    streamlit_image_coordinates = None

NM_PER_UM = 1000.0
GALLERY_PAGE_SIZE = 24
LABEL_IMAGE_WIDTH = 420
GALLERY_IMAGE_WIDTH = 180
MARK_COLUMNS = 6
MARK_PAGE_SIZE = 48
LAST_RUN_OVERVIEW_WIDTH = 720
LAST_RUN_CROP_COLUMNS = 4


def labels_store(project_root: str | Path) -> LabelStore:
    return LabelStore(Path(project_root) / "labels")


def _nsew_merge_radius_nm(config: dict[str, Any]) -> float:
    pixel_size = float(cfg_get(config, "pixel_size_nm", 960.0))
    merge_px = float(
        cfg_get(
            config,
            "measurement.nsew_merge_radius_px",
            DEFAULT_NSEW_MERGE_RADIUS_PX,
        )
    )
    return merge_px * pixel_size


def _nsew_size_match_fraction(config: dict[str, Any]) -> float:
    return float(
        cfg_get(
            config,
            "measurement.nsew_size_match_fraction",
            DEFAULT_NSEW_SIZE_MATCH_FRACTION,
        )
    )


def _labels_for_scene(
    labels_df: pd.DataFrame,
    config: dict[str, Any],
    extra_tiles: list[str] | None = None,
) -> pd.DataFrame:
    if labels_df is None or labels_df.empty:
        return labels_df
    scene = related_scene_tile_names(
        cfg_get(config, "input_dir", None), extra_tiles
    )
    if not scene:
        return labels_df
    tile = labels_df["source_tile"].astype(str).map(
        lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1]
    )
    kept = labels_df.loc[tile.isin(scene)]
    # A Tinder label is often stored on a quad. Combi and uniform still show
    # that hit, so the label has to come along.
    review_sets = {
        submap_id(name)
        for name in (extra_tiles or ())
        if submap_id(name) and (is_combi_tile(name) or is_submap_lighting_tile(name))
    }
    if review_sets and "source_tile" in labels_df.columns:
        same_set = labels_df["source_tile"].astype(str).map(
            lambda name: submap_id(name) in review_sets
            and (is_submap_lighting_tile(name) or is_combi_tile(name))
        )
        kept = pd.concat([kept, labels_df.loc[same_set]], ignore_index=True)
        if "key" in kept.columns:
            kept = kept.drop_duplicates(subset=["key"], keep="last")
    return kept.reset_index(drop=True)


def _collapse_review_hits(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    config: dict[str, Any],
) -> pd.DataFrame | None:
    """One id for a particle found on both pictures, still wearing its label.

    Snapping first puts the saved label key on whichever picture was marked.
    The collapse keeps that key, then a second snap catches a label that sits
    a little off the kept coordinate.
    """
    if detections is None:
        return None
    radius = _nsew_merge_radius_nm(config)
    fraction = _nsew_size_match_fraction(config)
    snapped = snap_detection_keys_to_labels(detections, labels, radius, fraction)
    keys: set[str] = set()
    if labels is not None and not labels.empty and "key" in labels.columns:
        keys.update(str(key) for key in labels["key"].astype(str))
    collapsed = collapse_submap_review_detections(
        snapped, radius, fraction, labeled_keys=keys
    )
    return snap_detection_keys_to_labels(collapsed, labels, radius, fraction)


def _share_exposure_hits(frame: pd.DataFrame | None, config: dict[str, Any]) -> pd.DataFrame | None:
    """Nearby v3 and v4 hits in an exposure pair share one label key."""
    if frame is None:
        return None
    return share_exposure_keys(
        frame,
        str(cfg_get(config, "input_dir", "") or ""),
        _nsew_merge_radius_nm(config),
        _nsew_size_match_fraction(config),
    )


def _queue_skip_keys(store, scene_labels: pd.DataFrame, config: dict[str, Any]):
    """Keys already labeled in this scene.

    A resolved scene with no saved labels yet skips nothing. That keeps a new
    ``v3_2of4.png`` run separate from older ``v3_2of4.bmp`` marks.
    """
    if scene_labels is not None and not scene_labels.empty and "key" in scene_labels.columns:
        return scene_labels["key"]
    scene = related_scene_tile_names(cfg_get(config, "input_dir", None))
    if scene:
        return []
    return store.labeled_keys()


def render_label_tab(
    config: dict[str, Any],
    project_root: str | Path,
    table: pd.DataFrame | None,
    mosaic: LazyMosaic | None,
) -> None:
    """One unlabeled crop at a time, labeled with arrow-key buttons."""
    store = labels_store(project_root)
    cache = st.session_state.setdefault("label_tile_cache", TileImageCache())
    origin_cache = st.session_state.setdefault("label_origin_cache", {})

    st.header("Label detections")
    st.caption("One crop at a time. The key for each action is on its button.")

    pipeline_csv, pipeline_config = load_last_pipeline(config)
    if not st.session_state.get("load_last_run"):
        st.button(
            "Load last run",
            key="label_load_last_run",
            disabled=pipeline_csv is None or not Path(pipeline_csv).is_file(),
            on_click=request_last_run,
            help=(
                f"Queue {pipeline_csv.parent.name}/particles.csv"
                if pipeline_csv is not None and Path(pipeline_csv).is_file()
                else "Run the pipeline first."
            ),
        )
        st.caption("Last run stays unloaded until you click Load last run.")
        return
    use_last_pipeline = use_last_pipeline_queue(
        st.session_state.get("label_source"), pipeline_csv
    )
    if use_last_pipeline:
        last_csv = pipeline_csv
        crop_config = pipeline_config
        tables = _last_pipeline_tables(pipeline_csv)
    else:
        last_csv = last_run_csv(project_root, config)
        crop_config = config
        tables = _detection_tables(project_root, table, config)
    tables = [
        shared
        for frame in tables
        if (shared := _share_exposure_hits(frame, crop_config)) is not None
    ]
    n_hits = sum(len(frame) for frame in tables)
    radius_nm = _nsew_merge_radius_nm(crop_config)
    size_frac = _nsew_size_match_fraction(crop_config)
    scene_labels = _labels_for_scene(store.load(), crop_config)
    skip_keys = _queue_skip_keys(store, scene_labels, crop_config)
    queue = unlabeled_queue(
        tables,
        skip_keys,
        nsew_merge_radius_nm=radius_nm,
        nsew_size_match_fraction=size_frac,
        labels=scene_labels,
    )
    placed = config_for_hit_placement(crop_config, queue)
    if float(placed.get("overlap_fraction", 0.0) or 0.0) != float(
        crop_config.get("overlap_fraction", 0.0) or 0.0
    ):
        crop_config = placed
    if label_queue_is_cleared(last_csv):
        queue = queue.iloc[0:0]

    front = st.session_state.get("label_front_key")
    if front is not None and not queue.empty:
        match = queue["key"].astype(str) == str(front)
        if match.any():
            queue = pd.concat([queue.loc[match], queue.loc[~match]], ignore_index=True)

    counts = store.counts()
    labeled_n = sum(counts.values())
    remaining = len(queue)
    load_col, empty_col = st.columns(2)
    with load_col:
        st.button(
            "Load last run",
            key="label_load_last_run",
            disabled=pipeline_csv is None or not Path(pipeline_csv).is_file(),
            on_click=_on_load_last_pipeline,
            help=(
                f"Queue {pipeline_csv.parent.name}/particles.csv"
                if pipeline_csv is not None and Path(pipeline_csv).is_file()
                else "Run Detection first so last_pipeline.json points at an output folder."
            ),
        )
    with empty_col:
        st.button(
            "Empty queue",
            key="label_empty_queue",
            disabled=last_csv is None or not Path(last_csv).is_file() or remaining == 0,
            on_click=_on_empty_label_queue,
            args=(str(last_csv),) if last_csv is not None else (),
            help="Drop the remaining unlabeled hits. Saved labels stay. Load last run puts this queue back.",
        )
    source_label = _source_caption(last_csv, last_pipeline=use_last_pipeline)
    st.caption(
        f"{remaining} remaining · labeled {labeled_n} "
        f"(particles {counts['particle']} · not sure {counts['not_sure']} · "
        f"not particles {counts['not_particle']}) · source {source_label}"
    )
    if st.session_state.pop("label_source_notice", False):
        folder = last_csv.parent.name if last_csv is not None else "none"
        st.info(
            f"Loaded **{n_hits}** detections from **{folder}**. "
            f"**{remaining}** unlabeled at 10 µm and above "
            f"(already-labeled keys are skipped)."
        )

    mode = st.radio(
        "Mode",
        ["One at a time", "All hits"],
        horizontal=True,
        key="label_review_mode",
    )

    marked_rejects = {
        str(key) for key in st.session_state.get("all_hits_reject_keys", [])
    }
    if mode == "All hits" and marked_rejects:
        labeled_ids = skip_keys
        queue = _queue_with_marked_particles(
            tables,
            labeled_ids,
            marked_rejects,
            radius_nm,
            size_frac,
            scene_labels,
        )
        if label_queue_is_cleared(last_csv):
            queue = queue.iloc[0:0]

    if queue.empty:
        st.success("Nothing left to label.")
        _undo_row(store, counts)
        return

    if mode == "All hits":
        _render_all_hits(
            store,
            queue,
            tables,
            crop_config,
            marked_rejects,
            radius_nm,
            size_frac,
            mosaic=mosaic,
            cache=cache,
            origin_cache=origin_cache,
        )
        return

    current = queue.iloc[0]
    candidates = _detection_candidates(tables)
    try:
        review = combi_or_uniform_crop(
            current,
            crop_config,
            candidates,
            merge_radius_nm=radius_nm,
            size_match_fraction=size_frac,
            mosaic=mosaic,
            cache=cache,
            origin_cache=origin_cache,
        )
        views = {}
        if review is not None:
            crop, _review_prominence = review
        else:
            views = nsew_direction_crops(
                current,
                crop_config,
                mosaic=mosaic,
                cache=cache,
                origin_cache=origin_cache,
            )
            if views:
                crop = next(iter(views.values()))
            else:
                crop = crop_particle(
                    current,
                    crop_config,
                    mosaic=mosaic,
                    cache=cache,
                    origin_cache=origin_cache,
                )
    except (FileNotFoundError, ValueError, OSError) as exc:
        _failure("This crop could not be opened.", "Load the last run again, or pick another tile.", exc)
        _undo_row(store, counts)
        return

    size_um = float(current["size"]) / NM_PER_UM
    if review is not None:
        image_col, _ = st.columns([LABEL_IMAGE_WIDTH, 800])
        image_col.image(
            crop.rgb,
            caption=(
                f"{crop.direction} · id {current.get('id', '')} · "
                f"{size_um:.2f} µm · prominence {_review_prominence:.3f}"
            ),
            width=LABEL_IMAGE_WIDTH,
        )
    elif views:
        cols = st.columns(len(views))
        for column, (direction, view) in zip(cols, views.items()):
            column.image(
                view.rgb,
                caption=f"{direction} · {view.tile_path.name}",
                width="stretch",
            )
        dirs = ", ".join(views)
        st.caption(
            f"id {current.get('id', '')} · {size_um:.2f} µm · "
            f"confidence {float(current.get('confidence', 0.0)):.3f} · "
            f"same location on {dirs} (one label for all)"
        )
    else:
        image_col, _ = st.columns([LABEL_IMAGE_WIDTH, 800])
        where = f"{crop.direction} · " if crop.direction else ""
        shared_with = str(current.get("exposure_with") or "")
        shared_note = ""
        if shared_with and shared_with.lower() != "nan":
            this_set = Path(str(current["source_tile"])).stem.removesuffix("_2of4")
            shared_note = f" · same particle on {this_set} and {shared_with}"
        image_col.image(
            crop.rgb,
            caption=(
                f"{where}id {current.get('id', '')} · {current['source_tile']} · "
                f"{size_um:.2f} µm · confidence {float(current.get('confidence', 0.0)):.3f}"
                f"{shared_note}"
            ),
            width=LABEL_IMAGE_WIDTH,
        )
    st.caption(f"This queue: 1 of {remaining} still unlabeled.")

    cols = st.columns(3)
    is_particle = cols[0].button(
        "Particle", shortcut="Right", type="primary", width="stretch", key="label_right"
    )
    not_sure = cols[1].button(
        "Not sure", shortcut="Down", width="stretch", key="label_down"
    )
    not_particle = cols[2].button(
        "Not a particle", shortcut="Left", width="stretch", key="label_left"
    )
    undo = st.button(
        "Undo",
        shortcut="Up",
        key="label_undo",
        disabled=labeled_n == 0,
    )

    if is_particle:
        store.apply_label(current, "particle", crop.rgb)
        st.session_state.pop("label_front_key", None)
        st.rerun()
    if not_sure:
        store.apply_label(current, "not_sure", crop.rgb)
        st.session_state.pop("label_front_key", None)
        st.rerun()
    if not_particle:
        store.apply_label(current, "not_particle", crop.rgb)
        st.session_state.pop("label_front_key", None)
        st.rerun()
    if undo:
        restored = store.undo_last()
        if restored is not None:
            st.session_state["label_front_key"] = str(restored["key"])
        st.rerun()


def _queue_with_marked_particles(
    tables: list[pd.DataFrame],
    labeled_keys,
    marked_particles: set[str],
    radius_nm: float,
    size_frac: float,
    labels: pd.DataFrame | None,
) -> pd.DataFrame:
    """Unlabeled hits, plus ones already pressed as not a particle."""
    skip = expand_labeled_keys(str(key) for key in labeled_keys)
    for key in marked_particles:
        skip.difference_update(nsew_key_aliases(key))
    return unlabeled_queue(
        tables,
        skip,
        nsew_merge_radius_nm=radius_nm,
        nsew_size_match_fraction=size_frac,
        labels=labels,
    )


def _marked_reject_button_css(keys: set[str]) -> None:
    rules = []
    for key in sorted(keys):
        rules.append(
            f".st-key-mark_not_particle_{key} button"
            "{background-color:#8f2d2d;border-color:#8f2d2d;color:#ffffff;}"
        )
    if not rules:
        return
    st.markdown("<style>\n" + "\n".join(rules) + "\n</style>", unsafe_allow_html=True)


def _detection_candidates(detections: list[pd.DataFrame]) -> pd.DataFrame | None:
    frames = [
        frame
        for frame in detections
        if frame is not None and not frame.empty and "source_tile" in frame.columns
    ]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _all_hits_crop(
    record: dict[str, Any],
    candidates: pd.DataFrame | None,
    crop_config: dict[str, Any],
    *,
    merge_radius_nm: float,
    size_match_fraction: float,
    mosaic: LazyMosaic | None,
    cache: TileImageCache,
    origin_cache: dict,
) -> tuple[ParticleCrop, float]:
    """Combi or uniform when this is a Groundup hit, else the brightest N/S/E/W."""
    review = combi_or_uniform_crop(
        record,
        crop_config,
        candidates,
        merge_radius_nm=merge_radius_nm,
        size_match_fraction=size_match_fraction,
        mosaic=mosaic,
        cache=cache,
        origin_cache=origin_cache,
    )
    if review is not None:
        return review
    best = brightest_prominence_direction_row(
        record,
        candidates,
        merge_radius_nm=merge_radius_nm,
        size_match_fraction=size_match_fraction,
    )
    if best is not None:
        shown = dict(record)
        shown["source_tile"] = best.get("source_tile", shown.get("source_tile"))
        shown["x_global"] = best.get("x_global", shown.get("x_global"))
        shown["y_global"] = best.get("y_global", shown.get("y_global"))
        if float(best.get("size") or 0.0) > 0:
            shown["size"] = best["size"]
        if best.get("confidence") not in (None, ""):
            shown["confidence"] = best["confidence"]
        crop = crop_particle(
            shown,
            crop_config,
            mosaic=mosaic,
            cache=cache,
            origin_cache=origin_cache,
        )
        direction = compass_direction(str(shown.get("source_tile") or "")) or ""
        prominence = float(shown.get("confidence") or 0.0)
        return (
            ParticleCrop(
                rgb=crop.rgb,
                x_local=crop.x_local,
                y_local=crop.y_local,
                crop_x0=crop.crop_x0,
                crop_y0=crop.crop_y0,
                tile_path=crop.tile_path,
                direction=direction,
            ),
            prominence,
        )
    crop = _tinder_crop(
        record,
        crop_config,
        mosaic=mosaic,
        cache=cache,
        origin_cache=origin_cache,
    )
    return crop, float(record.get("confidence") or 0.0)


def _tinder_crop(
    record: dict[str, Any],
    crop_config: dict[str, Any],
    *,
    mosaic: LazyMosaic | None,
    cache: TileImageCache,
    origin_cache: dict,
):
    """Directional crop when the hit is on a combined image, otherwise the source tile."""
    bright = brightest_direction_crop(
        record,
        crop_config,
        mosaic=mosaic,
        cache=cache,
        origin_cache=origin_cache,
    )
    if bright is not None:
        return bright
    return crop_particle(
        record,
        crop_config,
        mosaic=mosaic,
        cache=cache,
        origin_cache=origin_cache,
    )


def _render_all_hits(
    store: LabelStore,
    queue: pd.DataFrame,
    detections: list[pd.DataFrame],
    crop_config: dict[str, Any],
    marked_rejects: set[str],
    merge_radius_nm: float,
    size_match_fraction: float,
    *,
    mosaic: LazyMosaic | None,
    cache: TileImageCache,
    origin_cache: dict,
) -> None:
    """Show unlabeled hits in a grid. Not a particle marks one; All done saves the page."""
    pressed = queue["key"].astype(str).isin(marked_rejects)
    open_queue = queue.loc[~pressed]
    rejected = queue.loc[pressed]
    remaining = len(open_queue)
    done = st.button(
        "All done",
        type="primary",
        key="label_all_done",
        help=(
            f"Save {int(pressed.sum())} marked hits as not a particle and "
            f"label the other {remaining} as particles, including other pages."
        ),
    )
    n_pages = max(1, (len(queue) + MARK_PAGE_SIZE - 1) // MARK_PAGE_SIZE)
    stored_page = int(st.session_state.get("label_all_hits_page", 1) or 1)
    if stored_page < 1 or stored_page > n_pages:
        st.session_state["label_all_hits_page"] = min(max(stored_page, 1), n_pages)
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=n_pages,
            key="label_all_hits_page",
        )
    )
    st.caption(
        f"{len(queue)} hits · {int(pressed.sum())} marked not a particle · "
        f"page {page} of {n_pages}. "
        "Each Groundup picture is the combi or the uniform image with the highest prominence. "
        "A letter N/S/E/W hit stays on its brightest direction. "
        "Press Not a particle on the false ones. That only turns the button red. "
        "All done saves the red ones as not a particle and labels every other hit "
        "as a particle, including hits on other pages."
    )
    candidates = _detection_candidates(detections)
    _marked_reject_button_css(marked_rejects)
    if done:
        with st.spinner(
            f"Saving {len(rejected)} not-particles and {remaining} particles…"
        ):
            items, failed = _queued_label_items(
                rejected,
                "not_particle",
                candidates,
                crop_config,
                merge_radius_nm,
                size_match_fraction,
                mosaic,
                cache,
                origin_cache,
            )
            kept, kept_failed = _queued_label_items(
                open_queue,
                "particle",
                candidates,
                crop_config,
                merge_radius_nm,
                size_match_fraction,
                mosaic,
                cache,
                origin_cache,
            )
        items.extend(kept)
        failed.extend(kept_failed)
        if items:
            store.apply_labels(items)
            st.session_state.pop("label_front_key", None)
        if failed:
            st.error("Left unlabeled: " + "; ".join(failed[:8]))
            return
        st.session_state["all_hits_reject_keys"] = []
        st.rerun()

    start = (page - 1) * MARK_PAGE_SIZE
    page_rows = queue.iloc[start : start + MARK_PAGE_SIZE]
    records = page_rows.to_dict(orient="records")
    st.session_state["_all_hits_ctx"] = {
        "records": {
            str(record.get("key") or index): record
            for index, record in enumerate(records)
        },
        "candidates": candidates,
        "crop_config": crop_config,
        "merge_radius_nm": merge_radius_nm,
        "size_match_fraction": size_match_fraction,
        "mosaic": mosaic,
    }
    st.markdown(
        """
        <style>
        [class*="st-key-all_hits_card_"] [data-testid="stFullScreenFrame"] > div,
        [class*="st-key-all_hits_card_"] [data-testid="stImage"],
        [class*="st-key-all_hits_card_"] [data-testid="stImageContainer"] {
            width: 100% !important;
        }
        [class*="st-key-all_hits_card_"] [data-testid="stImage"] img {
            width: 100% !important;
            height: auto !important;
            aspect-ratio: 1 / 1;
            object-fit: contain;
        }
        [class*="st-key-all_hits_card_"] .all-hits-caption {
            min-height: 2.6em;
            margin: 0.2rem 0 0.35rem;
            font-size: 0.8rem;
            font-weight: 400;
            line-height: 1.3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(records), MARK_COLUMNS):
        row = records[row_start : row_start + MARK_COLUMNS]
        columns = st.columns(MARK_COLUMNS, vertical_alignment="top", wrap=False)
        for offset, record in enumerate(row):
            key = str(record.get("key") or (row_start + offset))
            with columns[offset]:
                _render_hit_card(key)


def _on_mark_not_particle(key: str) -> None:
    """Remember a reject until All done. Runs before the card redraws."""
    marked_now = [str(item) for item in (st.session_state.get("all_hits_reject_keys") or [])]
    if key not in marked_now:
        marked_now.append(key)
    st.session_state["all_hits_reject_keys"] = marked_now


@st.fragment
def _render_hit_card(key: str) -> None:
    """One grid cell. A click redraws this card only, and does not save yet."""
    ctx = st.session_state.get("_all_hits_ctx") or {}
    record = (ctx.get("records") or {}).get(key)
    if record is None:
        st.caption(f"Missing hit {key}")
        return
    cache = st.session_state.setdefault("label_tile_cache", TileImageCache())
    origin_cache = st.session_state.setdefault("label_origin_cache", {})
    try:
        crop, prominence = _all_hits_crop(
            record,
            ctx.get("candidates"),
            ctx.get("crop_config") or {},
            merge_radius_nm=float(ctx.get("merge_radius_nm") or 0.0),
            size_match_fraction=float(ctx.get("size_match_fraction") or 0.0),
            mosaic=ctx.get("mosaic"),
            cache=cache,
            origin_cache=origin_cache,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        _failure("This crop could not be opened.", "Load the last run again, or pick another tile.", exc)
        return
    size_um = float(record["size"]) / NM_PER_UM
    where = f"{crop.direction} · " if crop.direction else ""
    caption = (
        f"{where}id {record.get('id', '')} · {size_um:.1f} µm · {prominence:.2f}"
    )
    with st.container(key=f"all_hits_card_{key}"):
        st.image(_square_rgb(crop.rgb), width="stretch")
        st.markdown(
            f'<p class="all-hits-caption">{caption}</p>',
            unsafe_allow_html=True,
        )
        already = key in {
            str(item) for item in (st.session_state.get("all_hits_reject_keys") or [])
        }
        if already:
            st.markdown(
                (
                    f"<style>.st-key-mark_not_particle_{key} button"
                    "{background-color:#8f2d2d;border-color:#8f2d2d;color:#ffffff;}</style>"
                ),
                unsafe_allow_html=True,
            )
        st.button(
            "Not a particle",
            key=f"mark_not_particle_{key}",
            type="secondary" if already else "primary",
            width="stretch",
            on_click=_on_mark_not_particle,
            args=(key,),
        )


def _square_rgb(rgb: np.ndarray) -> np.ndarray:
    """Pad a crop to a square so every All hits cell is the same shape."""
    image = np.asarray(rgb)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    height, width = int(image.shape[0]), int(image.shape[1])
    side = max(height, width, 1)
    if height == side and width == side:
        return image
    canvas = np.zeros((side, side, int(image.shape[2])), dtype=image.dtype)
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = image
    return canvas


def _queued_label_items(
    queue: pd.DataFrame,
    label: str,
    candidates: pd.DataFrame | None,
    crop_config: dict[str, Any],
    merge_radius_nm: float,
    size_match_fraction: float,
    mosaic: LazyMosaic | None,
    cache: TileImageCache,
    origin_cache: dict,
) -> tuple[list[tuple[dict[str, Any], str, Any]], list[str]]:
    """Crops for one label. Nothing is written until the caller saves them together."""
    items: list[tuple[dict[str, Any], str, Any]] = []
    failed: list[str] = []
    if queue is None or queue.empty:
        return items, failed
    for record in queue.to_dict(orient="records"):
        try:
            crop, _prominence = _all_hits_crop(
                record,
                candidates,
                crop_config,
                merge_radius_nm=merge_radius_nm,
                size_match_fraction=size_match_fraction,
                mosaic=mosaic,
                cache=cache,
                origin_cache=origin_cache,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            failed.append(str(exc))
            continue
        items.append((record, label, crop.rgb))
    return items, failed


def request_last_run() -> None:
    """Load the saved pipeline run. Pages stay empty until this runs."""
    restore_label_queue()
    st.session_state["load_last_run"] = True
    st.session_state["label_source"] = "last_pipeline"
    st.session_state["label_source_notice"] = True
    st.session_state.pop("label_front_key", None)


def _on_load_last_pipeline() -> None:
    """Switch Label to Detection's last output folder (runs before widgets)."""
    request_last_run()


def _on_empty_label_queue(csv_path: str) -> None:
    """Drop the remaining unlabeled hits for this results file."""
    clear_label_queue(csv_path)
    st.session_state.pop("label_front_key", None)
    st.session_state["all_hits_reject_keys"] = []


def _advance_inspect_tile(delta: int) -> None:
    """Move to the previous/next tile. Must run in a button callback (before widgets)."""
    names = list(st.session_state.get("_inspect_tile_names") or [])
    if not names:
        return
    current = st.session_state.get("inspect_tile")
    index = names.index(current) if current in names else 0
    st.session_state["inspect_tile"] = names[(index + delta) % len(names)]
    st.session_state["inspect_cell"] = None
    st.session_state.pop("inspect_pending", None)
    st.session_state.pop("inspect_click_sig", None)


def _on_inspect_tile_selected() -> None:
    st.session_state["inspect_cell"] = None
    st.session_state.pop("inspect_pending", None)
    st.session_state.pop("inspect_click_sig", None)


def _set_inspect_cell(row: int, col: int) -> None:
    current = st.session_state.get("inspect_cell")
    st.session_state["inspect_cell"] = None if current == (row, col) else (row, col)
    st.session_state.pop("inspect_pending", None)
    st.session_state.pop("inspect_click_sig", None)


def _clear_inspect_cell() -> None:
    st.session_state["inspect_cell"] = None
    st.session_state.pop("inspect_pending", None)
    st.session_state.pop("inspect_click_sig", None)


def _last_pipeline_view(
    config: dict[str, Any],
    table: pd.DataFrame | None,
) -> tuple[dict[str, Any], Path | None, pd.DataFrame | None]:
    """Folder, CSV, and detections from last_pipeline.json (not the sidebar)."""
    last_csv, run_config = load_last_pipeline(dict(config))
    detections: pd.DataFrame | None = None
    if last_csv is not None and last_csv.is_file():
        detections = load_particles_csv(last_csv)
    elif table is not None:
        detections = tag_table(table, last_csv)
    detections = _share_exposure_hits(detections, run_config)
    return run_config, last_csv, detections


def _pointer_tile_name(path: Path) -> str:
    """Select-box name. A combi in ``vN`` keeps that folder so it is not a quad."""
    if is_combi_tile(path.name) and submap_id(str(path)):
        return lighting_tile_id(str(path))
    return lighting_source_name(path)


def _review_tile_label(name: str) -> str:
    """Short tile name. Combi and uniform stay distinct from the quads."""
    set_id = submap_id(name)
    prefix = f"{set_id} " if set_id else ""
    if is_combi_tile(name):
        return f"{prefix}Combi"
    role = lighting_role(name)
    if role == "all":
        return f"{prefix}Uniform"
    if role and role.startswith("quad"):
        return f"{prefix}Quad {role[-1]}"
    return str(name)


def _review_tile_rank(name: str) -> tuple[str, int, str]:
    set_id = submap_id(name)
    if is_combi_tile(name):
        kind = 0
    elif lighting_role(name) == "all":
        kind = 1
    elif (lighting_role(name) or "").startswith("quad"):
        kind = 2 + int(str(lighting_role(name))[-1])
    else:
        kind = 20
    return (set_id, kind, str(name))


def render_tile_inspect_tab(
    config: dict[str, Any],
    project_root: str | Path,
    table: pd.DataFrame | None,
    mosaic: LazyMosaic | None,
) -> None:
    """Walk the last-pipeline tile folder, with detection and label circles."""
    st.header("Inspect tiles")
    if not st.session_state.get("load_last_run"):
        pipeline_csv, _pipeline_config = load_last_pipeline(config)
        st.button(
            "Load last run",
            key="inspect_load_last_run",
            disabled=pipeline_csv is None or not Path(pipeline_csv).is_file(),
            on_click=request_last_run,
        )
        st.caption("Last run stays unloaded until you click Load last run.")
        return

    try:
        run_config, last_csv, detections = _last_pipeline_view(config, table)
    except (ValueError, OSError) as exc:
        _failure("The last run could not be opened.", "Run the pipeline again.", exc)
        return
    folder = str(cfg_get(run_config, "input_dir", "") or "")
    if not folder:
        st.info("Run the pipeline, or set a tile folder under Adjust parameters on the Run tab.")
        return
    pattern = str(cfg_get(run_config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    source = last_csv if last_csv is not None else "sidebar"
    st.caption(f"Map {folder} · detections {source}")
    inspect_mosaic = mosaic
    if str(cfg_get(config, "input_dir", "") or "") != folder:
        inspect_mosaic = None
    pointer_tiles = pointer_direction_tiles(folder)
    by_label = {tile.label: tile for tile in pointer_tiles}
    if pointer_tiles:
        paths = [tile.path for tile in pointer_tiles]
        names = [tile.label for tile in pointer_tiles]
    else:
        try:
            paths = matching_tile_paths(
                folder,
                pattern,
                run=cfg_get(run_config, "run", None),
                magnification=cfg_get(run_config, "magnification", None),
                include_unmatched=True,
            )
        except (FileNotFoundError, ValueError) as exc:
            _failure("The tile folder could not be read.", "Choose the folder under Adjust parameters on the Run tab, then open Inspect again.", exc)
            return
        if not paths:
            st.warning(f"No matching tiles in {folder}.")
            return
        names = [_pointer_tile_name(path) for path in paths]
    st.session_state["_inspect_tile_names"] = names
    if st.session_state.get("inspect_tile") not in names:
        st.session_state["inspect_tile"] = names[0]
        st.session_state["inspect_cell"] = None
        st.session_state.pop("inspect_pending", None)
        st.session_state.pop("inspect_click_sig", None)
    nav = st.columns([1, 4, 1])
    nav[0].button(
        "Previous",
        shortcut="Left",
        width="stretch",
        key="inspect_prev",
        on_click=_advance_inspect_tile,
        args=(-1,),
    )
    nav[1].selectbox(
        "Tile",
        names,
        format_func=_review_tile_label,
        key="inspect_tile",
        on_change=_on_inspect_tile_selected,
    )
    nav[2].button(
        "Next",
        shortcut="Right",
        width="stretch",
        key="inspect_next",
        on_click=_advance_inspect_tile,
        args=(1,),
    )

    tile_name = str(st.session_state["inspect_tile"])
    pointer = by_label.get(tile_name)
    if pointer is not None:
        tile_path = pointer.path
    else:
        tile_path = next(path for path in paths if _pointer_tile_name(path) == tile_name)
    index = names.index(tile_name)
    st.caption(f"Tile {index + 1} of {len(names)} · {tile_path}")

    store = labels_store(project_root)
    labels_df = store.load()
    paths_by_name = (
        {tile.label: tile.path for tile in pointer_tiles}
        if pointer_tiles
        else {_pointer_tile_name(path): path for path in paths}
    )
    if st.button(
        "Verify all sizes in this folder",
        key="inspect_verify_sizes",
        help=(
            "Remeasure every missed click in this folder on the photograph. "
            "Replace the assumed size with that diameter. "
            "Drop the mark, and its dark green circle, when the photo is under 20 µm."
        ),
    ):
        with st.spinner("Measuring missed clicks on the photos…"):
            report = verify_missed_photo_sizes(
                labels_df,
                paths_by_name,
                run_config,
                pixel_size_nm=float(cfg_get(run_config, "pixel_size_nm", 960.0)),
            )
            written = store.update_sizes(
                {
                    row["key"]: row["photo_nm"]
                    for row in report
                    if row["note"] == "updated"
                }
            )
            removed = 0
            for row in report:
                if row["note"] != "removed":
                    continue
                if store.unlabel(str(row["key"])) is not None:
                    removed += 1
        st.session_state["inspect_size_report"] = report
        st.session_state["inspect_size_written"] = written
        st.session_state["inspect_size_removed"] = removed
        labels_df = store.load()
    report = st.session_state.get("inspect_size_report")
    if report is not None:
        written = int(st.session_state.get("inspect_size_written") or 0)
        removed = int(st.session_state.get("inspect_size_removed") or 0)
        if not report:
            st.info("No missed clicks in this folder.")
        else:
            st.caption(
                f"Photo sizes for {len(report)} missed clicks. "
                f"Updated {written}. Removed {removed} under 20 µm."
            )
            st.dataframe(
                pd.DataFrame(report).drop(columns=["photo_nm"]),
                width="stretch",
                hide_index=True,
            )
    scene_labels = _labels_for_scene(labels_df, config, names)

    if detections is None:
        detections = pd.DataFrame()
    else:
        label_frame = labels_df if pointer is not None else scene_labels
        detections = _collapse_review_hits(detections, label_frame, config)
    if pointer is not None:
        tile_hits = pointer_set_hits(detections, labels_df, pointer.set_id)
        label_map = label_map_with_aliases(labels_df)
    else:
        tile_hits = detections_on_review_picture(detections, tile_name)
        misses = pointer_miss_rows(
            scene_labels,
            detections,
            tile_name,
            folder_tiles=list(
                related_scene_tile_names(cfg_get(config, "input_dir", None), names) or names
            ),
        )
        if not misses.empty:
            tile_hits = pd.concat([tile_hits, misses], ignore_index=True)
        label_map = label_map_with_aliases(scene_labels)

    cache = st.session_state.setdefault("label_tile_cache", TileImageCache())
    origin_cache = st.session_state.setdefault("label_origin_cache", {})
    try:
        with st.spinner("Loading tile…"):
            image = cache.load(tile_path)
            placement = placement_for_tile(
                tile_path, run_config, mosaic=inspect_mosaic, origin_cache=origin_cache
            )
            placements = candidate_placements(
                tile_path, run_config, mosaic=inspect_mosaic, origin_cache=origin_cache
            )
    except (FileNotFoundError, ValueError, OSError) as exc:
        _failure("This crop could not be opened.", "Load the last run again, or pick another tile.", exc)
        return

    pixel_size = float(cfg_get(run_config, "pixel_size_nm", 960.0))
    height, width = int(image.shape[0]), int(image.shape[1])
    x_locals: list[float] = []
    y_locals: list[float] = []
    if not tile_hits.empty:
        for _, row in tile_hits.iterrows():
            x_local, y_local = local_xy_on_tile(
                float(row["x_global"]),
                float(row["y_global"]),
                placements,
                pixel_size,
                width,
                height,
            )
            x_locals.append(x_local)
            y_locals.append(y_local)
    circles = circles_from_rows(tile_hits, x_locals, y_locals, label_map)
    for circle, (_, row) in zip(circles, tile_hits.iterrows(), strict=False):
        missed = row["missed"] if "missed" in row else False
        if missed is not True and str(missed).lower() != "true":
            continue
        _diameter_nm, edge = photo_particle_edge(
            image,
            float(circle["y_local"]),
            float(circle["x_local"]),
            pixel_size_nm=pixel_size,
            config=run_config,
        )
        circle["edge"] = edge
    pending_diameter_nm = 0.0
    pending_edge: list = []
    pending_early = st.session_state.get("inspect_pending")
    pending_mass = footprint_grow_mass(
        run_config, float(st.session_state.get("inspect_footprint_grow", 1.0))
    )
    if pending_early and pending_early.get("tile") == tile_name:
        pending_diameter_nm, pending_edge = photo_particle_edge(
            image,
            float(pending_early["y_local"]),
            float(pending_early["x_local"]),
            pixel_size_nm=pixel_size,
            config=run_config,
            mass_fraction=pending_mass,
        )
        circles.append(
            {
                "x_local": float(pending_early["x_local"]),
                "y_local": float(pending_early["y_local"]),
                "size": (
                    pending_diameter_nm
                    if pending_diameter_nm > 0
                    else DEFAULT_MISS_SIZE_UM * NM_PER_UM
                ),
                "color": MISSED_PARTICLE_COLOR,
                "key": "",
                "edge": pending_edge,
            }
        )

    st.write("Zoom cell")
    selected = st.session_state.get("inspect_cell")
    for row_i in range(GRID_N):
        grid_cols = st.columns(GRID_N)
        for col_i in range(GRID_N):
            name = CELL_NAMES[row_i][col_i]
            grid_cols[col_i].button(
                name,
                width="stretch",
                type="primary" if selected == (row_i, col_i) else "secondary",
                key=f"inspect_cell_{row_i}_{col_i}",
                on_click=_set_inspect_cell,
                args=(row_i, col_i),
            )
    st.button(
        "Whole tile",
        key="inspect_cell_clear",
        on_click=_clear_inspect_cell,
    )

    crop = None
    max_side = OVERVIEW_MAX_SIDE
    caption = f"{tile_name} overview"
    if selected is not None:
        crop = cell_bounds(int(image.shape[0]), int(image.shape[1]), selected[0], selected[1])
        max_side = ZOOM_MAX_SIDE
        caption = (
            f"{tile_name} · {CELL_NAMES[selected[0]][selected[1]]} "
            f"({crop[3] - crop[1]}×{crop[2] - crop[0]} px)"
        )
    view = overlay_view(
        image, circles, pixel_size, crop=crop, max_side=max_side
    )
    display_w = min(
        INSPECT_DISPLAY_WIDTH,
        int(view.rgb.shape[1]) if view.rgb.size else INSPECT_DISPLAY_WIDTH,
    )
    if view.rgb.size == 0:
        st.error("No pixels to show for this view. Try Whole tile, then a cell again.")
        click = None
    else:
        st.caption(caption + " · click an unmarked speck")
        st.image(view.rgb, width=display_w)
        click = None
        if streamlit_image_coordinates is not None:
            with st.expander("Click to mark a miss", expanded=False):
                click = streamlit_image_coordinates(
                    view.rgb,
                    key="inspect_click_overlay",
                    width=display_w,
                    image_format="JPEG",
                    jpeg_quality=85,
                    cursor="crosshair",
                )
    if click:
        disp_scale = display_w / max(int(view.rgb.shape[1]), 1)
        sig = (int(click["x"]), int(click["y"]), tile_name, selected)
        if st.session_state.get("inspect_click_sig") != sig:
            x_local, y_local = display_xy_to_local(
                float(click["x"]) / disp_scale,
                float(click["y"]) / disp_scale,
                view,
            )
            st.session_state["inspect_click_sig"] = sig
            st.session_state["inspect_pending"] = {
                "x_local": x_local,
                "y_local": y_local,
                "tile": tile_name,
            }
            st.session_state["inspect_footprint_grow"] = 1.0
            st.rerun()

    pending = st.session_state.get("inspect_pending")
    if pending and pending.get("tile") == tile_name:
        size_um = float(
            st.number_input(
                "Assumed size (µm)",
                min_value=10.0,
                max_value=100.0,
                value=DEFAULT_MISS_SIZE_UM,
                step=1.0,
                key="inspect_miss_size_um",
                help="Missed flakes have no DoG size. 15 µm is the audit target.",
            )
        )
        size_nm = size_um * NM_PER_UM
        st.slider(
            "Grow yellow area",
            min_value=1.0,
            max_value=300.0,
            value=1.0,
            step=0.5,
            key="inspect_footprint_grow",
            help=(
                "1 is the automatic area. Move right when the particle is "
                "larger than the yellow. The diameter follows the yellow area."
            ),
        )
        measured_nm = float(pending_diameter_nm)
        shown_nm = measured_nm if measured_nm > 0 else size_nm
        preview = preview_click_crop(
            image,
            float(pending["x_local"]),
            float(pending["y_local"]),
            shown_nm,
            pixel_size,
            edge=pending_edge,
            zoom=8.0,
        )
        if measured_nm <= 0:
            edge_caption = "No footprint on the photo. The circle uses the assumed size."
        elif measured_nm < 20_000.0:
            edge_caption = (
                f"{measured_nm / NM_PER_UM:.1f} µm from the yellow area, under 20 µm."
            )
        else:
            edge_caption = f"{measured_nm / NM_PER_UM:.1f} µm from the yellow area."
        st.image(preview, caption=edge_caption, width=520)
        mark_cols = st.columns(2)
        if mark_cols[0].button(
            "Mark missed particle",
            type="primary",
            width="stretch",
            key="inspect_mark",
        ):
            x_global, y_global = local_px_to_global_nm(
                float(pending["x_local"]),
                float(pending["y_local"]),
                placement,
                pixel_size,
            )
            saved_nm = measured_nm if measured_nm > 0 else size_nm
            if 0 < saved_nm < 20_000.0:
                st.warning("That yellow area is under 20 µm, so it is not kept.")
            else:
                record = missed_particle_record(tile_name, x_global, y_global, saved_nm)
                saved = preview_click_crop(
                    image,
                    float(pending["x_local"]),
                    float(pending["y_local"]),
                    shown_nm,
                    pixel_size,
                    zoom=8.0,
                )
                store.apply_label(record, "particle", saved)
                st.session_state.pop("inspect_pending", None)
                st.session_state.pop("inspect_click_sig", None)
                st.rerun()
        if mark_cols[1].button("Cancel click", width="stretch", key="inspect_cancel"):
            st.session_state.pop("inspect_pending", None)
            st.session_state.pop("inspect_click_sig", None)
            st.rerun()

    if not tile_hits.empty:
        display = tile_hits.copy()
        display["size_um"] = display["size"].astype(float) / NM_PER_UM
        display["label"] = display["key"].astype(str).map(lambda key: label_map.get(key, ""))
        keep = [col for col in ("id", "size_um", "confidence", "label", "key") if col in display.columns]
        st.dataframe(display[keep], width="stretch", hide_index=True)


def _advance_labeled_review_tile(delta: int) -> None:
    names = list(st.session_state.get("_labeled_review_tile_names") or [])
    if not names:
        return
    current = st.session_state.get("labeled_review_tile")
    index = names.index(current) if current in names else 0
    st.session_state["labeled_review_tile"] = names[(index + delta) % len(names)]


def _last_run_stat_card(label: str, value: int | str, color: str, wash: str) -> str:
    shown = f"{value:,}" if isinstance(value, int) else value
    return (
        "<div style='flex:0 0 auto;width:max-content;padding:0.7rem 0.85rem;"
        f"border-radius:12px;background:{wash};border:1px solid {color}66;'>"
        f"<div style='font-size:0.75rem;font-weight:500;letter-spacing:0.03em;"
        f"text-transform:uppercase;color:{color};white-space:nowrap;'>{label}</div>"
        f"<div style='font-size:1.1rem;font-weight:600;line-height:1.2;"
        f"font-variant-numeric:tabular-nums;color:{color};white-space:nowrap;'>{shown}</div>"
        "</div>"
    )


def _last_run_rate_text(value: float, *, defined: bool) -> str:
    if not defined:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _last_run_std_text(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _last_run_f_beta_text(counts: dict[str, float], *, beta: float, defined: bool) -> str:
    if not defined:
        return "—"
    return _last_run_rate_text(
        f_beta_score(
            float(counts.get("precision", 0.0)),
            float(counts.get("recall", 0.0)),
            beta,
        ),
        defined=True,
    )


def _tile_crop_std(
    tile_name: str,
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    run_config: dict[str, Any],
    mosaic: LazyMosaic | None,
    origin_cache: dict[str, tuple[int, int]],
    *,
    min_size_nm: float,
    folder_tiles: list[str],
    beta: float,
) -> dict[str, float | None]:
    """Spread of P, R, and Fβ across the 3×3 crops of one tile."""
    return std_of_crop_cells(
        _tile_crop_cells(
            tile_name,
            detections,
            labels,
            run_config,
            mosaic,
            origin_cache,
            min_size_nm=min_size_nm,
            folder_tiles=folder_tiles,
            beta=beta,
        )
    )


def _tile_crop_cells(
    tile_name: str,
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    run_config: dict[str, Any],
    mosaic: LazyMosaic | None,
    origin_cache: dict[str, tuple[int, int]],
    *,
    min_size_nm: float,
    folder_tiles: list[str],
    beta: float,
) -> list[dict[str, Any]]:
    """P, R, and Fβ for each cell of one tile's 3×3 zoom grid."""
    try:
        tile_path = find_tile_path(
            tile_name, input_dir=cfg_get(run_config, "input_dir", None), mosaic=mosaic
        )
        placements = candidate_placements(
            tile_path, run_config, mosaic=mosaic, origin_cache=origin_cache
        )
    except (FileNotFoundError, ValueError, OSError):
        return []
    if not placements:
        return []
    height = int(placements[0].height)
    width = int(placements[0].width)
    return crop_metric_cells(
        detections,
        labels,
        height=height,
        width=width,
        placements=placements,
        pixel_size_nm=float(cfg_get(run_config, "pixel_size_nm", 960.0)),
        tile_names=[tile_name],
        min_size_nm=min_size_nm,
        folder_tiles=folder_tiles,
        beta=beta,
    )


def _sum_tile_counts(table: pd.DataFrame | None) -> dict[str, float]:
    """Add each tile's own totals. The same speck on two tiles counts twice."""
    keys = (
        "n_detected",
        "n_real",
        "n_fake",
        "n_unlabeled",
        "n_undetected",
        "n_undetected_below",
        "n_not_sure",
    )
    totals = {key: 0.0 for key in keys}
    if table is not None and not table.empty:
        for key in keys:
            if key in table.columns:
                totals[key] = float(pd.to_numeric(table[key], errors="coerce").fillna(0).sum())
    totals["n_real_total"] = totals["n_real"] + totals["n_undetected"]
    detected = totals["n_detected"]
    real_total = totals["n_real_total"]
    totals["precision"] = (totals["n_real"] / detected) if detected else 0.0
    totals["recall"] = (totals["n_real"] / real_total) if real_total else 0.0
    totals["unhappiness"] = unhappiness_score(totals["precision"], totals["recall"])
    return totals


def _render_last_run_summary(
    counts: dict[str, float],
    *,
    min_size_nm: float = 0.0,
    beta: float = 1.0,
    crop_std: dict[str, float | None] | None = None,
) -> None:
    floor_um = float(min_size_nm) / NM_PER_UM if min_size_nm else 20.0
    miss_title = f"Undetected real ≥ {floor_um:.0f} µm"
    counts_row = [
        _last_run_stat_card("Detected", int(counts["n_detected"]), "#5b6472", "#5b647214"),
        _last_run_stat_card("Real of detected", int(counts["n_real"]), "#1f9d3a", "#28b44622"),
        _last_run_stat_card("Fake of detected", int(counts["n_fake"]), "#d61f1f", "#e3262622"),
        _last_run_stat_card("Total real", int(counts["n_real_total"]), "#187a2e", "#28b44618"),
        _last_run_stat_card(miss_title, int(counts["n_undetected"]), "#c56a00", "#d9770618"),
    ]
    rates_defined = int(counts["n_detected"]) > 0 and int(counts["n_real_total"]) > 0
    rates_row = [
        _last_run_stat_card(
            "P",
            _last_run_rate_text(
                float(counts.get("precision", 0.0)),
                defined=int(counts["n_detected"]) > 0,
            ),
            "#4b5d8a",
            "#4b5d8a18",
        ),
        _last_run_stat_card(
            "R",
            _last_run_rate_text(
                float(counts.get("recall", 0.0)),
                defined=int(counts["n_real_total"]) > 0,
            ),
            "#6b3fa0",
            "#6b3fa018",
        ),
        _last_run_stat_card(
            "Fβ",
            _last_run_f_beta_text(counts, beta=beta, defined=rates_defined),
            "#8a3b2b",
            "#8a3b2b18",
        ),
    ]
    if crop_std is not None:
        rates_row.extend(
            [
                _last_run_stat_card(
                    "P std",
                    _last_run_std_text(crop_std.get("precision")),
                    "#4b5d8a",
                    "#4b5d8a18",
                ),
                _last_run_stat_card(
                    "R std",
                    _last_run_std_text(crop_std.get("recall")),
                    "#6b3fa0",
                    "#6b3fa018",
                ),
                _last_run_stat_card(
                    "Fβ std",
                    _last_run_std_text(crop_std.get("f_beta")),
                    "#8a3b2b",
                    "#8a3b2b18",
                ),
            ]
        )
    divider = (
        "<div style='flex:0 0 auto;width:1px;align-self:stretch;"
        "background:#b7bdc8;margin:0.15rem 1.1rem;'></div>"
    )
    row_style = (
        "display:flex;flex-wrap:wrap;align-items:stretch;gap:0.7rem;"
        "margin:0.35rem 0 0.85rem 0;"
    )
    st.markdown(
        f"<div style='{row_style}'>{''.join(counts_row)}{divider}{''.join(rates_row)}</div>",
        unsafe_allow_html=True,
    )
    extra = []
    if counts["n_unlabeled"]:
        extra.append(f"{int(counts['n_unlabeled']):,} unlabeled (orange)")
    if counts["n_not_sure"]:
        extra.append(f"{int(counts['n_not_sure']):,} not sure (blue)")
    if counts.get("n_undetected_below"):
        extra.append(
            f"{int(counts['n_undetected_below']):,} inspect marks below "
            f"{float(min_size_nm) / NM_PER_UM:.0f} µm not counted"
        )
    if extra:
        st.caption(" · ".join(extra) + ".")


def _render_size_threshold_table(table: pd.DataFrame | None) -> None:
    st.caption("Detected particles at least this large (equivalent circular diameter).")
    st.dataframe(particles_at_least(table), width="content", hide_index=True)


def _percent_cell(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "—"
    return f"{100.0 * float(number):.1f}%"


def _metric_row(
    tile: str,
    crop: str,
    *,
    precision: float | None,
    recall: float | None,
    beta: float,
    precision_std: float | None = None,
    recall_std: float | None = None,
    f_beta_std: float | None = None,
    show_std: bool = False,
) -> dict[str, str]:
    """One score row. Std columns are filled only for a whole-tile row."""
    both = precision is not None and recall is not None
    f_beta = f_beta_score(float(precision), float(recall), beta) if both else None
    blank = "—"
    return {
        "Tile": tile,
        "Crop": crop,
        "P": _percent_cell(precision),
        "R": _percent_cell(recall),
        "Fβ": _percent_cell(f_beta),
        "P std": _last_run_std_text(precision_std) if show_std else blank,
        "R std": _last_run_std_text(recall_std) if show_std else blank,
        "Fβ std": _last_run_std_text(f_beta_std) if show_std else blank,
    }


def _render_score_table(
    per_tile_table: pd.DataFrame | None,
    cells_by_tile: dict[str, list[dict[str, Any]]],
    *,
    beta: float,
) -> None:
    """One table: every crop, then each tile scored as a whole."""
    if per_tile_table is None or per_tile_table.empty or "tile" not in per_tile_table.columns:
        return
    order = [name for row in CELL_NAMES for name in row]
    crop_rows: list[dict[str, str]] = []
    whole_rows: list[dict[str, str]] = []
    for record in per_tile_table.to_dict(orient="records"):
        tile = str(record["tile"])
        label = _review_tile_label(tile)
        by_name = {
            CELL_NAMES[int(cell["row"])][int(cell["col"])]: cell
            for cell in cells_by_tile.get(tile, [])
            if "row" in cell and "col" in cell
        }
        for name in order:
            cell = by_name.get(name) or {}
            crop_rows.append(
                _metric_row(
                    label,
                    name,
                    precision=cell.get("precision"),
                    recall=cell.get("recall"),
                    beta=beta,
                )
            )
        detected = int(record.get("n_detected") or 0)
        real_total = int(record.get("n_real_total") or 0)
        whole_rows.append(
            _metric_row(
                label,
                "All together",
                precision=float(record.get("precision") or 0.0) if detected else None,
                recall=float(record.get("recall") or 0.0) if real_total else None,
                beta=beta,
                precision_std=record.get("precision_std"),
                recall_std=record.get("recall_std"),
                f_beta_std=record.get("f_beta_std"),
                show_std=True,
            )
        )
    rows = crop_rows + whole_rows
    if not rows:
        return
    st.caption(
        "Crops are the 3×3 zoom grid of that picture, NW through SE. "
        "All together is that tile on its own. "
        "P std, R std, and Fβ std are the spread of its nine crops."
    )
    st.table(pd.DataFrame(rows).set_index(["Tile", "Crop"]))


def render_labeled_tiles_tab(
    config: dict[str, Any],
    project_root: str | Path,
    mosaic: LazyMosaic | None,
    table: pd.DataFrame | None = None,
) -> None:
    """Latest pipeline run: tiles with detector circles and zoomed crops."""
    if not st.session_state.get("load_last_run"):
        return
    st.header("Labeled tiles")
    run_config = dict(config)
    session_config = st.session_state.get("config")
    if isinstance(session_config, dict):
        run_config.update(session_config)
    last_csv, pipeline_config = load_last_pipeline(run_config)
    for key in (
        "input_dir",
        "output_dir",
        "pixel_size_nm",
        "overlap_fraction",
        "filename_pattern",
    ):
        value = pipeline_config.get(key)
        if value not in (None, ""):
            run_config[key] = value
    detections: pd.DataFrame | None = None
    if last_csv is not None and last_csv.is_file():
        try:
            detections = load_particles_csv(last_csv)
        except (ValueError, OSError) as exc:
            _failure("The detection table could not be read.", "Run the pipeline again.", exc)
            return
        # The open session can still hold an older run. The summary follows the
        # CSV last_pipeline.json points at.
        mosaic = None
    elif table is not None:
        detections = tag_table(table, last_csv)
    detections = _share_exposure_hits(detections, run_config)
    if detections is None:
        st.info("Run the pipeline on the Main Pipeline tab. This view follows that run.")
        return
    folder_label = str(cfg_get(run_config, "input_dir", "") or "")
    if folder_label:
        st.caption(folder_label)

    store = labels_store(project_root)
    labels_df = store.load()
    run_tiles = input_tile_names(run_config)
    scene_tiles = related_scene_tile_names(
        cfg_get(run_config, "input_dir", None), run_tiles
    )
    if labels_df is not None and not labels_df.empty and (scene_tiles or run_tiles):
        keep_names = {Path(str(name)).name for name in (scene_tiles or ())}
        keep_names.update(Path(str(name)).name for name in run_tiles)
        keep_names.update(
            str(name).replace("\\", "/")
            for name in list(scene_tiles or ()) + list(run_tiles)
            if is_submap_lighting_tile(str(name))
        )
        labels_df = labels_df.loc[
            labels_df["source_tile"].astype(str).map(
                lambda name: label_on_run_tiles(name, keep_names)
            )
        ].copy()
    detections = _collapse_review_hits(detections, labels_df, run_config)
    recovery_tiles = sorted(run_tiles) if run_tiles else tile_names_for_hits(detections)
    size_floor_nm = undetected_size_floor_nm(detections, run_config)
    names = expand_submap_lighting_names(
        expand_direction_tile_names(
            tile_names_for_hits(detections), recovery_tiles or run_tiles
        ),
        recovery_tiles or run_tiles,
    )
    listed = set(names)
    for name in recovery_tiles or []:
        if name not in listed:
            names.append(name)
            listed.add(name)
    names = sorted(names, key=_review_tile_rank)
    metric_tiles = list(recovery_tiles or names)
    label_tiles = list(scene_tiles) if scene_tiles else metric_tiles
    per_tile_table = last_run_class_counts_by_tile(
        detections,
        labels_df,
        metric_tiles,
        min_size_nm=size_floor_nm,
        folder_tiles=label_tiles,
    )
    folder_counts = _sum_tile_counts(per_tile_table)
    raw_beta = cfg_get(config, "report.f_beta", 1.0)
    beta = 1.0 if raw_beta is None else float(raw_beta)
    origin_cache = st.session_state.setdefault("label_origin_cache", {})
    crop_cells_by_tile: dict[str, list[dict[str, Any]]] = {}
    crop_std_by_tile: dict[str, dict[str, float | None]] = {}
    if per_tile_table is not None and not per_tile_table.empty:
        for tile in per_tile_table["tile"].astype(str):
            cells = _tile_crop_cells(
                tile,
                detections,
                labels_df,
                run_config,
                mosaic,
                origin_cache,
                min_size_nm=size_floor_nm,
                folder_tiles=label_tiles,
                beta=beta,
            )
            crop_cells_by_tile[tile] = cells
            crop_std_by_tile[tile] = std_of_crop_cells(cells)
        per_tile_table = per_tile_table.copy()
        tiles = per_tile_table["tile"].astype(str)
        per_tile_table["precision_std"] = pd.to_numeric(
            [crop_std_by_tile[tile]["precision"] for tile in tiles], errors="coerce"
        )
        per_tile_table["recall_std"] = pd.to_numeric(
            [crop_std_by_tile[tile]["recall"] for tile in tiles], errors="coerce"
        )
        per_tile_table["f_beta_std"] = pd.to_numeric(
            [crop_std_by_tile[tile]["f_beta"] for tile in tiles], errors="coerce"
        )
    _render_score_table(per_tile_table, crop_cells_by_tile, beta=beta)

    scope = st.radio(
        "Metrics",
        options=["all_tiles", "per_tile"],
        format_func=lambda value: (
            "All tiles in folder" if value == "all_tiles" else "Tile by tile"
        ),
        horizontal=True,
        key="last_run_metrics_scope",
        help="Each tile is counted on its own. All tiles adds those tiles. "
        "A fake on one picture is not a fake on the others.",
    )

    if names:
        st.session_state["_labeled_review_tile_names"] = names
        if st.session_state.get("labeled_review_tile") not in names:
            st.session_state["labeled_review_tile"] = names[0]
        nav = st.columns([1, 4, 1])
        nav[0].button(
            "Previous",
            width="stretch",
            key="labeled_review_prev",
            on_click=_advance_labeled_review_tile,
            args=(-1,),
        )
        nav[1].selectbox(
            "Tile",
            names,
            format_func=_review_tile_label,
            key="labeled_review_tile",
        )
        nav[2].button(
            "Next",
            width="stretch",
            key="labeled_review_next",
            on_click=_advance_labeled_review_tile,
            args=(1,),
        )

    if scope == "per_tile":
        tile_name = str(st.session_state.get("labeled_review_tile") or "")
        if tile_name:
            counts = last_run_class_counts(
                detections,
                labels_df,
                tile_names=[tile_name],
                min_size_nm=size_floor_nm,
                folder_tiles=label_tiles,
            )
        else:
            counts = folder_counts
        selected = lighting_tile_id(tile_name) if tile_name else ""
        if selected and selected not in crop_std_by_tile and tile_name:
            crop_std_by_tile[selected] = _tile_crop_std(
                tile_name,
                detections,
                labels_df,
                run_config,
                mosaic,
                origin_cache,
                min_size_nm=size_floor_nm,
                folder_tiles=label_tiles,
                beta=beta,
            )
        _render_last_run_summary(
            counts,
            min_size_nm=size_floor_nm,
            beta=beta,
            crop_std=crop_std_by_tile.get(selected),
        )
        _render_size_threshold_table(
            detections_on_review_picture(detections, tile_name) if tile_name else detections
        )
    else:
        _render_last_run_summary(folder_counts, min_size_nm=size_floor_nm, beta=beta)
        _render_size_threshold_table(detections)

    if not names:
        st.info("The last pipeline run has no detections.")
        return

    tile_name = str(st.session_state["labeled_review_tile"])
    picture_labels = labels_df
    if (
        labels_df is not None
        and not labels_df.empty
        and "source_tile" in labels_df.columns
        and (
            is_submap_lighting_tile(tile_name)
            or (is_combi_tile(tile_name) and submap_id(tile_name))
        )
    ):
        wanted_id = lighting_tile_id(tile_name)
        picture_labels = labels_df.loc[
            labels_df["source_tile"].astype(str).map(lighting_tile_id) == wanted_id
        ]
    label_map = label_map_with_aliases(picture_labels)

    detected_hits = detections_on_review_picture(detections, tile_name)
    misses = pointer_miss_rows(
        labels_df, detections, tile_name, folder_tiles=label_tiles
    )
    if misses.empty:
        tile_hits = detected_hits
    elif detected_hits.empty:
        tile_hits = misses
    else:
        tile_hits = pd.concat([detected_hits, misses], ignore_index=True)

    cache = st.session_state.setdefault("label_tile_cache", TileImageCache())
    origin_cache = st.session_state.setdefault("label_origin_cache", {})
    try:
        tile_path = find_tile_path(
            tile_name, input_dir=cfg_get(run_config, "input_dir", None), mosaic=mosaic
        )
        image = cache.load(tile_path)
        placements = candidate_placements(
            tile_path, run_config, mosaic=mosaic, origin_cache=origin_cache
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        _failure("This crop could not be opened.", "Load the last run again, or pick another tile.", exc)
        return

    pixel_size = float(cfg_get(run_config, "pixel_size_nm", 960.0))
    height, width = int(image.shape[0]), int(image.shape[1])
    x_locals: list[float] = []
    y_locals: list[float] = []
    if not tile_hits.empty:
        for _, row in tile_hits.iterrows():
            x_local, y_local = local_xy_on_tile(
                float(row["x_global"]),
                float(row["y_global"]),
                placements,
                pixel_size,
                width,
                height,
            )
            x_locals.append(x_local)
            y_locals.append(y_local)
    circles = circles_from_rows(tile_hits, x_locals, y_locals, label_map)
    view = overlay_view(image, circles, pixel_size, crop=None, max_side=OVERVIEW_MAX_SIDE)
    if view.rgb.size == 0:
        st.error("No pixels to show for this tile.")
    else:
        st.image(
            view.rgb,
            caption=_review_tile_label(tile_name),
            width=LAST_RUN_OVERVIEW_WIDTH,
        )

    st.subheader("Zoomed detector hits")
    if detected_hits.empty:
        st.info("No detector hits on this tile.")
        return
    n_detected = len(detected_hits)
    n_pages = max(1, (n_detected + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=n_pages,
            key=f"labeled_review_gallery_{tile_name}",
        )
    )
    start = (page - 1) * GALLERY_PAGE_SIZE
    end = start + GALLERY_PAGE_SIZE
    columns = st.columns(LAST_RUN_CROP_COLUMNS)
    records = detected_hits.to_dict(orient="records")
    for offset, record in enumerate(records[start:end]):
        with columns[offset % LAST_RUN_CROP_COLUMNS]:
            x_local = x_locals[start + offset]
            y_local = y_locals[start + offset]
            key = str(record.get("key", ""))
            color = label_color_for_key(key, label_map)
            missed = record.get("missed")
            if (missed is True or str(missed).lower() == "true") and color == LABEL_CIRCLE_COLORS["particle"]:
                color = MISSED_PARTICLE_COLOR
                status_override = "missed"
            else:
                status_override = ""
            crop = preview_click_crop(
                image,
                x_local,
                y_local,
                float(record["size"]),
                pixel_size,
                color=color,
            )
            size_um = float(record["size"]) / NM_PER_UM
            status = status_override or label_map.get(key, "unlabeled")
            caption = (
                f"id {record.get('id', '')} · {size_um:.1f} µm · {status}"
            )
            st.image(crop, caption=caption, width="stretch")


def _label_rows_missing_from_hits(
    labels_df: pd.DataFrame,
    tile_name: str,
    hits: pd.DataFrame,
    folder_tiles: list[str] | None = None,
    own_tile_only: bool = False,
) -> pd.DataFrame | None:
    """Labeled marks on this tile that are not in the current detection CSV.

    ``own_tile_only`` keeps marks stored on this file. Sibling quads, the
    uniform shot, and other N/S/E/W files stay off this picture.
    """
    if labels_df is None or labels_df.empty:
        return None
    names = labels_df["source_tile"].astype(str).map(
        lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1]
    )
    if own_tile_only:
        wanted = lighting_tile_id(tile_name)
        on_tile = labels_df.loc[
            labels_df["source_tile"].astype(str).map(lighting_tile_id) == wanted
        ].copy()
        if "label" in on_tile.columns:
            on_tile = on_tile.loc[on_tile["label"].astype(str) == "particle"].copy()
        if "key" in on_tile.columns:
            on_tile = on_tile.drop_duplicates(subset=["key"], keep="last")
    else:
        on_tile = labels_df.loc[
            labels_df["source_tile"].astype(str).map(
                lambda name: label_on_run_tiles(name, {str(tile_name)})
            )
        ].copy()
        if is_nsew_family_tile(tile_name):
            allowed = {
                Path(name).name
                for name in (folder_tiles or [tile_name])
                if is_nsew_family_tile(str(name))
            }
            allowed.add(str(tile_name))
            on_tile = labels_df.loc[names.isin(allowed)].copy()
            if "key" in on_tile.columns:
                on_tile = on_tile.drop_duplicates(subset=["key"], keep="last")
        elif is_submap_lighting_tile(tile_name) or (
            is_combi_tile(tile_name) and submap_id(tile_name)
        ):
            set_id = submap_id(tile_name)
            sibling = labels_df["source_tile"].astype(str).map(
                lambda name: submap_id(name) == set_id
                and (is_submap_lighting_tile(name) or is_combi_tile(name))
            )
            if set_id and bool(sibling.any()):
                on_tile = labels_df.loc[sibling].copy()
                if "key" in on_tile.columns:
                    on_tile = on_tile.drop_duplicates(subset=["key"], keep="last")
    if on_tile.empty:
        return None
    known = set()
    if hits is not None and not hits.empty and "key" in hits.columns:
        known = expand_labeled_keys(hits["key"].astype(str))
    missing = on_tile.loc[~on_tile["key"].astype(str).isin(known)]
    if missing.empty:
        return None
    return missing.reset_index(drop=True)


def render_collection_tab(
    project_root: str | Path,
    label: str,
    title: str,
    n_columns: int = 4,
    campaign: str | None = None,
    config: dict[str, Any] | None = None,
    show_area_line: bool = False,
) -> None:
    """Thumbnail gallery for one label bucket, optionally one campaign."""
    store = labels_store(project_root)
    df = store.by_label(label)
    if campaign is not None:
        df = rows_for_campaign(df, campaign)
    st.header(title)
    st.caption(f"{len(df)} crops")
    if df.empty:
        st.info("No crops in this collection yet.")
        return

    display = df.copy()
    display.insert(
        display.columns.get_loc("size") + 1,
        "size_um",
        display["size"].astype(float) / NM_PER_UM,
    )
    st.dataframe(
        display.drop(columns=["crop_path"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )

    n_pages = max(1, (len(df) + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=n_pages,
            key=f"gallery_page_{campaign or 'all'}_{label}",
        )
    )
    start = (page - 1) * GALLERY_PAGE_SIZE
    chunk = df.iloc[start : start + GALLERY_PAGE_SIZE]
    n_columns = max(1, int(n_columns))
    columns = st.columns(n_columns)
    for index, record in enumerate(chunk.to_dict(orient="records")):
        with columns[index % n_columns]:
            path = Path(str(record["crop_path"]))
            folder = crop_source_folder(
                record.get("source_csv", ""), record.get("source_tile", "")
            )
            caption = (
                f"{record.get('key', '')} · "
                f"{float(record['size']) / NM_PER_UM:.1f} µm"
                + (f" · {folder}" if folder else "")
            )
            if path.is_file():
                shown = _collection_image(
                    path, record, label, config, show_area_line=show_area_line
                )
                st.image(
                    shown if shown is not None else str(path),
                    caption=caption,
                    width=GALLERY_IMAGE_WIDTH,
                )
            else:
                st.caption(f"Missing file: {path.name}")
            if st.button(
                "Send back to queue",
                key=f"unlabel_{campaign or 'all'}_{label}_{record['key']}",
                width="stretch",
            ):
                store.unlabel(str(record["key"]))
                st.session_state["label_front_key"] = str(record["key"])
                st.rerun()


def _collection_image(
    path: Path,
    record: dict[str, Any],
    label: str,
    config: dict[str, Any] | None,
    show_area_line: bool = False,
) -> np.ndarray | None:
    """Particle crops gain a thin #0F238C outline of the measured area."""
    if not show_area_line or label != "particle" or config is None:
        return None
    try:
        size_nm = float(record.get("size") or 0.0)
    except (TypeError, ValueError):
        size_nm = 0.0
    pixel_size = float(cfg_get(config, "pixel_size_nm", 960.0) or 960.0)
    try:
        return outline_particle_crop_file(
            path,
            size_nm=size_nm,
            pixel_size_nm=pixel_size,
            config=config,
        )
    except (OSError, ValueError):
        return None


def _source_caption(path: Path | None, *, last_pipeline: bool) -> str:
    if path is None:
        return "none"
    folder = path.parent.name or str(path.parent)
    name = path.name
    if last_pipeline:
        return f"{folder}/{name} (last output)"
    return str(path.name if path.is_file() else path)


def _last_pipeline_tables(csv_path: Path | None) -> list[pd.DataFrame]:
    if csv_path is None or not Path(csv_path).is_file():
        return []
    return [load_particles_csv(csv_path)]


def _detection_tables(
    project_root: str | Path,
    table: pd.DataFrame | None,
    config: dict[str, Any],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    last_csv = last_run_csv(project_root, config)
    if last_csv.is_file():
        frames.append(load_particles_csv(last_csv))
    out = resolve_output_dir(config.get("output_dir")) / "particles.csv"
    if out.is_file() and Path(out).resolve() != Path(last_csv).resolve():
        frames.append(load_particles_csv(out))
    if table is not None and not table.empty:
        frames.append(tag_table(table, out))
    return frames


def _undo_row(store: LabelStore, counts: dict[str, int]) -> None:
    labeled_n = sum(counts.values())
    if st.button(
        "Undo",
        shortcut="Up",
        key="label_undo_empty",
        disabled=labeled_n == 0,
    ):
        restored = store.undo_last()
        if restored is not None:
            st.session_state["label_front_key"] = str(restored["key"])
        st.rerun()
