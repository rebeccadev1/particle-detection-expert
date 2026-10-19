"""Orchestrate per-tile load → preprocess → detect → measure. Accumulate rows only."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.overall.config import cfg_get, resolve_output_dir, with_recall_profile
from src.main_pipeline.detection.detector import (
    apply_structure_filters,
    build_fft_notch_mask,
    detect_particles,
)
from src.overall.io.tile_loader import (
    Tile,
    load_tile_image,
    matching_tile_paths,
    peek_tile_hw,
    try_parse_tile_filename,
)
from src.main_pipeline.measurement.measurer import (
    Particle,
    combi_source_name,
    is_combi_tile,
    is_nsew_family_tile,
    is_submap_lighting_tile,
    lighting_source_name,
    measure_and_dedupe,
    measure_candidates,
    particles_to_dataframe,
    submap_id,
)
from src.main_pipeline.nsew import prepare_detection_paths
from src.main_pipeline.ml.infer import apply_ml_filter
from src.main_pipeline.preprocessing.corrections import apply_corrections
from src.main_pipeline.stitching.stitcher import (
    LazyMosaic,
    TilePlacement,
    downsample_for_target,
    downsample_image,
    mosaic_shape,
    placement_from_tile,
)


ProgressCallback = Callable[[int, int, str], None]
StopCheck = Callable[[], bool]


class PipelineStopped(Exception):
    """Raised when a run is asked to stop before the next tile."""

_WORKER_STATE: dict[str, Any] = {
    "fft_mask": None,
    "thumbnail_factor": 0,
}


@dataclass(frozen=True)
class _TileWorkResult:
    """Lightweight per-tile output returned from worker processes."""

    particles: list[Particle]
    placement: TilePlacement | None
    name: str
    thumbnail: np.ndarray | None


def _set_worker_state(
    fft_mask: np.ndarray | None = None,
    thumbnail_factor: int = 0,
    limit_threads: bool = False,
) -> None:
    if limit_threads:
        # One OpenCV thread per process so N workers do not oversubscribe.
        cv2.setNumThreads(1)
    _WORKER_STATE["fft_mask"] = fft_mask
    _WORKER_STATE["thumbnail_factor"] = int(thumbnail_factor)


def _tile_from_path(path: Path, pattern: str) -> Tile:
    meta = try_parse_tile_filename(path.name, pattern) or {}
    image = load_tile_image(path)
    height, width = int(image.shape[0]), int(image.shape[1])
    return Tile(
        path=path,
        name=path.name,
        image=image,
        row=int(meta["row"]) if "row" in meta else None,
        col=int(meta["col"]) if "col" in meta else None,
        run=int(meta["run"]) if "run" in meta else None,
        magnification=float(meta["mag"]) if "mag" in meta else None,
        x_origin=float(meta["x"]) if "x" in meta else None,
        y_origin=float(meta["y"]) if "y" in meta else None,
        height=height,
        width=width,
    )


def _tile_is_placeable(tile: Tile) -> bool:
    if tile.x_origin is not None and tile.y_origin is not None:
        return True
    return tile.row is not None and tile.col is not None


def _placement_from_path(
    path: Path,
    pattern: str,
    overlap: float,
    row_origin: int,
    col_origin: int,
) -> TilePlacement | None:
    """Grid placement from filename + TIFF header (no pixel decode)."""
    meta = try_parse_tile_filename(path.name, pattern)
    if meta is None:
        return None
    height, width = peek_tile_hw(path)
    tile = Tile(
        path=path,
        name=path.name,
        image=np.empty((0, 0), dtype=np.uint8),
        row=int(meta["row"]) if "row" in meta else None,
        col=int(meta["col"]) if "col" in meta else None,
        run=int(meta["run"]) if "run" in meta else None,
        magnification=float(meta["mag"]) if "mag" in meta else None,
        x_origin=float(meta["x"]) if "x" in meta else None,
        y_origin=float(meta["y"]) if "y" in meta else None,
        height=height,
        width=width,
    )
    if not _tile_is_placeable(tile):
        return None
    return placement_from_tile(tile, overlap, row_origin, col_origin)


def _grid_index_origin(paths: list[Path], pattern: str) -> tuple[int, int]:
    """Smallest row/col among tiles so the mosaic starts at the first real column."""
    rows: list[int] = []
    cols: list[int] = []
    for path in paths:
        meta = try_parse_tile_filename(path.name, pattern)
        if meta is None:
            continue
        if "row" in meta:
            rows.append(int(meta["row"]))
        if "col" in meta:
            cols.append(int(meta["col"]))
    return (min(rows) if rows else 0, min(cols) if cols else 0)


def _prepare_fft_mask(paths: list[Path], config: dict[str, Any]) -> np.ndarray | None:
    """Shared lattice mask from the first tile, or None for per-tile notches."""
    if str(cfg_get(config, "detection.method", "fft")).lower() != "fft":
        return None
    mode = str(cfg_get(config, "detection.fft_mask", "per_tile")).lower()
    if mode != "shared":
        return None
    corrected = apply_corrections(load_tile_image(paths[0]), config)
    return build_fft_notch_mask(
        corrected,
        float(cfg_get(config, "detection.fft_peak_threshold", 0.35)),
        int(cfg_get(config, "detection.fft_notch_radius", 3)),
    )


def _thumbnail_factor(placements: list[TilePlacement], config: dict[str, Any]) -> int:
    configured = int(cfg_get(config, "report.downsample", 0) or 0)
    if configured > 0:
        return configured
    height, width = mosaic_shape(placements)
    return downsample_for_target(
        height,
        width,
        target_mb=float(cfg_get(config, "report.target_mb", 20.0)),
        channels=3,
    )


def _process_tile(
    path_str: str,
    pattern: str,
    overlap: float,
    pixel_size: float,
    config: dict[str, Any],
    row_origin: int = 0,
    col_origin: int = 0,
) -> _TileWorkResult:
    """Load and process one tile. Top-level for ProcessPoolExecutor pickling."""
    tile = _tile_from_path(Path(path_str), pattern)
    if _tile_is_placeable(tile):
        placement = placement_from_tile(tile, overlap, row_origin, col_origin)
        origin_x = float(placement.x0)
        origin_y = float(placement.y0)
    else:
        placement = None
        origin_x = 0.0
        origin_y = 0.0
    corrected = apply_corrections(tile.image, config)
    ml_on = bool(cfg_get(config, "ml.enabled", False))
    score_before_structure = ml_on and bool(
        cfg_get(config, "ml.score_before_structure", True)
    )
    candidates = detect_particles(
        corrected,
        config,
        fft_mask=_WORKER_STATE.get("fft_mask"),
        structure_filters=not score_before_structure,
    )
    source_name = (
        combi_source_name(tile.path)
        if is_combi_tile(tile.path.name)
        else lighting_source_name(tile.path)
    )
    candidates = apply_ml_filter(
        candidates, config, image=corrected, source_tile=source_name
    )
    if score_before_structure:
        candidates = apply_structure_filters(candidates, config)
    particles = measure_candidates(
        candidates,
        origin_x=origin_x,
        origin_y=origin_y,
        pixel_size_nm=pixel_size,
        source_tile=source_name,
    )
    factor = int(_WORKER_STATE.get("thumbnail_factor") or 0)
    thumbnail = None
    if placement is not None and factor > 1:
        thumbnail = downsample_image(
            np.asarray(tile.image, dtype=np.float32), factor
        )
    return _TileWorkResult(
        particles=particles,
        placement=placement,
        name=source_name,
        thumbnail=thumbnail,
    )


def _worker_count(config: dict[str, Any], total_tiles: int) -> int:
    requested = int(cfg_get(config, "pipeline.workers", 0) or 0)
    if requested <= 0:
        requested = int(cpu_count() or 1)
    return max(1, min(requested, total_tiles))


def _collect_tile_result(
    result: _TileWorkResult,
    placed_particles: list[Particle],
    unplaced_particles: dict[str, list[Particle]],
    placements: list[TilePlacement],
    thumbnails: dict[str, np.ndarray],
) -> None:
    if result.placement is not None:
        placements.append(result.placement)
        placed_particles.extend(result.particles)
        if result.thumbnail is not None:
            thumbnails[result.name] = result.thumbnail
        return
    unplaced_particles.setdefault(result.name, []).extend(result.particles)


def _combine_particle_tables(
    placed_particles: list[Particle],
    unplaced_particles: dict[str, list[Particle]],
    config: dict[str, Any],
) -> pd.DataFrame:
    frames = [measure_and_dedupe(placed_particles, config)]
    directional: list[Particle] = []
    lighting: dict[str, list[Particle]] = {}
    for name, particles in unplaced_particles.items():
        # Combi joins the quads and uniform shot of the same vN folder so one
        # particle found on both pictures keeps one id.
        if is_submap_lighting_tile(name) or (is_combi_tile(name) and submap_id(name)):
            lighting.setdefault(submap_id(name) or name, []).extend(particles)
        elif is_nsew_family_tile(name):
            directional.extend(particles)
        else:
            frames.append(measure_and_dedupe(particles, config))
    fraction = float(cfg_get(config, "measurement.nsew_size_match_fraction", 0.4))
    for particles in lighting.values():
        # Four quads plus the uniform shot of one submap are one field.
        # Another vN folder is a different field, even with the same filenames.
        frames.append(
            measure_and_dedupe(
                particles,
                config,
                size_match_fraction=fraction,
                directional=True,
            )
        )
    if directional:
        # Same wafer location in N/W/E/S is one particle; reported size may differ.
        frames.append(
            measure_and_dedupe(
                directional,
                config,
                size_match_fraction=fraction,
                directional=True,
            )
        )
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return frames[0] if frames else particles_to_dataframe([])
    table = pd.concat(nonempty, ignore_index=True)
    table["id"] = np.arange(1, len(table) + 1)
    return table


def _raise_if_stopped(should_stop: StopCheck | None) -> None:
    if should_stop is not None and should_stop():
        raise PipelineStopped("Pipeline stopped.")


def run_pipeline(
    config: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
    *,
    write_outputs: bool = True,
    should_stop: StopCheck | None = None,
) -> tuple[pd.DataFrame, LazyMosaic]:
    """Run detection on each tile without retaining full-resolution images.

    Returns the de-duplicated particle table and a lazy mosaic for reporting.
    Mosaic thumbnails are generated during the detection pass when the overlay
    downsample factor is greater than 1.

    When ``write_outputs`` is true and ``output_dir`` is set, writes
    ``particles.csv``, ``particles.xlsx``, ``last_pipeline.json`` (every setting
    used for this run, in that output folder), ``mosaic_overlay.png`` when tiles
    were stitched, and ``last_run.pdf`` (counts, size histogram, and that mosaic).
    The workbook's second sheet is the same settings as the JSON file. The
    particle table keeps id, global x/y in µm, size in µm, source tile, and confidence.

    ``should_stop`` is checked between tiles. When it returns true the run
    raises ``PipelineStopped`` and does not write outputs.
    """
    config = with_recall_profile(config)
    folder = cfg_get(config, "input_dir", "")
    if not folder:
        raise ValueError("config.input_dir is required.")
    pattern = str(cfg_get(config, "filename_pattern"))
    overlap = float(cfg_get(config, "overlap_fraction", 0.0))
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    run_filter = cfg_get(config, "run", None)
    mag_filter = cfg_get(config, "magnification", None)

    paths = matching_tile_paths(
        folder, pattern, run_filter, mag_filter, include_unmatched=True
    )

    def _prepare_progress(current: int, total_cells: int, name: str) -> None:
        _raise_if_stopped(should_stop)
        if progress_cb is not None:
            progress_cb(current, total_cells, name)

    _raise_if_stopped(should_stop)
    paths = prepare_detection_paths(
        paths, _prepare_progress if progress_cb is not None or should_stop is not None else None
    )
    total = len(paths)
    if total == 0:
        raise FileNotFoundError(
            f"No image tiles in {folder}"
            + (
                f" (run={run_filter}, mag={mag_filter})"
                if run_filter not in (None, "") or mag_filter not in (None, "")
                else ""
            )
        )

    if progress_cb is not None:
        progress_cb(0, total, f"found {total} tiles…")
    _raise_if_stopped(should_stop)

    workers = _worker_count(config, total)
    row_origin, col_origin = _grid_index_origin(paths, pattern)
    placements_plan = [
        placement
        for path in paths
        if (placement := _placement_from_path(path, pattern, overlap, row_origin, col_origin))
        is not None
    ]
    thumb_factor = _thumbnail_factor(placements_plan, config) if placements_plan else 0
    fft_mask = _prepare_fft_mask(paths, config)
    _set_worker_state(fft_mask, thumb_factor)

    placed_particles: list[Particle] = []
    unplaced_particles: dict[str, list[Particle]] = {}
    placements: list[TilePlacement] = []
    thumbnails: dict[str, np.ndarray] = {}
    # Threads, not processes: ProcessPoolExecutor spawned from Streamlit hangs
    # on macOS (workers re-import the server) and the UI stays on "Idle".
    if progress_cb is not None:
        kind = "thread" if workers > 1 else "serial"
        progress_cb(0, total, f"starting {workers} {kind} worker(s)…")

    if workers == 1:
        for index, path in enumerate(paths, start=1):
            _raise_if_stopped(should_stop)
            result = _process_tile(
                str(path), pattern, overlap, pixel_size, config, row_origin, col_origin
            )
            _collect_tile_result(
                result, placed_particles, unplaced_particles, placements, thumbnails
            )
            if progress_cb is not None:
                progress_cb(index, total, result.name)
    else:
        cv2.setNumThreads(1)
        completed = 0
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = {
            pool.submit(
                _process_tile,
                str(path),
                pattern,
                overlap,
                pixel_size,
                config,
                row_origin,
                col_origin,
            ): path
            for path in paths
        }
        try:
            _raise_if_stopped(should_stop)
            for future in as_completed(futures):
                _raise_if_stopped(should_stop)
                result = future.result()
                _collect_tile_result(
                    result, placed_particles, unplaced_particles, placements, thumbnails
                )
                completed += 1
                if progress_cb is not None:
                    progress_cb(completed, total, result.name)
        except PipelineStopped:
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=False)

    _raise_if_stopped(should_stop)
    n_gathered = len(placed_particles) + sum(len(v) for v in unplaced_particles.values())
    if progress_cb is not None:
        progress_cb(total, total, f"merging {n_gathered} detections…")
    table = _combine_particle_tables(placed_particles, unplaced_particles, config)
    mosaic = LazyMosaic(
        placements,
        thumbnails=thumbnails or None,
        thumbnail_factor=thumb_factor if thumbnails else None,
    )
    if write_outputs:
        if progress_cb is not None:
            progress_cb(total, total, "writing report…")
        _write_run_outputs(table, mosaic, config)
    return table, mosaic


def _write_run_outputs(
    table: pd.DataFrame,
    mosaic: LazyMosaic,
    config: dict[str, Any],
) -> None:
    """Save the particle table, the settings that produced it, and the report."""
    if not str(cfg_get(config, "output_dir", "") or "").strip():
        return
    from src.overall.io.results_writer import write_csv, write_run_settings, write_xlsx
    from src.main_pipeline.report.report_generator import overlay_markers, write_last_run_pdf

    out = resolve_output_dir(cfg_get(config, "output_dir"))
    csv_path, _json_path, xlsx_path = default_output_paths(config)
    write_csv(table, csv_path)
    write_xlsx(table, xlsx_path, config)
    write_run_settings(config, out / "last_pipeline.json", csv_path=csv_path)

    mosaic_png = out / "mosaic_overlay.png"
    mosaic_jpg = out / "mosaic_overlay.jpg"
    mosaic_rgb = None
    if mosaic.placements:
        mosaic_rgb = overlay_markers(mosaic, table, config, crop=None)
        _write_overlay_png(mosaic_rgb, mosaic_png)
        if mosaic_jpg.is_file():
            mosaic_jpg.unlink()
    else:
        for stale in (mosaic_png, mosaic_jpg):
            if stale.is_file():
                stale.unlink()
    write_last_run_pdf(table, config, out / "last_run.pdf", mosaic_rgb=mosaic_rgb)


def _write_overlay_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write mosaic image to {path}")


def default_output_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """CSV, JSON, and Excel destinations under ``output_dir``."""
    out = resolve_output_dir(cfg_get(config, "output_dir"))
    return out / "particles.csv", out / "particles.json", out / "particles.xlsx"
