"""Full-tile inspection overlays for the recall-audit tab."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from src.overall.config import DEFAULT_INPUT_DIR, cfg_get, resolve_output_dir
from src.main_pipeline.detection.detector import blob_footprint
from src.overall.io.tile_loader import (
    DEFAULT_FILENAME_PATTERN,
    TILE_SUFFIXES,
    load_tile_image,
    parse_tile_filename,
    resolve_input_dir,
)
from src.verification.crops import (
    CIRCLE_COLOR,
    CIRCLE_THICKNESS,
    DEFAULT_WAFER_FOLDER,
    MIN_CIRCLE_RADIUS,
    MIN_HALF_PX,
    WINDOW_SCALE,
    direction_images_for_combined,
    draw_particle_circle,
    find_tile_path,
    global_nm_to_local_px,
    grayscale_as_rgb,
    placement_for_tile,
)
from src.verification.queue import (
    detection_key,
    expand_labeled_keys,
    label_map_with_aliases,
    nsew_key_aliases,
    tagged_if_needed,
)
from src.main_pipeline.measurement.measurer import (
    DIRECTION_STEMS,
    combi_key_family,
    is_combi_tile,
    is_direction_tile,
    is_grid_tile,
    is_nsew_family_tile,
    is_particles_only_tile,
    is_submap_lighting_tile,
    lighting_role,
    lighting_tile_id,
    submap_id,
)
from src.main_pipeline.stitching.stitcher import LazyMosaic, TilePlacement

LABEL_CIRCLE_COLORS: dict[str, tuple[int, int, int]] = {
    "particle": (40, 180, 70),
    "not_particle": (255, 40, 40),
    "not_sure": (60, 140, 220),
}
MISSED_PARTICLE_COLOR = (12, 78, 18)
UNLABELED_CIRCLE_COLOR = CIRCLE_COLOR
OVERVIEW_MAX_SIDE = 960
ZOOM_MAX_SIDE = 1080
INSPECT_DISPLAY_WIDTH = 960
GRID_N = 3
CELL_NAMES = (
    ("NW", "N", "NE"),
    ("W", "C", "E"),
    ("SW", "S", "SE"),
)
MISS_SOURCE = "tile_inspect"
MARKED_MOSAIC_NAME = "mosaic_marked.png"
DEFAULT_MISS_SIZE_UM = 15.0
MISSED_MIN_SIZE_NM = 20_000.0
_PARTICLES_ONLY_DIR_NAMES = {"particles only", "particles only output"}


@dataclass(frozen=True)
class OverlayView:
    """Displayed RGB plus the transform back to tile-local pixels."""

    rgb: np.ndarray
    crop_x0: int
    crop_y0: int
    scale: float


def cell_bounds(
    height: int,
    width: int,
    row: int,
    col: int,
    grid: int = GRID_N,
) -> tuple[int, int, int, int]:
    """Return ``(y0, x0, y1, x1)`` for one cell of a ``grid`` × ``grid`` split."""
    if grid < 1:
        raise ValueError("grid must be >= 1")
    y0 = int(height * row / grid)
    x0 = int(width * col / grid)
    y1 = int(height * (row + 1) / grid)
    x1 = int(width * (col + 1) / grid)
    return y0, x0, max(y1, y0 + 1), max(x1, x0 + 1)


def cell_of_local_px(
    x: float,
    y: float,
    height: int,
    width: int,
    grid: int = GRID_N,
) -> tuple[int, int]:
    """Cell whose ``cell_bounds`` contain ``(x, y)``. Points outside clamp to the edge."""
    if grid < 1:
        raise ValueError("grid must be >= 1")
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    x = min(max(float(x), 0.0), float(width) - 1e-9)
    y = min(max(float(y), 0.0), float(height) - 1e-9)
    for row in range(grid):
        for col in range(grid):
            y0, x0, y1, x1 = cell_bounds(height, width, row, col, grid)
            if x0 <= x < x1 and y0 <= y < y1:
                return row, col
    return grid - 1, grid - 1


def f_beta_score(precision: float, recall: float, beta: float) -> float:
    """Weighted harmonic mean of precision and recall. β = 1 is F1."""
    weight = float(beta) ** 2
    denominator = weight * float(precision) + float(recall)
    if denominator <= 0.0:
        return 0.0
    return (1.0 + weight) * float(precision) * float(recall) / denominator


def _sample_std(values: list[float]) -> float | None:
    """Sample standard deviation. Undefined when fewer than two scores exist."""
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def rows_detected_on_file(table: pd.DataFrame, tile_name: str) -> pd.Series:
    """True when this file is the row's source tile.

    A merged detection keeps one ``source_tile``. It is counted and drawn on
    that file only, not on every quad named in ``nsew_dirs``.
    """
    sources = table["source_tile"].astype(str)
    return sources.map(lighting_tile_id) == lighting_tile_id(tile_name)


def detections_on_this_file(table: pd.DataFrame | None, tile_name: str) -> pd.DataFrame:
    """Hits whose source file is ``tile_name``.

    A hit stored on quad 1 stays off quad 0, the uniform shot, and the combi.
    """
    if table is None or table.empty or "source_tile" not in table.columns:
        return pd.DataFrame()
    return table.loc[rows_detected_on_file(table, tile_name)].reset_index(drop=True)


def _nsew_dir_set(text: object) -> set[str]:
    return {
        part.strip().upper()
        for part in str(text or "").replace(";", ",").split(",")
        if part.strip() and part.strip().lower() != "nan"
    }


def detections_on_review_picture(
    table: pd.DataFrame | None, tile_name: str
) -> pd.DataFrame:
    """Hits to draw on a Pointer combi or uniform image.

    A uniform detection is marked on the uniform image even when the saved
    source file is a quad that was merged with it. A combi detection stays on
    the combi. A particle found on both is therefore marked on both pictures.
    Quads keep only the hits stored on that file.
    """
    own = detections_on_this_file(table, tile_name)
    if table is None or table.empty or "source_tile" not in table.columns:
        return own
    set_id = submap_id(tile_name)
    if lighting_role(tile_name) == "all":
        stem = "ALL"
    elif is_combi_tile(tile_name):
        stem = "COMBI"
    else:
        return own
    if not set_id or "nsew_dirs" not in table.columns:
        return own
    tiles = table["source_tile"].astype(str)
    same_set = tiles.map(submap_id) == set_id
    found_here = table["nsew_dirs"].map(lambda text: stem in _nsew_dir_set(text))
    extra = table.loc[same_set & found_here]
    if extra.empty:
        return own
    combined = pd.concat([own, extra], ignore_index=True)
    combined = tagged_if_needed(combined)
    if "key" in combined.columns:
        combined = combined.drop_duplicates(subset=["key"], keep="first")
    return combined.reset_index(drop=True)


def detections_on_review_tiles(
    table: pd.DataFrame | None, tile_names: Sequence[str]
) -> pd.DataFrame:
    """Hits drawn on any of ``tile_names``, one row per particle.

    A particle found on both the combi and the uniform shot is included once
    when both pictures are listed, and on each picture when listed alone.
    """
    if table is None or table.empty or not tile_names:
        return pd.DataFrame()
    frames = []
    for name in tile_names:
        shown = detections_on_review_picture(table, name)
        if shown is not None and not shown.empty:
            frames.append(shown)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "key" in combined.columns:
        combined = combined.drop_duplicates(subset=["key"], keep="first")
    return combined.reset_index(drop=True)


def _particle_label_rows(scoped: pd.DataFrame | None) -> pd.DataFrame:
    """Particle labels, one row per key."""
    if scoped is None or scoped.empty or "label" not in scoped.columns:
        return pd.DataFrame()
    particles = scoped.loc[scoped["label"].astype(str) == "particle"].copy()
    if particles.empty or "key" not in particles.columns:
        return particles.reset_index(drop=True)
    return particles.drop_duplicates(subset=["key"], keep="last").reset_index(drop=True)


def pointer_miss_rows(
    labels: pd.DataFrame | None,
    detections: pd.DataFrame | None,
    tile_name: str,
    folder_tiles: Sequence[str] | None = None,
    run_dir: str | None = None,
) -> pd.DataFrame:
    """Particle-pointer marks the detector missed on this tile.

    A click stays on the picture it was marked on. A hit on another tile in
    the same subfolder does not erase it, and does not draw it there.
    A ``(row,col)`` click also stays in its scan folder. A mark with no
    folder does not attach once ``run_dir`` is set.
    """
    if labels is None or labels.empty or "source_tile" not in labels.columns:
        return pd.DataFrame()
    if "label" not in labels.columns or "source_csv" not in labels.columns:
        return pd.DataFrame()
    marks = labels.loc[
        (labels["label"].astype(str) == "particle")
        & (labels["source_csv"].astype(str) == MISS_SOURCE)
    ].copy()
    if marks.empty:
        return pd.DataFrame()
    set_id = submap_id(tile_name)
    lighting_set = bool(set_id) and (
        is_submap_lighting_tile(tile_name) or is_combi_tile(tile_name)
    )
    if lighting_set:
        wanted_id = lighting_tile_id(tile_name)
        marks = marks.loc[
            marks["source_tile"].astype(str).map(lighting_tile_id) == wanted_id
        ]
    elif is_nsew_family_tile(tile_name):
        allowed = {
            Path(str(name)).name
            for name in (folder_tiles or [tile_name])
            if is_nsew_family_tile(str(name))
        }
        allowed.add(Path(str(tile_name)).name)
        names = marks["source_tile"].astype(str).map(
            lambda path: Path(str(path).replace("\\", "/")).name
        )
        marks = marks.loc[names.isin(allowed)]
    else:
        wanted = lighting_tile_id(tile_name)
        marks = marks.loc[
            marks["source_tile"].astype(str).map(lighting_tile_id) == wanted
        ]
        if run_dir and not marks.empty:
            grid = marks["source_tile"].astype(str).map(is_grid_tile)
            if bool(grid.any()) or is_grid_tile(tile_name):
                on_this_scan = marks["source_tile"].astype(str).map(
                    lambda name: label_on_run_tiles(name, {tile_name}, run_dir)
                )
                marks = marks.loc[~grid | on_this_scan]
    if marks.empty:
        return pd.DataFrame()
    if "key" in marks.columns:
        marks = marks.drop_duplicates(subset=["key"], keep="last")
    found: set[str] = set()
    if (
        detections is not None
        and not detections.empty
        and "key" in detections.columns
        and "source_tile" in detections.columns
    ):
        scoped = detections
        if lighting_set:
            scoped = detections_on_review_picture(detections, tile_name)
        if scoped is not None and not scoped.empty and "key" in scoped.columns:
            found = expand_labeled_keys(scoped["key"].astype(str))
    if found and "key" in marks.columns:
        marks = marks.loc[~marks["key"].astype(str).isin(found)]
    if marks.empty:
        return pd.DataFrame()
    marks = marks.copy()
    marks["missed"] = True
    return marks.reset_index(drop=True)


_STAGED_2OF4 = re.compile(r"^(v\d+)_2of4$", re.IGNORECASE)


@dataclass(frozen=True)
class PointerTile:
    """One N/S/E/W photo to show instead of a particles-only detection image."""

    label: str
    path: Path
    set_id: str
    direction: str


def pointer_direction_tiles(folder: str | Path | None) -> list[PointerTile]:
    """Directional photos for a staged ``vN_2of4`` detection folder.

    Empty when the folder is a normal tile set. Order is v1 N, S, E, W, then v2.
    """
    resolved = _resolve_scene_folder(folder) if folder not in (None, "") else None
    if resolved is None or not resolved.is_dir():
        return []
    staged: list[tuple[int, Path]] = []
    try:
        entries = list(resolved.iterdir())
    except OSError:
        return []
    for path in entries:
        match = _STAGED_2OF4.fullmatch(path.stem)
        if path.is_file() and match and path.suffix.lower() in TILE_SUFFIXES:
            staged.append((int(match.group(1)[1:]), path))
    staged.sort(key=lambda item: item[0])
    tiles: list[PointerTile] = []
    for _, staged_path in staged:
        directions = direction_images_for_combined(staged_path)
        set_id = _STAGED_2OF4.fullmatch(staged_path.stem).group(1)
        for direction in DIRECTION_STEMS:
            image = directions.get(direction)
            if image is None:
                continue
            tiles.append(
                PointerTile(
                    label=f"{set_id}/{direction}{image.suffix}",
                    path=image,
                    set_id=set_id,
                    direction=direction,
                )
            )
    return tiles


def _pointer_set_mask(source: pd.Series, set_id: str) -> pd.Series:
    text = source.astype(str).str.replace("\\", "/", regex=False)
    base = text.map(_basename)
    parent = text.map(lambda name: Path(name).parent.name)
    combined = base.str.match(rf"{re.escape(set_id)}_2of4\.", case=False)
    directional = parent.str.fullmatch(set_id, case=False) & base.map(is_direction_tile)
    return combined | directional


def pointer_set_hits(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    set_id: str,
) -> pd.DataFrame:
    """Detections and missed marks for one field, shared by its N/S/E/W photos."""
    hits = pd.DataFrame()
    if detections is not None and not detections.empty and "source_tile" in detections.columns:
        mask = _pointer_set_mask(detections["source_tile"], set_id)
        hits = tagged_if_needed(detections.loc[mask].copy())
        if not hits.empty and "key" in hits.columns:
            hits = hits.drop_duplicates(subset=["key"], keep="first")
    known = (
        expand_labeled_keys(hits["key"].astype(str))
        if not hits.empty and "key" in hits.columns
        else set()
    )
    extra = _labels_for_pointer_set(labels, set_id)
    if extra.empty:
        return hits.reset_index(drop=True)
    if "key" not in extra.columns:
        extra = extra.copy()
        extra["key"] = [detection_key(row) for row in extra.to_dict(orient="records")]
    keep: list[dict[str, Any]] = []
    for record in extra.to_dict(orient="records"):
        aliases = nsew_key_aliases(str(record.get("key") or detection_key(record)))
        if aliases.isdisjoint(known):
            keep.append(record)
            known.update(aliases)
    if not keep:
        return hits.reset_index(drop=True)
    added = pd.DataFrame(keep)
    added["missed"] = True
    if hits.empty:
        return added.reset_index(drop=True)
    return pd.concat([hits, added], ignore_index=True)


def _labels_for_pointer_set(labels: pd.DataFrame | None, set_id: str) -> pd.DataFrame:
    if labels is None or labels.empty:
        return pd.DataFrame()
    keys = labels["key"].astype(str) if "key" in labels.columns else pd.Series("", index=labels.index)
    tile_mask = (
        _pointer_set_mask(labels["source_tile"], set_id)
        if "source_tile" in labels.columns
        else pd.Series(False, index=labels.index)
    )
    key_mask = keys.str.startswith(f"{set_id}_NSEW_") | keys.str.startswith(f"{set_id}_2of4_")
    return labels.loc[tile_mask | key_mask].copy()


_STAGED_SET = re.compile(r"^(v\d+)_2of4$", re.IGNORECASE)
_SET_DIR = re.compile(r"^v\d+$", re.IGNORECASE)


def staged_set_id(source_tile: str) -> str:
    """Field id for ``v5/N.png`` and ``v5_2of4.png``. Empty for other tiles.

    A Groundup submap folder is also named ``v5``, but its combi and uniform
    shot are not the letter field ``v5_2of4.png``.
    """
    text = str(source_tile).replace("\\", "/")
    if is_combi_tile(text) or is_submap_lighting_tile(text):
        return ""
    parent = Path(text).parent.name
    if _SET_DIR.fullmatch(parent):
        return parent.lower()
    match = _STAGED_SET.fullmatch(Path(text).stem)
    if match:
        return match.group(1).lower()
    return ""


@lru_cache(maxsize=256)
def _directory_key(folder: str) -> str:
    """One id for a scan folder, whether it is stored relative or absolute."""
    text = str(folder).replace("\\", "/").strip().strip("/")
    if not text or text == ".":
        return ""
    resolved = _resolve_scene_folder(text)
    if resolved is None:
        return text
    try:
        return resolved.resolve().relative_to(DEFAULT_INPUT_DIR.resolve()).as_posix()
    except (ValueError, OSError):
        try:
            return resolved.resolve().as_posix()
        except OSError:
            return text


def _grid_parent(source_tile: str) -> str:
    parent = Path(str(source_tile).replace("\\", "/")).parent.as_posix()
    if parent in ("", "."):
        return ""
    return _directory_key(parent)


def _grid_id(source_tile: str, assume_dir: str | None = None) -> str:
    """Folder plus filename. A bare filename stays bare unless ``assume_dir`` is set."""
    name = _basename(source_tile)
    parent = _grid_parent(source_tile)
    if not parent and assume_dir:
        parent = _directory_key(assume_dir)
    if parent:
        return f"{parent}/{name}"
    return name


def with_grid_folder(record: Mapping[str, Any], run_dir: str | None) -> dict[str, Any]:
    """Copy a detection so a ``(row,col)`` label keeps this scan folder.

    The results file stores only ``(1,1).png``. Without the folder, the saved
    mark matches every scan that uses that filename and is then dropped.
    """
    row = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    tile = str(row.get("source_tile") or "")
    if not run_dir or not is_grid_tile(tile) or _grid_parent(tile):
        return row
    folder = _directory_key(str(run_dir))
    if not folder:
        return row
    row["source_tile"] = f"{folder}/{_basename(tile)}"
    return row


def label_on_run_tiles(
    source_tile: str,
    tile_names: set[str] | None,
    run_dir: str | None = None,
) -> bool:
    """True when this label belongs to one of the last-run tiles.

    A grid file matches the folder and the filename. ``(1,1).png`` in another
    scan does not. A Pointer click stored as ``v5/N.png`` belongs to
    ``v5_2of4.png``. ``v6/N.png`` does not. A grid mark with no folder does
    not attach to a run that has one.
    """
    if tile_names is None:
        return True
    if is_grid_tile(source_tile):
        if run_dir and not _grid_parent(source_tile):
            return False
        label_id = _grid_id(source_tile)
        for name in tile_names:
            other = _grid_id(name, assume_dir=run_dir if not _grid_parent(name) else None)
            if label_id == other:
                return True
        return False
    base = _basename(source_tile)
    light_review = is_submap_lighting_tile(source_tile) or (
        is_combi_tile(source_tile) and bool(submap_id(source_tile))
    )
    if light_review:
        tile_id = lighting_tile_id(source_tile)
        if tile_id in tile_names or base in tile_names:
            return True
        return any(lighting_tile_id(name) == tile_id for name in tile_names)
    if base in tile_names:
        return True
    set_id = staged_set_id(source_tile)
    if not set_id:
        return False
    suffix = Path(base).suffix.lower()
    return any(
        staged_set_id(name) == set_id and Path(name).suffix.lower() == suffix
        for name in tile_names
    )


def label_colors_for_hits(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    run_dir: str | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Review colour for each detection key on this run.

    A label from another scan folder is left out. Hits with no mark are absent,
    so the mosaic keeps them orange.
    """
    tagged = tagged_if_needed(detections) if detections is not None else pd.DataFrame()
    if tagged is None or tagged.empty or "key" not in tagged.columns:
        return {}
    scoped = labels
    if (
        scoped is not None
        and not scoped.empty
        and "source_tile" in scoped.columns
        and "source_tile" in tagged.columns
    ):
        tiles = set(tagged["source_tile"].astype(str))
        keep = scoped["source_tile"].astype(str).map(
            lambda name: label_on_run_tiles(name, tiles, run_dir)
        )
        scoped = scoped.loc[keep]
    label_map = label_map_with_aliases(scoped)
    colors: dict[str, tuple[int, int, int]] = {}
    for key in tagged["key"].astype(str):
        name = label_map.get(key, "")
        if name in LABEL_CIRCLE_COLORS:
            colors[key] = LABEL_CIRCLE_COLORS[name]
    return colors


def write_marked_mosaic(
    config: Mapping[str, Any],
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    *,
    mosaic: LazyMosaic | None = None,
) -> Path | None:
    """Write ``mosaic_marked.png`` in this run's output folder.

    The plain mosaic is copied and the review circles are drawn on it.
    Green, red, and blue are Label marks. Orange is still unlabeled.
    Dark green is a particle clicked in Inspect. Returns None when this
    run has no stitched mosaic. ``mosaic_overlay.png`` is left as it was.
    """
    settings = dict(config)
    output_dir = str(cfg_get(settings, "output_dir", "") or "").strip()
    if not output_dir:
        return None
    if mosaic is None or not getattr(mosaic, "placements", None):
        from src.main_pipeline.runner import load_report_mosaic

        mosaic = load_report_mosaic(settings)
    if not getattr(mosaic, "placements", None) or int(mosaic.full_width) < 1:
        return None

    from src.main_pipeline.report.report_generator import (
        draw_overlay_markers,
        overlay_rgb,
    )

    out = resolve_output_dir(output_dir)
    plain_path = out / "mosaic_plain.png"
    scale: float | None = None
    if plain_path.is_file():
        bgr = cv2.imread(str(plain_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Could not read {plain_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        scale = rgb.shape[1] / float(mosaic.full_width)
    else:
        rgb = overlay_rgb(mosaic, settings, crop=None)
    run_dir = str(cfg_get(settings, "input_dir", "") or "") or None
    frame, colors = _mosaic_review_marks(detections, labels, mosaic, run_dir)
    marked = draw_overlay_markers(
        rgb, mosaic, frame, settings, colors=colors, scale=scale
    )
    dest = out / MARKED_MOSAIC_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.cvtColor(np.asarray(marked), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(dest), written):
        raise OSError(f"Failed to write mosaic image to {dest}")
    return dest


def _mosaic_review_marks(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    mosaic: LazyMosaic,
    run_dir: str | None,
) -> tuple[pd.DataFrame, dict[str, tuple[int, int, int]]]:
    """Detection rows plus Inspect clicks, with a colour for each key."""
    tagged = tagged_if_needed(detections) if detections is not None else pd.DataFrame()
    if tagged is None:
        tagged = pd.DataFrame()
    colors = label_colors_for_hits(tagged, labels, run_dir=run_dir)
    tile_names = {placement.name for placement in mosaic.placements}
    if not tagged.empty and "source_tile" in tagged.columns:
        tile_names.update(tagged["source_tile"].astype(str))
    misses = _misses_for_mosaic(labels, tagged, tile_names, run_dir)
    if misses.empty:
        return tagged, colors
    placed = {placement.name for placement in mosaic.placements}
    fallback = next(iter(placed))

    def _on_mosaic(name: str) -> str:
        base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
        return base if base in placed else fallback

    misses = misses.copy()
    misses["source_tile"] = misses["source_tile"].astype(str).map(_on_mosaic)
    for key in misses["key"].astype(str):
        colors[key] = MISSED_PARTICLE_COLOR
    keep = [
        column
        for column in ("source_tile", "x_global", "y_global", "size", "key")
        if column in misses.columns
    ]
    extra = misses.loc[:, keep]
    if tagged.empty:
        return extra.reset_index(drop=True), colors
    return pd.concat([tagged, extra], ignore_index=True), colors


def _misses_for_mosaic(
    labels: pd.DataFrame | None,
    detections: pd.DataFrame,
    tile_names: set[str],
    run_dir: str | None,
) -> pd.DataFrame:
    """Inspect clicks on this run that are not already detector hits."""
    if (
        labels is None
        or labels.empty
        or "source_tile" not in labels.columns
        or "label" not in labels.columns
        or "source_csv" not in labels.columns
    ):
        return pd.DataFrame()
    marks = labels.loc[
        (labels["label"].astype(str) == "particle")
        & (labels["source_csv"].astype(str) == MISS_SOURCE)
    ].copy()
    if marks.empty or not tile_names:
        return pd.DataFrame()
    on_run = marks["source_tile"].astype(str).map(
        lambda name: label_on_run_tiles(name, tile_names, run_dir)
    )
    marks = marks.loc[on_run]
    if marks.empty:
        return pd.DataFrame()
    if "key" in marks.columns and not detections.empty and "key" in detections.columns:
        found = expand_labeled_keys(detections["key"].astype(str))
        marks = marks.loc[~marks["key"].astype(str).isin(found)]
    if marks.empty or "key" not in marks.columns:
        return pd.DataFrame()
    return marks.drop_duplicates(subset=["key"], keep="last").reset_index(drop=True)


def _tiles_for_folder_labels(
    tile_names: Sequence[str] | None,
    folder_tiles: Sequence[str] | None = None,
) -> set[str] | None:
    """Exact tile basenames whose labels count for this folder.

    N/S/E/W and particles-only versions share labels only among files that
    are actually in ``folder_tiles`` (or ``tile_names``). ``N.bmp`` from
    another folder is not mixed with ``N.png``.
    """
    if tile_names is None and folder_tiles is None:
        return None
    wanted = {
        str(name).replace("\\", "/") if is_combi_tile(name) else _basename(name)
        for name in (tile_names or ())
    }
    allowed = {
        _basename(name)
        for name in (folder_tiles if folder_tiles is not None else tile_names or ())
    }
    if any(is_nsew_family_tile(name) for name in wanted) or any(
        combi_key_family(name) == "nsew" for name in (tile_names or ())
    ):
        wanted.update(name for name in allowed if is_nsew_family_tile(name))
    for name in tile_names or ():
        if is_submap_lighting_tile(name) or (is_combi_tile(name) and submap_id(name)):
            wanted.add(lighting_tile_id(name))
    return wanted


def last_run_class_counts(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    tile_names: Sequence[str] | None = None,
    min_size_nm: float = 0.0,
    folder_tiles: Sequence[str] | None = None,
    run_dir: str | None = None,
) -> dict[str, float]:
    """Last-run detection totals split by label, plus missed inspect marks.

    * ``n_detected`` — every last-run hit on these tiles
    * ``n_real`` — detected and labeled particle (green)
    * ``n_fake`` — detected and not labeled as particle (red, orange, blue)
    * ``n_unlabeled`` — hits with no label (orange); also counted in ``n_fake``
    * ``n_undetected`` — labeled particles on these tiles whose keys are not
      in the last run (inspect marks and other misses), at least
      ``min_size_nm`` when that floor is > 0 (Last Run shows this as
      undetected real ≥ that size, typically 20 µm). A hit on another tile
      in the same subfolder does not count. A grid label from another folder
      does not count. ``run_dir`` is that folder.
    * ``n_real_total`` — ``n_real`` + ``n_undetected``
    * ``precision`` — ``n_real / n_detected`` (unlabeled hits count as fake)
    * ``recall`` — ``n_real / n_real_total``
    """
    tagged = tagged_if_needed(detections) if detections is not None else pd.DataFrame()
    if tile_names is not None and not tagged.empty and "source_tile" in tagged.columns:
        tagged = detections_on_review_tiles(tagged, tile_names)
    n_detected = int(len(tagged)) if tagged is not None and not tagged.empty else 0
    scoped = labels
    label_tiles = _tiles_for_folder_labels(tile_names, folder_tiles)
    if scoped is not None and not scoped.empty and label_tiles is not None:
        keep = scoped["source_tile"].astype(str).map(
            lambda name: label_on_run_tiles(name, label_tiles, run_dir)
        )
        scoped = scoped.loc[keep]
    label_map = label_map_with_aliases(scoped)
    particles = _particle_label_rows(scoped)

    n_real = n_fake = n_not_sure = n_unlabeled = 0
    det_keys: set[str] = set()
    if n_detected and "key" in tagged.columns:
        for key in tagged["key"].astype(str):
            det_keys.add(str(key))
            name = label_map.get(str(key), "")
            if name == "particle":
                n_real += 1
            else:
                n_fake += 1
                if name == "not_sure":
                    n_not_sure += 1
                elif name != "not_particle":
                    n_unlabeled += 1
    n_undetected = 0
    n_undetected_below = 0
    if not particles.empty and "key" in particles.columns:
        found = expand_labeled_keys(det_keys)
        missing = particles.loc[~particles["key"].astype(str).isin(found)]
        n_all_missing = int(len(missing))
        floor = float(min_size_nm or 0.0)
        if floor > 0 and not missing.empty and "size" in missing.columns:
            sizes = pd.to_numeric(missing["size"], errors="coerce")
            missing = missing.loc[sizes >= floor]
        n_undetected = int(len(missing))
        n_undetected_below = max(0, n_all_missing - n_undetected)
    n_real_total = n_real + n_undetected
    precision = (n_real / n_detected) if n_detected else 0.0
    recall = (n_real / n_real_total) if n_real_total else 0.0
    return {
        "n_detected": n_detected,
        "n_real": n_real,
        "n_fake": n_fake,
        "n_unlabeled": n_unlabeled,
        "n_undetected": n_undetected,
        "n_undetected_below": n_undetected_below,
        "n_not_sure": n_not_sure,
        "n_real_total": n_real_total,
        "precision": precision,
        "recall": recall,
    }


def last_run_class_counts_by_tile(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    tile_names: Sequence[str],
    min_size_nm: float = 0.0,
    folder_tiles: Sequence[str] | None = None,
    run_dir: str | None = None,
) -> pd.DataFrame:
    """``last_run_class_counts`` for each tile, one row per tile.

    Hits are those drawn on that picture. A shared combi and uniform detection
    counts on both pictures. Submap shots keep ``v4/filename`` so a quad in v4
    is not mixed with v5.
    ``folder_tiles`` is the label scene (e.g. N.png plus particles-only).
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    label_tiles = list(folder_tiles) if folder_tiles is not None else list(tile_names)
    for name in tile_names:
        tile = lighting_tile_id(name)
        if tile in seen:
            continue
        seen.add(tile)
        counts = last_run_class_counts(
            detections,
            labels,
            tile_names=[name],
            min_size_nm=min_size_nm,
            folder_tiles=label_tiles,
            run_dir=run_dir,
        )
        rows.append({"tile": tile, **counts})
    return pd.DataFrame(rows)


def std_of_crop_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Sample standard deviation of precision, recall, and Fβ across crops.

    A crop with no detections is left out of precision. A crop with no labeled
    real particles is left out of recall. Fβ uses only crops where both are
    defined. The result is undefined when fewer than two crops qualify.
    """
    precisions = [float(cell["precision"]) for cell in cells if cell.get("precision") is not None]
    recalls = [float(cell["recall"]) for cell in cells if cell.get("recall") is not None]
    f_betas = [float(cell["f_beta"]) for cell in cells if cell.get("f_beta") is not None]
    return {
        "precision": _sample_std(precisions),
        "recall": _sample_std(recalls),
        "f_beta": _sample_std(f_betas),
    }


def _score_records(
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if records is None:
        return []
    if isinstance(records, pd.DataFrame):
        if records.empty:
            return []
        return records.to_dict(orient="records")
    return list(records)


def std_of_tile_scores(
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    *,
    beta: float = 1.0,
) -> dict[str, float | None]:
    """Sample standard deviation of precision, recall, and Fβ across tiles.

    Each tile is one observation, including a tile with no detections or no
    labeled real particles (those scores are 0). The result is undefined when
    fewer than two tiles are present.
    """
    precisions: list[float] = []
    recalls: list[float] = []
    f_betas: list[float] = []
    for record in _score_records(records):
        n_detected = float(record.get("n_detected") or 0.0)
        n_real = float(record.get("n_real") or 0.0)
        n_undetected = float(record.get("n_undetected") or 0.0)
        n_real_total = float(record.get("n_real_total") or (n_real + n_undetected))
        precision = (n_real / n_detected) if n_detected else 0.0
        recall = (n_real / n_real_total) if n_real_total else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f_betas.append(f_beta_score(precision, recall, beta))
    return {
        "precision": _sample_std(precisions),
        "recall": _sample_std(recalls),
        "f_beta": _sample_std(f_betas),
    }


def subfolder_of_tile(tile: str) -> str:
    """``v7`` for a shot inside a ``vN`` folder. Empty when the tile sits in the run folder."""
    return submap_id(tile)


def subfolder_score_rows(
    per_tile: pd.DataFrame | None,
    *,
    folder_name: str = "",
    beta: float = 1.0,
    std_tiles: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """One score per subfolder, adding every tile that lives there.

    Tiles in the run folder, such as ``(1,1).png`` and ``(3,3).png``, share
    one row named for that folder. ``v5/…`` and ``v7/…`` stay on separate rows.
    Counts are summed. P and R are scored from those totals. The standard
    deviations are the spread of each tile's own P, R, and Fβ. ``std_tiles``
    is that sample when it should include every tile in the input folder,
    including tiles that contributed no counts.
    """
    if per_tile is None or per_tile.empty or "tile" not in per_tile.columns:
        return []
    plain_name = Path(str(folder_name).replace("\\", "/")).name or "Folder"
    order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in per_tile.to_dict(orient="records"):
        key = subfolder_of_tile(str(record["tile"]))
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(record)
    std_grouped: dict[str, list[Mapping[str, Any]]] = {}
    if std_tiles is not None and not std_tiles.empty and "tile" in std_tiles.columns:
        for record in std_tiles.to_dict(orient="records"):
            std_grouped.setdefault(subfolder_of_tile(str(record["tile"])), []).append(record)

    rows: list[dict[str, Any]] = []
    count_keys = (
        "n_detected",
        "n_real",
        "n_fake",
        "n_unlabeled",
        "n_undetected",
        "n_undetected_below",
        "n_not_sure",
    )
    for key in order:
        records = grouped[key]
        totals = {name: 0.0 for name in count_keys}
        for record in records:
            for name in count_keys:
                totals[name] += float(record.get(name) or 0.0)
        n_detected = int(totals["n_detected"])
        n_real = int(totals["n_real"])
        n_undetected = int(totals["n_undetected"])
        n_real_total = n_real + n_undetected
        spread = std_of_tile_scores(std_grouped.get(key, records), beta=beta)
        rows.append(
            {
                "name": key or plain_name,
                "n_detected": n_detected,
                "n_real": n_real,
                "n_undetected": n_undetected,
                "n_real_total": n_real_total,
                "precision": (n_real / n_detected) if n_detected else None,
                "recall": (n_real / n_real_total) if n_real_total else None,
                "precision_std": spread["precision"],
                "recall_std": spread["recall"],
                "f_beta_std": spread["f_beta"],
            }
        )
    return rows


def crop_metric_cells(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    *,
    height: int,
    width: int,
    placements: Sequence[TilePlacement],
    pixel_size_nm: float,
    tile_names: Sequence[str] | None = None,
    min_size_nm: float = 0.0,
    folder_tiles: Sequence[str] | None = None,
    beta: float = 1.0,
    grid: int = GRID_N,
    run_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Precision, recall, and Fβ for each cell of the ``grid`` × ``grid`` zoom split.

    Cells are row-major, matching ``CELL_NAMES``. ``precision`` is missing when
    the crop has no detections. ``recall`` is missing when it has no labeled
    real particles. ``f_beta`` is missing unless both are defined.
    """
    if height < 1 or width < 1 or not placements or grid < 1:
        return []
    tagged = tagged_if_needed(detections) if detections is not None else pd.DataFrame()
    if tile_names is not None and not tagged.empty and "source_tile" in tagged.columns:
        tagged = detections_on_review_tiles(tagged, tile_names)
    scoped = labels
    label_tiles = _tiles_for_folder_labels(tile_names, folder_tiles)
    if scoped is not None and not scoped.empty and label_tiles is not None:
        keep = scoped["source_tile"].astype(str).map(
            lambda name: label_on_run_tiles(name, label_tiles, run_dir)
        )
        scoped = scoped.loc[keep]
    label_map = label_map_with_aliases(scoped)
    particles = _particle_label_rows(scoped)

    cells = [
        {"n_detected": 0, "n_real": 0, "n_undetected": 0}
        for _ in range(grid * grid)
    ]
    det_keys: set[str] = set()
    if tagged is not None and not tagged.empty and "key" in tagged.columns:
        xs = pd.to_numeric(tagged["x_global"], errors="coerce") if "x_global" in tagged.columns else None
        ys = pd.to_numeric(tagged["y_global"], errors="coerce") if "y_global" in tagged.columns else None
        for index, key in enumerate(tagged["key"].astype(str)):
            det_keys.add(str(key))
            if xs is None or ys is None:
                continue
            x_global = float(xs.iloc[index])
            y_global = float(ys.iloc[index])
            if not (np.isfinite(x_global) and np.isfinite(y_global)):
                continue
            x_local, y_local = local_xy_on_tile(
                x_global, y_global, placements, pixel_size_nm, width, height
            )
            row, col = cell_of_local_px(x_local, y_local, height, width, grid)
            cell = cells[row * grid + col]
            cell["n_detected"] += 1
            if label_map.get(str(key), "") == "particle":
                cell["n_real"] += 1
    if not particles.empty and "key" in particles.columns:
        found = expand_labeled_keys(det_keys)
        missing = particles.loc[~particles["key"].astype(str).isin(found)]
        floor = float(min_size_nm or 0.0)
        if floor > 0 and not missing.empty and "size" in missing.columns:
            sizes = pd.to_numeric(missing["size"], errors="coerce")
            missing = missing.loc[sizes >= floor]
        if not missing.empty and "x_global" in missing.columns and "y_global" in missing.columns:
            xs = pd.to_numeric(missing["x_global"], errors="coerce")
            ys = pd.to_numeric(missing["y_global"], errors="coerce")
            for index in range(len(missing)):
                x_global = float(xs.iloc[index])
                y_global = float(ys.iloc[index])
                if not (np.isfinite(x_global) and np.isfinite(y_global)):
                    continue
                x_local, y_local = local_xy_on_tile(
                    x_global, y_global, placements, pixel_size_nm, width, height
                )
                row, col = cell_of_local_px(x_local, y_local, height, width, grid)
                cells[row * grid + col]["n_undetected"] += 1

    rated: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        n_detected = int(cell["n_detected"])
        n_real = int(cell["n_real"])
        n_real_total = n_real + int(cell["n_undetected"])
        precision = (n_real / n_detected) if n_detected else None
        recall = (n_real / n_real_total) if n_real_total else None
        f_beta = (
            f_beta_score(precision, recall, beta)
            if precision is not None and recall is not None
            else None
        )
        row, col = divmod(index, grid)
        rated.append(
            {
                **cell,
                "row": row,
                "col": col,
                "precision": precision,
                "recall": recall,
                "f_beta": f_beta,
            }
        )
    return rated


def undetected_size_floor_nm(
    detections: pd.DataFrame | None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Size gate for Last Run misses: this run's min size, else config."""
    floor = 0.0
    if config is not None:
        floor = float(cfg_get(dict(config), "detection.min_size_nm", 0.0) or 0.0)
    if detections is not None and not detections.empty and "size" in detections.columns:
        run_min = float(pd.to_numeric(detections["size"], errors="coerce").min())
        if np.isfinite(run_min) and run_min > 0:
            run_floor = float(np.floor(run_min / 1000.0) * 1000.0)
            floor = max(floor, run_floor)
    return floor


def tile_names_for_hits(hits: pd.DataFrame | None) -> list[str]:
    """Unique ``source_tile`` basenames, first-seen order."""
    if hits is None or hits.empty or "source_tile" not in hits.columns:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for name in hits["source_tile"].astype(str).map(lighting_tile_id):
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def expand_direction_tile_names(
    names: Sequence[str],
    available: Sequence[str] | None,
) -> list[str]:
    """If hits include N/S/E/W or particles-only, list every family file in the folder."""
    ordered = list(names)
    if available is None:
        return ordered
    if not any(is_nsew_family_tile(name) for name in ordered):
        return ordered
    have = {_basename(name) for name in ordered}
    extras = [
        _basename(name)
        for name in available
        if is_nsew_family_tile(name) and _basename(name) not in have
    ]

    def _family_sort(name: str) -> tuple[int, str]:
        stem = Path(name).stem.upper()
        if stem in DIRECTION_STEMS:
            return (DIRECTION_STEMS.index(stem), name)
        if is_particles_only_tile(name):
            return (10, name.lower())
        return (99, name.lower())

    extras.sort(key=_family_sort)
    return ordered + extras


def expand_submap_lighting_names(
    names: Sequence[str],
    available: Sequence[str] | None,
) -> list[str]:
    """If a submap shot is listed, include its quads and uniform ``all`` shot.

    Other submaps stay out, even when the filenames match.
    """
    ordered = list(names)
    if available is None:
        return ordered
    sets = {submap_id(name) for name in ordered if is_submap_lighting_tile(name) and submap_id(name)}
    if not sets:
        return ordered
    have = set(ordered)
    extras = [
        name
        for name in available
        if name not in have
        and submap_id(name) in sets
        and (is_submap_lighting_tile(name) or is_combi_tile(name))
    ]
    extras.sort(key=lambda name: (submap_id(name), lighting_tile_id(name)))
    return ordered + extras


def _scene_image_names(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    names: set[str] = set()
    try:
        entries = folder.iterdir()
    except OSError:
        return names
    for path in entries:
        if path.is_file() and path.suffix.lower() in TILE_SUFFIXES:
            names.add(path.name)
    return names


def _resolve_scene_folder(folder: str | Path) -> Path | None:
    text = str(folder).strip()
    if not text:
        return None
    try:
        return resolve_input_dir(text)
    except FileNotFoundError:
        path = Path(text).expanduser()
        try:
            if path.is_dir():
                return path.resolve()
        except OSError:
            return None
        return None


def scene_folders(folder: str | Path) -> list[Path]:
    """This folder plus sibling N/S/E/W and Particles-only dirs for the same scene."""
    resolved = _resolve_scene_folder(folder)
    if resolved is None:
        return []
    folder = resolved
    roots: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved_path = path.resolve()
        except OSError:
            return
        if resolved_path.is_dir() and resolved_path not in roots:
            roots.append(resolved_path)

    add(folder)
    lower = folder.name.lower()
    if lower in _PARTICLES_ONLY_DIR_NAMES:
        add(folder.parent)
        if folder.parent.name.lower().startswith("output "):
            add(folder.parent.parent)
            add(folder.parent / "Particles only")
    elif lower.startswith("output "):
        add(folder.parent)
        add(folder / "Particles only")
    else:
        add(folder / "Particles only")
        add(folder / "Particles only output")
        add(folder / f"Output {folder.name}")
        add(folder / f"Output {folder.name}" / "Particles only")
    return roots


def related_scene_tile_names(
    folder: str | Path | None,
    current_names: Sequence[str] | None = None,
) -> set[str]:
    """N/S/E/W and particles-only basenames for this scene, matching this folder's suffix.

    Groundup v3 (``.png``) is not mixed with Groundup v2 (``.bmp``).
    """
    current = {_basename(name) for name in (current_names or ())}
    if folder in (None, ""):
        return _with_family_suffix_names(current)
    path = _resolve_scene_folder(folder) or Path(str(folder))
    if not current:
        current = _scene_image_names(path)
    suffixes = {Path(name).suffix.lower() for name in current if Path(name).suffix}
    related = set(current)
    for root in scene_folders(folder):
        for name in _scene_image_names(root):
            if not is_nsew_family_tile(name):
                continue
            if suffixes and Path(name).suffix.lower() not in suffixes:
                continue
            related.add(name)
    return _with_family_suffix_names(related)


def _with_family_suffix_names(names: set[str]) -> set[str]:
    """If this scene is particles-only, also keep N/S/E/W with the same suffix."""
    related = set(names)
    suffixes = {Path(name).suffix.lower() for name in related if Path(name).suffix}
    if not suffixes:
        return related
    if not any(is_particles_only_tile(name) for name in related):
        return related
    if any(is_direction_tile(name) for name in related):
        return related
    for suffix in suffixes:
        related.update(f"{stem}{suffix}" for stem in DIRECTION_STEMS)
    return related


def label_color_for_key(key: str, labels: Mapping[str, str]) -> tuple[int, int, int]:
    name = labels.get(str(key))
    if name not in LABEL_CIRCLE_COLORS:
        for alias in nsew_key_aliases(str(key)):
            name = labels.get(alias)
            if name in LABEL_CIRCLE_COLORS:
                break
    if name in LABEL_CIRCLE_COLORS:
        return LABEL_CIRCLE_COLORS[name]
    return UNLABELED_CIRCLE_COLOR


def overlay_view(
    image: np.ndarray,
    circles: Sequence[Mapping[str, Any]],
    pixel_size_nm: float,
    *,
    crop: tuple[int, int, int, int] | None = None,
    max_side: int | None = OVERVIEW_MAX_SIDE,
) -> OverlayView:
    """Show ``image`` as stored grayscale, optional crop, downsample, draw circles."""
    gray = np.asarray(image)
    if crop is not None:
        y0, x0, y1, x1 = crop
        gray = gray[y0:y1, x0:x1]
        crop_x0, crop_y0 = int(x0), int(y0)
    else:
        crop_x0, crop_y0 = 0, 0
    rgb = grayscale_as_rgb(gray)
    if rgb.size == 0:
        return OverlayView(rgb=rgb, crop_x0=crop_x0, crop_y0=crop_y0, scale=1.0)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    scale = 1.0
    if max_side is not None and max(height, width) > int(max_side):
        scale = float(max_side) / float(max(height, width))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.ascontiguousarray(rgb).copy()
    for circle in circles:
        color = tuple(int(c) for c in circle.get("color", UNLABELED_CIRCLE_COLOR))
        x_s = (float(circle["x_local"]) - crop_x0) * scale
        y_s = (float(circle["y_local"]) - crop_y0) * scale
        size_nm = float(circle["size"])
        pixel = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
        radius = max(
            int(round(size_nm / pixel / 2.0 * scale)) + 10,
            MIN_CIRCLE_RADIUS,
        )
        x_d = int(round(x_s))
        y_d = int(round(y_s))
        if out.size == 0:
            continue
        if not (0 <= x_d < out.shape[1] and 0 <= y_d < out.shape[0]):
            continue
        cv2.circle(
            out,
            (x_d, y_d),
            radius,
            color,
            thickness=max(CIRCLE_THICKNESS, 3),
            lineType=cv2.LINE_AA,
        )
        _draw_footprint_edge(
            out,
            circle.get("edge") or [],
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            scale=scale,
        )
    return OverlayView(rgb=out, crop_x0=crop_x0, crop_y0=crop_y0, scale=scale)


FOOTPRINT_EDGE_COLOR = (255, 220, 40)


def _draw_footprint_edge(
    rgb: np.ndarray,
    edge: Sequence[np.ndarray],
    *,
    crop_x0: float = 0.0,
    crop_y0: float = 0.0,
    scale: float = 1.0,
    color: tuple[int, int, int] = FOOTPRINT_EDGE_COLOR,
    thickness: int = cv2.FILLED,
) -> None:
    """Draw the footprint. Contours are full-image ``(x, y)`` pixels.

    The default fills the area. A positive ``thickness`` strokes the outline.
    """
    if rgb.size == 0 or not edge:
        return
    drawn: list[np.ndarray] = []
    for contour in edge:
        pts = np.asarray(contour, dtype=np.float64)
        if pts.size == 0:
            continue
        pts = pts.copy()
        pts[..., 0] = (pts[..., 0] - crop_x0) * scale
        pts[..., 1] = (pts[..., 1] - crop_y0) * scale
        drawn.append(np.round(pts).astype(np.int32))
    if not drawn:
        return
    stroke = {"lineType": cv2.LINE_AA} if thickness != cv2.FILLED else {}
    cv2.drawContours(
        rgb,
        drawn,
        contourIdx=-1,
        color=color,
        thickness=thickness,
        **stroke,
    )


AREA_LINE_COLOR = (15, 35, 140)
AREA_LINE_THICKNESS = 2


def _saved_marker_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels of the saved marker circle, orange or green, allowing JPEG rounding."""
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    orange = (red > 190) & (green > 70) & (green < 200) & (blue < 100) & (red > green + 30)
    marked_green = (green > 140) & (green > red + 40) & (green > blue + 20) & (red < 140) & (
        blue < 140
    )
    if int(orange.sum()) >= int(marked_green.sum()):
        return orange
    return marked_green


def outline_particle_area(
    rgb: np.ndarray,
    *,
    size_nm: float,
    pixel_size_nm: float,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Copy of a saved crop with a thin #0F238C line around the measured area.

    The saved marker circle is left in place and is not part of the area. The line
    follows the same contrast footprint as the equivalent diameter: pixels
    that stand out by ``size_mass_fraction`` of the peak contrast.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return image
    shown = np.ascontiguousarray(image[:, :, :3])
    if shown.dtype != np.uint8:
        shown = np.clip(shown, 0, 255).astype(np.uint8)
    else:
        shown = shown.copy()
    marker = _saved_marker_mask(shown)
    gray = shown.astype(np.float64).mean(axis=2)
    height, width = gray.shape
    if int(marker.sum()) >= 12:
        ys, xs = np.where(marker)
        cy = float(np.median(ys))
        cx = float(np.median(xs))
        ring = float(np.median(np.hypot(ys.astype(np.float64) - cy, xs.astype(np.float64) - cx)))
        radius = max(ring - 10.0, 2.0)
        keep = ~marker
        fill = float(np.median(gray[keep])) if np.any(keep) else float(np.median(gray))
        gray = gray.copy()
        gray[marker] = fill
    else:
        cy = 0.5 * float(height - 1)
        cx = 0.5 * float(width - 1)
        scale = float(pixel_size_nm) if float(pixel_size_nm) > 0 else 1.0
        radius = max(float(size_nm) / scale / 2.0, 2.0)
    cfg = dict(config or {})
    mass = float(cfg_get(cfg, "detection.size_mass_fraction", 0.4) or 0.4)
    bright = bool(cfg_get(cfg, "detection.particles_bright", True))
    _diameter, edge = blob_footprint(
        gray,
        cy,
        cx,
        radius,
        mass_fraction=mass,
        bright=bright,
        half=max(height, width),
    )
    if not edge or _diameter <= 0:
        return shown
    area = np.pi * (0.5 * _diameter) ** 2
    if area > 0.45 * float(height * width):
        return shown
    _draw_footprint_edge(
        shown,
        edge,
        color=AREA_LINE_COLOR,
        thickness=AREA_LINE_THICKNESS,
    )
    return shown


def outline_particle_crop_file(
    path: str | Path,
    *,
    size_nm: float,
    pixel_size_nm: float,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    """Load a database JPEG and draw the measured-area outline."""
    file_path = Path(path)
    if not file_path.is_file():
        return None
    bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return outline_particle_area(
        rgb,
        size_nm=size_nm,
        pixel_size_nm=pixel_size_nm,
        config=config,
    )


def local_xy_on_tile(
    x_global: float,
    y_global: float,
    placements: Sequence[TilePlacement],
    pixel_size_nm: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Map globals to tile pixels using the first placement that lands in-frame.

    Detector CSVs use the full R3 mosaic origin. Click-marks from a 6-tile
    folder use that folder's origin. Trying both keeps circles on the image.
    """
    if not placements:
        raise ValueError("placements must not be empty")
    fallback = global_nm_to_local_px(
        x_global, y_global, placements[0], pixel_size_nm
    )
    for placement in placements:
        x_local, y_local = global_nm_to_local_px(
            x_global, y_global, placement, pixel_size_nm
        )
        if 0.0 <= x_local < float(width) and 0.0 <= y_local < float(height):
            return x_local, y_local
    return fallback


def extra_grid_placements(base: TilePlacement, config: Mapping[str, Any]) -> list[TilePlacement]:
    """Placements for last-run CSVs that used the full R3 grid, not the sidebar folder."""
    pattern = str(cfg_get(dict(config), "filename_pattern", DEFAULT_FILENAME_PATTERN))
    try:
        meta = parse_tile_filename(base.name, pattern)
    except ValueError:
        return []
    if "row" not in meta or "col" not in meta:
        return []
    row = int(meta["row"])
    col = int(meta["col"])
    configured = float(cfg_get(dict(config), "overlap_fraction", 0.0) or 0.0)
    overlaps: list[float] = []
    for overlap in (configured, 0.0, 0.1):
        if overlap not in overlaps:
            overlaps.append(float(overlap))
    placements: list[TilePlacement] = []
    seen: set[tuple[int, int]] = set()
    for overlap in overlaps:
        step_x = float(base.width) * (1.0 - overlap)
        step_y = float(base.height) * (1.0 - overlap)
        for row_origin, col_origin in ((1, 1), (0, 0)):
            x0 = int(round((col - col_origin) * step_x))
            y0 = int(round((row - row_origin) * step_y))
            key = (x0, y0)
            if key in seen:
                continue
            seen.add(key)
            placements.append(
                TilePlacement(
                    path=base.path,
                    name=base.name,
                    y0=y0,
                    x0=x0,
                    height=base.height,
                    width=base.width,
                )
            )
    return placements


def candidate_placements(
    tile_path: Any,
    config: Mapping[str, Any],
    mosaic: Any | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> list[TilePlacement]:
    """Current-folder placement, full-wafer grid, R3 04-08, then origin (0, 0)."""
    path = Path(tile_path)
    primary = placement_for_tile(
        path, dict(config), mosaic=mosaic, origin_cache=origin_cache
    )
    ordered = [primary]
    seen = {(int(primary.x0), int(primary.y0))}
    for extra in extra_grid_placements(primary, config):
        key = (int(extra.x0), int(extra.y0))
        if key not in seen:
            ordered.append(extra)
            seen.add(key)
    try:
        wafer_path = find_tile_path(path.name, input_dir=DEFAULT_WAFER_FOLDER)
        wafer_config = dict(config)
        wafer_config["input_dir"] = DEFAULT_WAFER_FOLDER
        wafer = placement_for_tile(
            wafer_path, wafer_config, mosaic=None, origin_cache=origin_cache
        )
        key = (int(wafer.x0), int(wafer.y0))
        if key not in seen:
            ordered.append(wafer)
            seen.add(key)
    except (FileNotFoundError, ValueError, OSError):
        pass
    if (0, 0) not in seen:
        ordered.append(
            TilePlacement(
                path=primary.path,
                name=primary.name,
                y0=0,
                x0=0,
                height=primary.height,
                width=primary.width,
            )
        )
    return ordered


def fresh_inspect_click(click: Mapping[str, Any] | None, seen: Any) -> int | None:
    """Click time when this press has not already been turned into a mark.

    The click widget keeps returning its last press after the picture changes.
    Opening a crop must not paint that stored press as a new dark-green circle.
    """
    if not isinstance(click, Mapping):
        return None
    try:
        stamp = int(click.get("unix_time"))
    except (TypeError, ValueError):
        return None
    if seen is not None:
        try:
            if stamp == int(seen):
                return None
        except (TypeError, ValueError):
            pass
    return stamp


def display_xy_to_local(
    x_display: float,
    y_display: float,
    view: OverlayView,
) -> tuple[float, float]:
    """Map a click on the displayed overlay back to tile-local pixels."""
    scale = float(view.scale) if float(view.scale) > 0 else 1.0
    x_local = float(view.crop_x0) + float(x_display) / scale
    y_local = float(view.crop_y0) + float(y_display) / scale
    return x_local, y_local


def missed_particle_record(
    tile_name: str,
    x_global: float,
    y_global: float,
    size_nm: float,
) -> dict[str, Any]:
    """CSV row for a click that was not a detector hit."""
    record: dict[str, Any] = {
        "id": "",
        "source_tile": str(tile_name),
        "x_global": float(x_global),
        "y_global": float(y_global),
        "size": float(size_nm),
        "confidence": 0.0,
        "source_csv": MISS_SOURCE,
    }
    record["key"] = detection_key(record)
    return record


def folder_missed_marks(
    labels: pd.DataFrame | None,
    tile_names: Sequence[str],
) -> pd.DataFrame:
    """Latest particle click-marks whose tile is shown in this Pointer folder."""
    if labels is None or labels.empty or not tile_names:
        return pd.DataFrame()
    names = {str(name).replace("\\", "/") for name in tile_names}
    tile = labels["source_tile"].astype(str).str.replace("\\", "/", regex=False)
    source = labels["source_csv"].astype(str) if "source_csv" in labels.columns else ""
    kind = labels["label"].astype(str) if "label" in labels.columns else ""
    kept = labels.loc[
        tile.isin(names) & (source == MISS_SOURCE) & (kind == "particle")
    ].copy()
    if kept.empty:
        return kept
    if "labeled_at" in kept.columns:
        times = pd.to_datetime(kept["labeled_at"], utc=True, errors="coerce")
        kept = kept.assign(_labeled_at=times).sort_values("_labeled_at")
        kept = kept.drop(columns="_labeled_at")
    if "key" in kept.columns:
        kept = kept.drop_duplicates(subset=["key"], keep="last")
    return kept.reset_index(drop=True)


def photo_equivalent_diameter_nm(
    image: np.ndarray,
    y_local: float,
    x_local: float,
    *,
    pixel_size_nm: float,
    config: Mapping[str, Any],
) -> float:
    """Equivalent circular diameter (nm) of the speck on the photograph.

    The window is the run's max size, so an assumed 15 µm click does not
    clip a larger flake. The center snaps to the brightest nearby pixel.
    """
    diameter_nm, _edge = photo_particle_edge(
        image,
        y_local,
        x_local,
        pixel_size_nm=pixel_size_nm,
        config=config,
    )
    return diameter_nm


def footprint_grow_mass(config: Mapping[str, Any], grow: float) -> float:
    """Brightness cut for a Pointer click.

    ``grow`` 1 keeps the automatic cut. 300 reaches the local background.
    """
    base = float(cfg_get(dict(config), "detection.size_mass_fraction", 0.4) or 0.4)
    span = max(0.0, min(299.0, float(grow) - 1.0))
    return float(max(0.0, base * (1.0 - span / 299.0)))


def photo_particle_edge(
    image: np.ndarray,
    y_local: float,
    x_local: float,
    *,
    pixel_size_nm: float,
    config: Mapping[str, Any],
    mass_fraction: float | None = None,
) -> tuple[float, list[np.ndarray]]:
    """Photo diameter (nm) and the footprint outline in full-image pixels.

    Diameter is ``2 √(A / π)`` for the yellow area. The center snaps to the
    brightest nearby pixel, same as size verification. ``mass_fraction``
    overrides the automatic brightness cut when the click area is grown.
    """
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    pixel = float(pixel_size_nm) if float(pixel_size_nm) > 0 else 1.0
    max_nm = float(cfg_get(dict(config), "detection.max_size_nm", 100_000.0) or 100_000.0)
    mass = (
        float(mass_fraction)
        if mass_fraction is not None
        else float(cfg_get(dict(config), "detection.size_mass_fraction", 0.4))
    )
    bright = bool(cfg_get(dict(config), "detection.particles_bright", True))
    max_d = max(max_nm / pixel, 8.0)
    radius = 0.5 * max_d
    half = max(int(np.ceil(radius)) + 4, 8)
    search = max(4.0, min(12.0, 0.25 * max_d))
    y_peak, x_peak = _snap_to_peak(gray, y_local, x_local, search, bright=bright)
    diameter_px, edge = blob_footprint(
        gray,
        y_peak,
        x_peak,
        radius,
        mass_fraction=mass,
        bright=bright,
        half=half,
    )
    return float(diameter_px) * pixel, edge


def verify_missed_photo_sizes(
    labels: pd.DataFrame | None,
    paths_by_name: Mapping[str, Path],
    config: Mapping[str, Any],
    *,
    pixel_size_nm: float,
) -> list[dict[str, Any]]:
    """Measure each missed click in this folder on its photo.

    ``photo_nm`` is 0 when the click has no footprint. Those rows keep the
    stored size. A measured diameter under 20 µm is marked ``removed``.
    """
    marks = folder_missed_marks(labels, list(paths_by_name))
    if marks.empty:
        return []
    images: dict[str, np.ndarray] = {}
    placements: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for record in marks.to_dict(orient="records"):
        tile_name = str(record["source_tile"]).replace("\\", "/")
        path = paths_by_name.get(tile_name)
        stored = float(record.get("size") or 0.0)
        key = str(record.get("key") or detection_key(record))
        if path is None or not Path(path).is_file():
            rows.append(_size_row(key, tile_name, stored, 0.0))
            continue
        if tile_name not in images:
            images[tile_name] = load_tile_image(path)
            placements[tile_name] = placement_for_tile(Path(path), dict(config))
        image = images[tile_name]
        height, width = int(image.shape[0]), int(image.shape[1])
        x_local, y_local = local_xy_on_tile(
            float(record["x_global"]),
            float(record["y_global"]),
            [placements[tile_name]],
            pixel_size_nm,
            width,
            height,
        )
        photo = photo_equivalent_diameter_nm(
            image,
            y_local,
            x_local,
            pixel_size_nm=pixel_size_nm,
            config=config,
        )
        rows.append(_size_row(key, tile_name, stored, photo))
    return rows


def _size_row(key: str, tile_name: str, stored_nm: float, photo_nm: float) -> dict[str, Any]:
    assumed_nm = DEFAULT_MISS_SIZE_UM * 1000.0
    grown = stored_nm >= MISSED_MIN_SIZE_NM and abs(stored_nm - assumed_nm) > 1.0
    if photo_nm <= 0:
        note = "kept stored size"
    elif grown and photo_nm + 1.0 < stored_nm:
        note = "kept stored size"
    elif photo_nm < MISSED_MIN_SIZE_NM:
        note = "removed"
    elif abs(photo_nm - stored_nm) >= 1.0:
        note = "updated"
    else:
        note = "already the photo size"
    return {
        "key": key,
        "source_tile": tile_name,
        "stored_um": stored_nm / 1000.0,
        "photo_um": photo_nm / 1000.0,
        "photo_nm": photo_nm,
        "note": note,
    }


def _snap_to_peak(
    image: np.ndarray,
    y: float,
    x: float,
    search_px: float,
    *,
    bright: bool,
) -> tuple[float, float]:
    height, width = int(image.shape[0]), int(image.shape[1])
    iy, ix = int(round(y)), int(round(x))
    radius = int(max(1, round(search_px)))
    y0, y1 = max(0, iy - radius), min(height, iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(width, ix + radius + 1)
    patch = np.asarray(image[y0:y1, x0:x1])
    if patch.size == 0:
        return float(y), float(x)
    flat = int(np.argmax(patch) if bright else np.argmin(patch))
    py, px = divmod(flat, patch.shape[1])
    return float(y0 + py), float(x0 + px)


def preview_click_crop(
    image: np.ndarray,
    x_local: float,
    y_local: float,
    size_nm: float,
    pixel_size_nm: float,
    color: tuple[int, int, int] | None = None,
    edge: Sequence[np.ndarray] | None = None,
    zoom: float = 1.0,
) -> np.ndarray:
    """Full-res crop around a hit, padded so the circle is not clipped.

    ``zoom`` above 1 tightens the window onto the speck and magnifies it.
    Pointer uses that so a few-pixel flake is visible.
    """
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    diameter_px = float(size_nm) / scale
    radius = max(int(round(diameter_px / 2.0)) + 10, MIN_CIRCLE_RADIUS)
    if float(zoom) > 1.0:
        half = max(
            int(radius + CIRCLE_THICKNESS + 12),
            int(round(1.8 * float(diameter_px))),
        )
    else:
        half = max(
            int(MIN_HALF_PX),
            int(round(float(WINDOW_SCALE) * float(diameter_px))),
            int(radius + CIRCLE_THICKNESS + 24),
        )
    height, width = int(image.shape[0]), int(image.shape[1])
    cx = int(round(float(x_local)))
    cy = int(round(float(y_local)))
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)
    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(height, y1), min(width, x1)
    patch = np.asarray(image[y0c:y1c, x0c:x1c])
    if pad_top or pad_bottom or pad_left or pad_right:
        patch = cv2.copyMakeBorder(
            patch,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
    rgb = grayscale_as_rgb(patch)
    draw_particle_circle(
        rgb,
        x_local,
        y_local,
        size_nm,
        pixel_size_nm,
        crop_x0=x0,
        crop_y0=y0,
        color=LABEL_CIRCLE_COLORS["particle"] if color is None else color,
    )
    _draw_footprint_edge(rgb, edge or [], crop_x0=x0, crop_y0=y0, scale=1.0)
    factor = float(zoom)
    if factor > 1.0 and rgb.size:
        side = max(int(rgb.shape[0]), int(rgb.shape[1]))
        if side * factor > 720.0:
            factor = 720.0 / float(side)
        if factor > 1.01:
            rgb = cv2.resize(
                rgb,
                (
                    max(1, int(round(rgb.shape[1] * factor))),
                    max(1, int(round(rgb.shape[0] * factor))),
                ),
                interpolation=cv2.INTER_NEAREST,
            )
    return rgb


def circles_from_rows(
    rows: pd.DataFrame,
    x_locals: Sequence[float],
    y_locals: Sequence[float],
    labels: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build overlay circle dicts, tagging each row with its label colour."""
    circles: list[dict[str, Any]] = []
    for (x_local, y_local), (_, row) in zip(
        zip(x_locals, y_locals, strict=True), rows.iterrows(), strict=True
    ):
        key = str(row["key"]) if "key" in row and pd.notna(row["key"]) else detection_key(row)
        color = label_color_for_key(key, labels)
        missed = row["missed"] if "missed" in row else False
        if missed is True or str(missed).lower() == "true":
            if color == LABEL_CIRCLE_COLORS["particle"]:
                color = MISSED_PARTICLE_COLOR
        circles.append(
            {
                "x_local": float(x_local),
                "y_local": float(y_local),
                "size": float(row["size"]),
                "color": color,
                "key": key,
            }
        )
    return circles


def _basename(path: str) -> str:
    text = str(path).replace("\\", "/")
    return text.rsplit("/", 1)[-1]
