from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.overall.config import DEFAULT_INPUT_DIR, cfg_get
from src.overall.io.tile_loader import (
    DEFAULT_FILENAME_PATTERN,
    TILE_SUFFIXES,
    load_tile_image,
    matching_tile_paths,
    peek_tile_hw,
    resolve_input_dir,
    try_parse_tile_filename,
)
from src.analyse.nsew import positional_cell_sides, positional_direction
from src.overall.io.naming import (
    DIRECTION_STEMS,
    is_combi_tile,
    lighting_role,
    quad_direction,
    is_direction_tile,
    is_grid_tile,
    is_particles_only_tile,
    is_submap_lighting_tile,
    submap_id,
)
from src.overall.io.markers import CIRCLE_COLOR, draw_marker, marker_radius
from src.overall.io.placement import (
    TilePlacement,
    grid_index_origin,
    placement_from_path,
)
from src.analyse.stitching.stitcher import LazyMosaic

MIN_HALF_PX = 96
WINDOW_SCALE = 1.8


@dataclass(frozen=True)
class ParticleCrop:
    

    rgb: np.ndarray
    x_local: float
    y_local: float
    crop_x0: int
    crop_y0: int
    tile_path: Path
    direction: str = ""


class TileImageCache:
    

    def __init__(self) -> None:
        self.path: Path | None = None
        self.image: np.ndarray | None = None

    def load(self, path: Path) -> np.ndarray:
        resolved = Path(path).resolve()
        if self.path == resolved and self.image is not None:
            return self.image
        self.image = load_tile_image(resolved)
        self.path = resolved
        return self.image


def global_nm_to_local_px(
    x_global: float,
    y_global: float,
    placement: TilePlacement,
    pixel_size_nm: float,
) -> tuple[float, float]:
    
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    x_local = float(x_global) / scale - float(placement.x0)
    y_local = float(y_global) / scale - float(placement.y0)
    return x_local, y_local


def local_px_to_global_nm(
    x_local: float,
    y_local: float,
    placement: TilePlacement,
    pixel_size_nm: float,
) -> tuple[float, float]:
    
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    x_global = (float(x_local) + float(placement.x0)) * scale
    y_global = (float(y_local) + float(placement.y0)) * scale
    return x_global, y_global


def crop_window(
    x_local: float,
    y_local: float,
    diameter_px: float,
    height: int,
    width: int,
    min_half: int = MIN_HALF_PX,
    scale: float = WINDOW_SCALE,
) -> tuple[int, int, int, int]:
    
    half = max(int(min_half), int(round(float(scale) * float(diameter_px))))
    cx = _clamp_index(int(round(float(x_local))), int(width))
    cy = _clamp_index(int(round(float(y_local))), int(height))
    x0, x1 = _local_span(cx, half, int(width))
    y0, y1 = _local_span(cy, half, int(height))
    return y0, x0, y1, x1


def _clamp_index(index: int, limit: int) -> int:
    if limit <= 0:
        return 0
    return min(max(int(index), 0), limit - 1)


def _local_span(center: int, half: int, limit: int) -> tuple[int, int]:
    
    if limit <= 0:
        return 0, 0
    start = max(0, int(center) - int(half))
    end = min(int(limit), int(center) + int(half))
    if end <= start:
        return 0, int(limit)
    target = min(int(limit), 2 * int(half))
    if end - start < target:
        if start == 0:
            end = min(int(limit), start + target)
        elif end == int(limit):
            start = max(0, end - target)
    return start, end


def grayscale_as_rgb(image: np.ndarray) -> np.ndarray:
    
    array = np.asarray(image)
    if array.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    if array.dtype == np.uint8:
        gray = np.ascontiguousarray(array)
    elif np.issubdtype(array.dtype, np.integer):
        shift = max(0, int(array.dtype.itemsize) * 8 - 8)
        gray = np.ascontiguousarray((array >> shift).astype(np.uint8))
    else:
        values = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0)
        peak = float(np.max(values)) if values.size else 0.0
        if peak <= 1.0:
            gray = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
        elif peak <= 255.0:
            gray = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        else:
            gray = np.clip(np.rint(values / 256.0), 0, 255).astype(np.uint8)
    if gray.ndim == 3:
        return np.ascontiguousarray(gray[..., :3])
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def draw_particle_circle(
    rgb: np.ndarray,
    x_local: float,
    y_local: float,
    size_nm: float,
    pixel_size_nm: float,
    crop_x0: int = 0,
    crop_y0: int = 0,
    color: tuple[int, int, int] = CIRCLE_COLOR,
) -> np.ndarray:
    
    x_d = int(round(float(x_local) - crop_x0))
    y_d = int(round(float(y_local) - crop_y0))
    return draw_marker(
        rgb,
        x_d,
        y_d,
        marker_radius(size_nm, pixel_size_nm),
        color=color,
    )


def config_for_hit_placement(
    config: dict[str, Any],
    rows: pd.DataFrame | list[dict[str, Any]],
) -> dict[str, Any]:
    
    records = _placement_records(rows)
    if not records:
        return config
    nominal = float(cfg_get(config, "overlap_fraction", 0.0) or 0.0)
    geometry = _tile_geometry(records, config)
    if not geometry:
        return config
    pixel_size = float(cfg_get(config, "pixel_size_nm", 960.0) or 960.0)
    nominal_hits = _hits_inside(records, geometry, nominal, pixel_size)
    if nominal_hits >= len(records):
        return config
    best_overlap = nominal
    best_hits = nominal_hits
    for step in range(0, 11):
        overlap = step / 20.0
        hits = _hits_inside(records, geometry, overlap, pixel_size)
        if hits > best_hits:
            best_overlap = overlap
            best_hits = hits
    if best_hits <= nominal_hits:
        return config
    updated = dict(config)
    updated["overlap_fraction"] = best_overlap
    return updated


def _placement_records(rows: pd.DataFrame | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return []
        records = rows.to_dict(orient="records")
    else:
        records = list(rows)
    if len(records) <= 80:
        return records
    step = max(1, len(records) // 80)
    return records[::step][:80]


def _tile_geometry(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, tuple[float, float, float, float]]:
    
    pattern = str(cfg_get(config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    input_dir = cfg_get(config, "input_dir", None)
    folders: dict[str, tuple[int, int]] = {}
    geometry: dict[str, tuple[float, float, float, float]] = {}
    for record in records:
        name = Path(str(record.get("source_tile", "") or "")).name
        if not name or name in geometry:
            continue
        try:
            path = find_tile_path(name, input_dir=input_dir)
        except (FileNotFoundError, OSError):
            continue
        meta = try_parse_tile_filename(path.name, pattern) or {}
        if "row" not in meta or "col" not in meta:
            continue
        folder = str(path.parent.resolve())
        if folder not in folders:
            paths = matching_tile_paths(path.parent, pattern)
            folders[folder] = grid_index_origin(paths, pattern)
        row_origin, col_origin = folders[folder]
        height, width = peek_tile_hw(path)
        geometry[name] = (
            float(int(meta["row"]) - row_origin),
            float(int(meta["col"]) - col_origin),
            float(height),
            float(width),
        )
    return geometry


def _hits_inside(
    records: list[dict[str, Any]],
    geometry: dict[str, tuple[float, float, float, float]],
    overlap: float,
    pixel_size_nm: float = 960.0,
) -> int:
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    inside = 0
    for record in records:
        name = Path(str(record.get("source_tile", "") or "")).name
        geom = geometry.get(name)
        if geom is None:
            continue
        row_off, col_off, height, width = geom
        try:
            x_px = float(record["x_global"]) / scale
            y_px = float(record["y_global"]) / scale
        except (KeyError, TypeError, ValueError):
            continue
        x_local = x_px - col_off * width * (1.0 - overlap)
        y_local = y_px - row_off * height * (1.0 - overlap)
        if 0.0 <= x_local < width and 0.0 <= y_local < height:
            inside += 1
    return inside


def find_tile_path(
    tile_name: str,
    input_dir: str | Path | None = None,
    mosaic: LazyMosaic | None = None,
) -> Path:
    
    raw = str(tile_name).replace("\\", "/")
    name = Path(raw).name
    relative = Path(raw)
    if mosaic is not None:
        for placement in mosaic.placements:
            if placement.name in (raw, name) and Path(placement.path).is_file():
                return Path(placement.path)
    for folder in _candidate_folders(input_dir):
        if relative.parent.name:
            found = _existing_tile(folder / relative)
            if found is not None:
                return found
        found = _existing_tile(folder / name)
        if found is not None:
            return found
        if relative.parent.name:
            nested = _existing_tile(folder / relative.parent.name / name)
            if nested is not None:
                return nested
    raise FileNotFoundError(f"Tile {raw!r} was not found in the tile folder.")


def _existing_tile(path: Path) -> Path | None:
    
    if path.is_file():
        return path
    seen = {path.suffix.lower()}
    for suffix in TILE_SUFFIXES:
        if suffix in seen:
            continue
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def placement_for_tile(
    tile_path: Path,
    config: dict[str, Any],
    mosaic: LazyMosaic | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> TilePlacement:
    
    name = tile_path.name
    if mosaic is not None:
        for placement in mosaic.placements:
            if placement.name == name:
                return placement
    pattern = str(cfg_get(config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    overlap = float(cfg_get(config, "overlap_fraction", 0.0))
    folder = tile_path.parent
    cache_key = str(folder.resolve())
    if origin_cache is not None and cache_key in origin_cache:
        row_origin, col_origin = origin_cache[cache_key]
    else:
        paths = matching_tile_paths(folder, pattern)
        row_origin, col_origin = grid_index_origin(paths, pattern)
        if origin_cache is not None:
            origin_cache[cache_key] = (row_origin, col_origin)
    extra_meta: dict[str, int] = {}
    parsed = positional_direction(tile_path)
    if parsed is not None:
        extra_meta["row"], extra_meta["col"] = parsed[0], parsed[1]
    placed = placement_from_path(
        tile_path,
        pattern,
        overlap,
        row_origin,
        col_origin,
        extra_meta=extra_meta or None,
        origin_fallback=True,
    )
    assert placed is not None
    return placed


_STAGED_2OF4 = re.compile(r"(v\d+)_2of4$", re.IGNORECASE)
_SET_DIR = re.compile(r"v\d+$", re.IGNORECASE)
_GROUNDUP_SERIES = re.compile(r"Groundup v(\d+)", re.IGNORECASE)


def direction_images_for_combined(tile_path: str | Path) -> dict[str, Path]:
    
    path = Path(tile_path)
    sides = positional_cell_sides(path)
    if len(sides) >= 2:
        return sides
    beside = list_folder_direction_images(path.parent)
    if len(beside) >= 2 and not is_direction_tile(path.name):
        return beside
    set_name = _set_name_for_combined(path)
    if not set_name:
        return {}
    seen: set[Path] = set()
    for folder in _direction_set_folders(path, set_name):
        try:
            key = folder.resolve()
        except OSError:
            key = folder
        if key in seen:
            continue
        seen.add(key)
        found = list_folder_direction_images(folder)
        if len(found) >= 2:
            return found
    return {}


def _direction_set_folders(tile_path: Path, set_name: str) -> list[Path]:
    
    match = _GROUNDUP_SERIES.search(tile_path.parent.name)
    series_names = [f"Groundup v{match.group(1)}"] if match else ["Groundup v4"]
    folders: list[Path] = []
    for series in series_names:
        for root in (tile_path.parent.parent / series, DEFAULT_INPUT_DIR / series):
            folders.extend(_set_folders_named(root, set_name))
    return folders


def _set_folders_named(root: Path, set_name: str) -> list[Path]:
    if not root.is_dir():
        return []
    exact = root / set_name
    if exact.is_dir():
        return [exact]
    pattern = re.compile(rf"^{re.escape(set_name)}(?:\s|\()", re.IGNORECASE)
    return [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and pattern.match(child.name)
    ]


def _set_name_for_combined(tile_path: Path) -> str | None:
    match = _STAGED_2OF4.fullmatch(tile_path.stem)
    if match:
        return match.group(1)
    if is_particles_only_tile(tile_path.name):
        for parent in tile_path.parents:
            if _SET_DIR.fullmatch(parent.name):
                return parent.name
    return None


def _spot_contrast(image: np.ndarray, x: float, y: float, diameter_px: float) -> float:
    
    gray = image if image.ndim == 2 else image.mean(axis=2)
    radius = max(2.0, float(diameter_px) / 2.0)
    outer = radius + max(4.0, 0.5 * radius)
    height, width = int(gray.shape[0]), int(gray.shape[1])
    x0 = max(0, int(np.floor(x - outer)))
    x1 = min(width, int(np.ceil(x + outer)) + 1)
    y0 = max(0, int(np.floor(y - outer)))
    y1 = min(height, int(np.ceil(y + outer)) + 1)
    if x1 <= x0 or y1 <= y0:
        return float("-inf")
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    patch = np.asarray(gray[y0:y1, x0:x1], dtype=np.float32)
    particle = dist2 <= radius ** 2
    background = (dist2 > radius ** 2) & (dist2 <= outer ** 2)
    if not np.any(particle) or not np.any(background):
        return float("-inf")
    return float(patch[particle].mean() - patch[background].mean())


def review_surface_name(source_tile: str) -> str:
    
    text = str(source_tile)
    if is_combi_tile(text) and submap_id(text):
        return "Combi"
    if lighting_role(text) == "all" and submap_id(text):
        return "Uniform"
    return ""


def _is_groundup_review_hit(source_tile: str) -> bool:
    
    text = str(source_tile)
    if not submap_id(text):
        return False
    return is_submap_lighting_tile(text) or is_combi_tile(text)


def list_combi_uniform_images(folder: str | Path) -> dict[str, Path]:
    
    directory = Path(folder)
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in TILE_SUFFIXES:
            continue
        name = review_surface_name(f"{directory.name}/{path.name}")
        if name and name not in found:
            found[name] = path
    return found


def brightest_combi_or_uniform_row(
    record: dict[str, Any],
    candidates: pd.DataFrame | None,
    *,
    merge_radius_nm: float,
    size_match_fraction: float,
) -> dict[str, Any] | None:
    
    if not _is_groundup_review_hit(str(record.get("source_tile") or "")):
        return None
    frames: list[pd.DataFrame] = []
    if (
        candidates is not None
        and not candidates.empty
        and {"source_tile", "x_global", "y_global"} <= set(candidates.columns)
    ):
        frames.append(candidates)
    if review_surface_name(str(record.get("source_tile") or "")):
        frames.append(pd.DataFrame([record]))
    if not frames:
        return None
    pool = pd.concat(frames, ignore_index=True)
    scene = submap_id(str(record.get("source_tile") or ""))
    scenes = pool["source_tile"].astype(str).map(submap_id)
    surfaces = pool["source_tile"].astype(str).map(review_surface_name)
    subset = pool.loc[(scenes == scene) & surfaces.ne("")]
    if subset.empty:
        return None
    x = float(record["x_global"])
    y = float(record["y_global"])
    size = float(record.get("size") or 0.0)
    dist = np.hypot(
        subset["x_global"].astype(float).to_numpy() - x,
        subset["y_global"].astype(float).to_numpy() - y,
    )
    other_size = (
        subset["size"].astype(float).to_numpy()
        if "size" in subset.columns
        else np.zeros(len(subset))
    )
    limit = np.full(len(subset), float(merge_radius_nm))
    if size_match_fraction > 0:
        limit = np.maximum(limit, float(size_match_fraction) * np.maximum(size, other_size))
    near = subset.loc[dist <= limit]
    if near.empty:
        return None
    confidence = (
        near["confidence"].astype(float).fillna(0.0)
        if "confidence" in near.columns
        else pd.Series(0.0, index=near.index)
    )
    return near.loc[confidence.idxmax()].to_dict()


def combi_or_uniform_crop(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
    candidates: pd.DataFrame | None = None,
    *,
    merge_radius_nm: float,
    size_match_fraction: float,
    mosaic: LazyMosaic | None = None,
    cache: TileImageCache | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> tuple[ParticleCrop, float] | None:
    
    record = dict(row) if not isinstance(row, dict) else dict(row)
    tile_name = str(record.get("source_tile", "") or "")
    if not _is_groundup_review_hit(tile_name):
        return None
    best = brightest_combi_or_uniform_row(
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
            config,
            mosaic=mosaic,
            cache=cache,
            origin_cache=origin_cache,
        )
        surface = review_surface_name(str(shown.get("source_tile") or ""))
        return (
            ParticleCrop(
                rgb=crop.rgb,
                x_local=crop.x_local,
                y_local=crop.y_local,
                crop_x0=crop.crop_x0,
                crop_y0=crop.crop_y0,
                tile_path=crop.tile_path,
                direction=surface,
            ),
            float(shown.get("confidence") or 0.0),
        )
    input_dir = cfg_get(config, "input_dir", None)
    try:
        tile_path = find_tile_path(tile_name, input_dir=input_dir, mosaic=mosaic)
    except (FileNotFoundError, OSError):
        return None
    surfaces = list_combi_uniform_images(tile_path.parent)
    if not surfaces:
        return None
    placement = placement_for_tile(
        tile_path, config, mosaic=mosaic, origin_cache=origin_cache
    )
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    x_local, y_local = global_nm_to_local_px(
        float(record["x_global"]),
        float(record["y_global"]),
        placement,
        pixel_size,
    )
    diameter_px = float(record["size"]) / (pixel_size if pixel_size > 0 else 1.0)
    loader = cache if cache is not None else TileImageCache()
    best_name = ""
    best_score = float("-inf")
    for name in ("Combi", "Uniform"):
        path = surfaces.get(name)
        if path is None:
            continue
        score = _spot_contrast(loader.load(path), x_local, y_local, diameter_px)
        if score > best_score:
            best_score = score
            best_name = name
    if not best_name:
        best_name = next(iter(surfaces))
    sibling = dict(record)
    sibling["source_tile"] = surfaces[best_name].name
    crop = crop_particle(
        sibling,
        config,
        mosaic=mosaic,
        cache=loader,
        origin_cache=origin_cache,
        tile_path=surfaces[best_name],
    )
    return (
        ParticleCrop(
            rgb=crop.rgb,
            x_local=crop.x_local,
            y_local=crop.y_local,
            crop_x0=crop.crop_x0,
            crop_y0=crop.crop_y0,
            tile_path=crop.tile_path,
            direction=best_name,
        ),
        float(record.get("confidence") or 0.0),
    )


def crop_particle(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
    mosaic: LazyMosaic | None = None,
    cache: TileImageCache | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
    tile_path: Path | None = None,
) -> ParticleCrop:
    
    record = dict(row) if not isinstance(row, dict) else dict(row)
    tile_name = str(record["source_tile"])
    input_dir = cfg_get(config, "input_dir", None)
    if tile_path is None:
        tile_path = find_tile_path(tile_name, input_dir=input_dir, mosaic=mosaic)
    else:
        tile_path = Path(tile_path)
    placement = placement_for_tile(
        tile_path, config, mosaic=mosaic, origin_cache=origin_cache
    )
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    x_local, y_local = global_nm_to_local_px(
        float(record["x_global"]),
        float(record["y_global"]),
        placement,
        pixel_size,
    )
    loader = cache if cache is not None else TileImageCache()
    image = loader.load(tile_path)
    height, width = int(image.shape[0]), int(image.shape[1])
    diameter_px = float(record["size"]) / (pixel_size if pixel_size > 0 else 1.0)
    y0, x0, y1, x1 = crop_window(x_local, y_local, diameter_px, height, width)
    patch = np.asarray(image[y0:y1, x0:x1])
    rgb = grayscale_as_rgb(patch)
    draw_particle_circle(
        rgb,
        x_local,
        y_local,
        float(record["size"]),
        pixel_size,
        crop_x0=x0,
        crop_y0=y0,
    )
    return ParticleCrop(
        rgb=rgb,
        x_local=x_local,
        y_local=y_local,
        crop_x0=x0,
        crop_y0=y0,
        tile_path=tile_path,
    )


def list_folder_direction_images(folder: str | Path) -> dict[str, Path]:
    
    directory = Path(folder)
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        if not path.is_file():
            continue
        stem = path.stem.upper()
        if stem not in DIRECTION_STEMS:
            stem = quad_direction(path.name) or ""
        if stem not in DIRECTION_STEMS or path.suffix.lower() not in TILE_SUFFIXES:
            continue
        found[stem] = path
    return found


def _combi_file_in_folder(folder: Path) -> Path | None:
    
    if not folder.is_dir():
        return None
    found = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in TILE_SUFFIXES
        and is_combi_tile(path.name)
    ]
    if not found:
        return None
    found.sort(key=lambda path: (0 if "2of4" in path.stem.lower() else 1, path.name.lower()))
    return found[0]


def nsew_combi_path(tile_path: str | Path) -> Path | None:
    
    path = Path(tile_path)
    if _is_groundup_review_hit(str(path)):
        return None
    if (
        is_combi_tile(path.name)
        or is_particles_only_tile(path.name)
        or _STAGED_2OF4.fullmatch(path.stem)
    ):
        return path
    parsed = positional_direction(path)
    if parsed is not None:
        row, col, _direction = parsed
        for suffix in TILE_SUFFIXES:
            candidate = path.parent / f"({int(row)},{int(col)}){suffix}"
            if candidate.is_file():
                return candidate
        return None
    if is_grid_tile(path.name) and positional_cell_sides(path):
        return path
    if is_direction_tile(path.name):
        return _combi_file_in_folder(path.parent)
    return None


def nsew_combi_crop(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
    mosaic: LazyMosaic | None = None,
    cache: TileImageCache | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> ParticleCrop | None:
    
    record = dict(row) if not isinstance(row, dict) else dict(row)
    tile_name = str(record.get("source_tile", "") or "")
    if not tile_name or _is_groundup_review_hit(tile_name):
        return None
    input_dir = cfg_get(config, "input_dir", None)
    try:
        tile_path = find_tile_path(tile_name, input_dir=input_dir, mosaic=mosaic)
    except (FileNotFoundError, OSError):
        return None
    combi_path = nsew_combi_path(tile_path)
    if combi_path is None:
        return None
    crop = crop_particle(
        record,
        config,
        mosaic=mosaic,
        cache=cache,
        origin_cache=origin_cache,
        tile_path=combi_path,
    )
    return ParticleCrop(
        rgb=crop.rgb,
        x_local=crop.x_local,
        y_local=crop.y_local,
        crop_x0=crop.crop_x0,
        crop_y0=crop.crop_y0,
        tile_path=crop.tile_path,
        direction="Combi",
    )


def _candidate_folders(input_dir: str | Path | None) -> list[Path]:
    if input_dir in (None, ""):
        return []
    try:
        return [resolve_input_dir(input_dir)]
    except FileNotFoundError:
        return []
