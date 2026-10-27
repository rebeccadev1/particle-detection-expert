from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src.overall.config import DEFAULT_INPUT_DIR

DIRECTION_STEMS = ("N", "S", "E", "W")
LIGHTING_ROLES = ("quad0", "quad1", "quad2", "quad3", "all")
QUAD_DIRECTIONS = ("N", "E", "S", "W")
_PARTICLES_ONLY_PREFIX = "particles_only"
_SET_DIR = re.compile(r"^v\d+$", re.IGNORECASE)
_GRID_TILE = re.compile(r"^\(\d+,\d+\)$")
_WAFER_LIGHT = re.compile(
    r"^wafer_shot_\d+_(quad[0-3]|all)_exp\d+$",
    re.IGNORECASE,
)


def direction_stem(source_tile: str) -> str | None:
    stem = Path(str(source_tile)).stem.upper()
    return stem if stem in DIRECTION_STEMS else None


def is_direction_tile(source_tile: str) -> bool:
    
    return direction_stem(source_tile) is not None


def is_particles_only_tile(source_tile: str) -> bool:
    
    stem = Path(str(source_tile)).stem.lower()
    return stem == _PARTICLES_ONLY_PREFIX or stem.startswith(f"{_PARTICLES_ONLY_PREFIX}_")


def is_nsew_family_tile(source_tile: str) -> bool:
    
    return is_direction_tile(source_tile) or is_particles_only_tile(source_tile)


def is_grid_tile(source_tile: str) -> bool:
    
    stem = Path(str(source_tile).replace("\\", "/")).stem
    return _GRID_TILE.fullmatch(stem) is not None


def is_combi_tile(source_tile: str) -> bool:
    
    stem = Path(str(source_tile).replace("\\", "/")).stem.lower()
    return stem == "combi" or stem.startswith("combi_")


def lighting_role(source_tile: str) -> str | None:
    
    match = _WAFER_LIGHT.fullmatch(Path(str(source_tile).replace("\\", "/")).stem)
    if match is None:
        return None
    return match.group(1).lower()


def quad_direction(source_tile: str) -> str | None:
    
    role = lighting_role(source_tile)
    if role is None or not role.startswith("quad"):
        return None
    index = int(role[-1])
    if index >= len(QUAD_DIRECTIONS):
        return None
    return QUAD_DIRECTIONS[index]


def is_submap_lighting_tile(source_tile: str) -> bool:
    
    return lighting_role(source_tile) is not None


def submap_id(source_tile: str) -> str:
    
    parent = Path(str(source_tile).replace("\\", "/")).parent.name
    if _SET_DIR.fullmatch(parent):
        return parent.lower()
    return ""


def lighting_tile_id(source_tile: str) -> str:
    
    text = str(source_tile).replace("\\", "/")
    name = Path(text).name
    if is_submap_lighting_tile(text) or is_combi_tile(text):
        set_id = submap_id(text)
        if set_id:
            return f"{set_id}/{name}"
    return name


def combi_source_name(path: str | Path) -> str:
    
    file_path = Path(path).resolve()
    try:
        return file_path.relative_to(DEFAULT_INPUT_DIR.resolve()).as_posix()
    except ValueError:
        return str(file_path)


def _resolve_source_file(source_tile: str) -> Path | None:
    raw = Path(str(source_tile).replace("\\", "/"))
    if raw.is_file():
        return raw
    candidate = DEFAULT_INPUT_DIR / raw
    if candidate.is_file():
        return candidate
    return None


@lru_cache(maxsize=256)
def combi_key_family(source_tile: str) -> str:
    
    if not is_combi_tile(source_tile):
        return ""
    resolved = _resolve_source_file(source_tile)
    if resolved is None:
        return ""
    has_light = False
    has_dir = False
    try:
        children = list(resolved.parent.iterdir())
    except OSError:
        return ""
    for child in children:
        if not child.is_file():
            continue
        if is_submap_lighting_tile(child.name):
            has_light = True
        elif is_direction_tile(child.name):
            has_dir = True
    if has_light:
        return "light"
    if has_dir:
        return "nsew"
    return ""


def input_relative_tile(path: str | Path) -> str:
    
    file_path = Path(path)
    try:
        resolved = file_path.resolve()
    except OSError:
        resolved = file_path
    try:
        return resolved.relative_to(DEFAULT_INPUT_DIR.resolve()).as_posix()
    except ValueError:
        return file_path.name


def lighting_source_name(path: str | Path) -> str:
    
    file_path = Path(path)
    if is_submap_lighting_tile(file_path.name) and _SET_DIR.fullmatch(file_path.parent.name):
        return f"{file_path.parent.name}/{file_path.name}"
    if is_grid_tile(file_path.name):
        return input_relative_tile(file_path)
    return file_path.name


def nsew_merge_stem(source_tile: str) -> str | None:
    
    if is_combi_tile(source_tile):
        return "COMBI"
    stem = direction_stem(source_tile)
    if stem is not None:
        return stem
    role = lighting_role(source_tile)
    if role is not None:
        return role.upper()
    if is_particles_only_tile(source_tile):
        return Path(str(source_tile)).stem.upper()
    return None
