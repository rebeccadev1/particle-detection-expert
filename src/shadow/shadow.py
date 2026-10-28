from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.overall.config import DEFAULT_INPUT_DIR, PACKAGE_ROOT, cfg_get, resolve_output_dir
from src.overall.io.tile_loader import TILE_SUFFIXES, load_tile_image
from src.verification.crops import (
    config_for_hit_placement,
    find_tile_path,
    global_nm_to_local_px,
    placement_for_tile,
)
from src.verification.inspect import candidate_placements
from src.verification.queue import detection_key
from src.verification.store import LabelStore, crop_source_folder
from src.analyse.report.report_generator import photo_display_rgb

PICK_MAX_SIDE = 960

DEFAULT_OUT = "Shadow check"
LIGHT_FROM_DEG = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
SHADOW_DIRECTIONS = ("N", "E", "S", "W")
OPPOSITE_DIRECTION = {"N": "S", "E": "W", "S": "N", "W": "E"}
DIRECTION_RGB = {
    "N": (255, 70, 70),
    "E": (255, 170, 40),
    "S": (80, 170, 255),
    "W": (186, 90, 255),
}
LIGHT_HEIGHT_MM = 5.0
RING_EDGE_MM = 6.6
RING_CENTRE_MM = 10.0
MAD_TO_SIGMA = 1.4826
PNG_SCALE = 4


@dataclass
class ShadowResult:
    found: bool
    particle_type: str
    area_px: float
    length_px: float
    width_px: float
    contrast: float
    background: float
    noise: float
    touches_window_edge: bool


@dataclass
class ShadowRun:
    table: pd.DataFrame
    images: list[tuple[str, np.ndarray]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    out_dir: Path | None = None


def shadow_direction(light_from_deg: float) -> tuple[float, float]:
    
    a = math.radians(float(light_from_deg))
    return -math.sin(a), math.cos(a)


def _component_at(mask: np.ndarray, py: int, px: int) -> np.ndarray:
    _n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if labels[py, px] == 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == labels[py, px]


def _segment_particle(
    patch: np.ndarray,
    py: int,
    px: int,
    r: float,
    along: np.ndarray,
    dist: np.ndarray,
    background: float,
    noise: float,
    k_mad: float,
    mass_fraction: float,
) -> tuple[str, np.ndarray]:
    
    seed_r = max(float(r), 2.0)
    seed = dist <= seed_r
    peak = float(patch[seed].max()) if np.any(seed) else float(patch[py, px])
    center = float(patch[seed].mean()) if np.any(seed) else float(patch[py, px])
    lo = float(patch[seed].min()) if np.any(seed) else center
    seed_along = float(along[py, px])
    along_ok = along <= seed_along + 1.25 * seed_r
    close = max(3, int(round(seed_r)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))

    if peak - background > k_mad * noise:
        particle_type = "bright"
        thresh = background + mass_fraction * (peak - background)
        mask = (patch >= thresh) & along_ok
        peak_idx = np.unravel_index(
            np.argmax(np.where(seed, patch, -np.inf)), patch.shape
        )
        sy, sx = int(peak_idx[0]), int(peak_idx[1])
    elif background - center > k_mad * noise:
        particle_type = "dark"
        thresh = background - mass_fraction * (background - lo)
        mask = (patch <= thresh) & along_ok
        sy, sx = py, px
    else:
        return "flat", seed

    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    particle = _component_at(closed.astype(bool), sy, sx)
    if not particle.any():
        particle = _component_at(mask, sy, sx)
    if np.count_nonzero(particle) < 4:
        particle = seed
    return particle_type, particle


def _mask_thickness_px(mask: np.ndarray) -> float:
    
    if not np.any(mask):
        return 1.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return max(2.0 * float(dist[mask].max()), 1.0)


def _crop_for_shadow(
    image: np.ndarray,
    x: float,
    y: float,
    diameter_px: float,
    light_from_deg: float,
    max_length_factor: float,
    particle_mask: np.ndarray | None = None,
    pad_all_sides: bool = False,
) -> tuple[np.ndarray, int, int, float, float]:
    
    height, width = image.shape
    r = max(0.5 * float(diameter_px), 1.0)
    shadow_reach = int(math.ceil(max_length_factor * 2.0 * r)) + 8
    along_half = int(math.ceil(r + shadow_reach))
    across_half = int(math.ceil(max(12.0 * r, float(along_half), 64.0)))
    dx, dy = shadow_direction(light_from_deg)
    half_x = max(int(round(abs(dx) * along_half + abs(dy) * across_half)), along_half)
    half_y = max(int(round(abs(dy) * along_half + abs(dx) * across_half)), along_half)
    if pad_all_sides:
        half = max(half_x, half_y)
        if particle_mask is not None and np.any(particle_mask):
            ys, xs = np.where(particle_mask)
            pad = max(shadow_reach, int(math.ceil(2.0 * r)), 16)
            y0 = max(int(ys.min()) - pad, 0)
            y1 = min(int(ys.max()) + pad + 1, height)
            x0 = max(int(xs.min()) - pad, 0)
            x1 = min(int(xs.max()) + pad + 1, width)
        else:
            ix, iy = int(round(x)), int(round(y))
            y0, y1 = max(iy - half, 0), min(iy + half + 1, height)
            x0, x1 = max(ix - half, 0), min(ix + half + 1, width)
        patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
        return patch, x0, y0, float(x) - x0, float(y) - y0
    if particle_mask is not None and np.any(particle_mask):
        ys, xs = np.where(particle_mask)
        pad_light = max(int(math.ceil(2.0 * r)), 16)
        pad_shadow = max(shadow_reach, 32)
        y0 = max(int(ys.min()) - (pad_shadow if dy < 0 else pad_light), 0)
        y1 = min(int(ys.max()) + (pad_shadow if dy > 0 else pad_light) + 1, height)
        x0 = max(int(xs.min()) - (pad_shadow if dx < 0 else pad_light), 0)
        x1 = min(int(xs.max()) + (pad_shadow if dx > 0 else pad_light) + 1, width)
        y0 = max(min(y0, int(ys.min()) - pad_light), 0)
        y1 = min(max(y1, int(ys.max()) + pad_light + 1), height)
        x0 = max(min(x0, int(xs.min()) - pad_light), 0)
        x1 = min(max(x1, int(xs.max()) + pad_light + 1), width)
    else:
        ix, iy = int(round(x)), int(round(y))
        y0, y1 = max(iy - half_y, 0), min(iy + half_y + 1, height)
        x0, x1 = max(ix - half_x, 0), min(ix + half_x + 1, width)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    return patch, x0, y0, float(x) - x0, float(y) - y0


def measure_shadow(
    image: np.ndarray,
    x: float,
    y: float,
    diameter_px: float,
    light_from_deg: float,
    *,
    k_mad: float = 3.0,
    max_length_factor: float = 3.0,
    mass_fraction: float = 0.4,
    min_area_px: int = 4,
    particle_mask: np.ndarray | None = None,
    half_plane: bool = False,
    pad_all_sides: bool = False,
) -> tuple[ShadowResult, dict[str, Any]]:
    
    height, width = image.shape
    r = max(0.5 * float(diameter_px), 1.0)
    dx, dy = shadow_direction(light_from_deg)
    full_mask = None
    if particle_mask is not None:
        full_mask = np.asarray(particle_mask, dtype=bool)
        if full_mask.shape != (height, width):
            raise ValueError(
                f"particle_mask shape {full_mask.shape} != image {(height, width)}"
            )

    patch, x0, y0, cx, cy = _crop_for_shadow(
        image, x, y, 2.0 * r, light_from_deg, max_length_factor, full_mask,
        pad_all_sides=pad_all_sides,
    )
    iy, ix = int(round(y)), int(round(x))
    py = min(max(iy - y0, 0), patch.shape[0] - 1)
    px = min(max(ix - x0, 0), patch.shape[1] - 1)

    yy, xx = np.indices(patch.shape, dtype=np.float64)
    along = (xx - cx) * dx + (yy - cy) * dy
    across = -(xx - cx) * dy + (yy - cy) * dx
    dist = np.hypot(xx - cx, yy - cy)

    corridor = (along > 0) & (np.abs(across) <= 1.5 * r)
    ring = (dist >= 1.5 * r) & ~corridor
    if full_mask is not None and full_mask.any():
        particle_preview = full_mask[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
        ring = ring & ~particle_preview
    bg_vals = patch[ring] if np.count_nonzero(ring) >= 8 else patch.ravel()
    background = float(np.median(bg_vals))
    noise = max(MAD_TO_SIGMA * float(np.median(np.abs(bg_vals - background))), 1e-6)

    if full_mask is not None and full_mask.any():
        particle = full_mask[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
        center = float(patch[particle].mean()) if particle.any() else float(patch[py, px])
        if center - background > k_mad * noise:
            particle_type = "bright"
        elif background - center > k_mad * noise:
            particle_type = "dark"
        else:
            particle_type = "flat"
    else:
        particle_type, particle = _segment_particle(
            patch, py, px, r, along, dist, background, noise, k_mad, mass_fraction
        )
    if particle.any():
        ring = ~particle & ~corridor
        if np.count_nonzero(ring) >= 8:
            bg_vals = patch[ring]
            background = float(np.median(bg_vals))
            noise = max(
                MAD_TO_SIGMA * float(np.median(np.abs(bg_vals - background))), 1e-6
            )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grown = cv2.dilate(particle.astype(np.uint8), kernel).astype(bool)
    touch = cv2.dilate(grown.astype(np.uint8), kernel).astype(bool)
    particle_along_max = float(along[particle].max()) if particle.any() else 0.0

    def _collect_shadow(threshold: float) -> np.ndarray:
        dark = (patch < background - threshold * noise) & ~particle
        if half_plane:
            limit = particle_along_max if particle.any() else 0.0
            dark = dark & (along > limit)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark.astype(np.uint8), connectivity=8
        )
        found = np.zeros_like(dark)
        for label in range(1, n_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_area_px:
                continue
            comp = labels == label
            if not np.any(comp & touch):
                continue
            if float(along[comp].mean()) <= particle_along_max:
                continue
            found |= comp
        return found

    shadow = _collect_shadow(float(k_mad))
    if not shadow.any() and full_mask is not None:
        shadow = _collect_shadow(max(float(k_mad) * 0.5, 0.25))

    length = 0.0
    width_px = 0.0
    mid_across = 0.0
    mid_edge_along = 0.0
    if particle.any():
        mid_across = float(across[particle].mean())
        mid_band = np.abs(across - mid_across) <= 0.75
        on_mid = particle & mid_band
        mid_edge_along = float(along[on_mid if on_mid.any() else particle].max())
    if shadow.any() and particle.any():
        mid_shadow = shadow & mid_band & (along > mid_edge_along)
        if mid_shadow.any():
            length = float(along[mid_shadow].max()) - mid_edge_along
        width_px = float(across[shadow].max() - across[shadow].min() + 1.0)
    debug = {
        "patch": patch,
        "particle": particle,
        "shadow": shadow,
        "cx": cx,
        "cy": cy,
        "x0": x0,
        "y0": y0,
        "dir": (dx, dy),
        "length_px": length,
        "along": along,
        "across": across,
        "mid_across": mid_across,
        "mid_edge_along": mid_edge_along,
    }
    if not shadow.any():
        return ShadowResult(False, particle_type, 0.0, 0.0, 0.0, 0.0, background, noise,
                            False), debug

    contrast = (background - float(patch[shadow].mean())) / max(abs(background), 1e-6)
    edge = bool(shadow[0, :].any() or shadow[-1, :].any()
                or shadow[:, 0].any() or shadow[:, -1].any())
    return ShadowResult(
        True, particle_type, float(np.count_nonzero(shadow)), length, width_px,
        float(contrast), background, noise, edge,
    ), debug


def measure_nsew_shadows(
    image: np.ndarray,
    x: float,
    y: float,
    diameter_px: float,
    **kwargs: Any,
) -> tuple[dict[str, ShadowResult], dict[str, Any]]:
    
    measured: dict[str, tuple[ShadowResult, dict[str, Any]]] = {}
    for direction in SHADOW_DIRECTIONS:
        light_from = LIGHT_FROM_DEG[OPPOSITE_DIRECTION[direction]]
        result, debug = measure_shadow(
            image, x, y, diameter_px, light_from,
            half_plane=True, pad_all_sides=True, **kwargs,
        )
        measured[direction] = (result, debug)
    results = {name: pair[0] for name, pair in measured.items()}
    return results, _composite_nsew_debug(measured)


def _composite_nsew_debug(
    measured: dict[str, tuple[ShadowResult, dict[str, Any]]],
) -> dict[str, Any]:
    
    x0 = min(int(debug["x0"]) for _result, debug in measured.values())
    y0 = min(int(debug["y0"]) for _result, debug in measured.values())
    x1 = max(int(debug["x0"]) + int(debug["patch"].shape[1]) for _result, debug in measured.values())
    y1 = max(int(debug["y0"]) + int(debug["patch"].shape[0]) for _result, debug in measured.values())
    patch = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1)), dtype=np.float64)
    particle = np.zeros(patch.shape, dtype=bool)
    shadows: dict[str, np.ndarray] = {}
    directions: dict[str, dict[str, Any]] = {}
    cx = cy = 0.0
    for name, (_result, debug) in measured.items():
        py = int(debug["y0"]) - y0
        px = int(debug["x0"]) - x0
        sub = np.asarray(debug["patch"], dtype=np.float64)
        ph, pw = sub.shape
        patch[py : py + ph, px : px + pw] = sub
        particle[py : py + ph, px : px + pw] |= np.asarray(debug["particle"], dtype=bool)
        mask = np.zeros(patch.shape, dtype=bool)
        mask[py : py + ph, px : px + pw] = np.asarray(debug["shadow"], dtype=bool)
        shadows[name] = mask
        directions[name] = {
            "dir": debug["dir"],
            "length_px": float(debug["length_px"]),
            "mid_across": float(debug["mid_across"]),
            "mid_edge_along": float(debug["mid_edge_along"]),
            "color": DIRECTION_RGB[name],
        }
        cx = float(debug["cx"]) + int(debug["x0"]) - x0
        cy = float(debug["cy"]) + int(debug["y0"]) - y0
    union = np.zeros(patch.shape, dtype=bool)
    for mask in shadows.values():
        union |= mask
    return {
        "patch": patch,
        "particle": particle,
        "shadow": union,
        "shadows": shadows,
        "cx": cx,
        "cy": cy,
        "dir": (0.0, 1.0),
        "length_px": 0.0,
        "directions": directions,
        "x0": x0,
        "y0": y0,
    }


def _direction_label(result: ShadowResult, name: str, um: float) -> str:
    if result.found:
        return f"{name}  {result.length_px:.0f} px = {result.length_px * um:.1f} um"
    return f"{name}  none"


def nsew_shadow_title(results: dict[str, ShadowResult], um: float) -> str:
    
    left = (_direction_label(results["N"], "N", um), _direction_label(results["E"], "E", um))
    right = (_direction_label(results["S"], "S", um), _direction_label(results["W"], "W", um))
    return "\n".join(f"{a}\t{b}" for a, b in zip(left, right))


def _wrap_cv2_text(text: str, max_w: int, font: int, scale: float, thickness: int) -> list[str]:
    
    lines: list[str] = []
    line = ""
    for ch in text:
        if line and cv2.getTextSize(line + ch, font, scale, thickness)[0][0] > max_w:
            cut = max(line.rfind("/"), line.rfind(" "))
            if cut > 0:
                lines.append(line[: cut + 1])
                line = line[cut + 1 :]
            else:
                lines.append(line)
                line = ""
        line += ch
    if line:
        lines.append(line)
    return lines


def _footer_band(source: list[str], width: int) -> np.ndarray:
    
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    lines = [w for text in source for w in _wrap_cv2_text(text, width - 16, font, scale, thickness)]
    line_h = cv2.getTextSize("Ag", font, scale, thickness)[0][1] + 12
    band = np.full((line_h * len(lines) + 12, width, 3), 255, dtype=np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(band, text, (8, line_h * (i + 1)), font, scale, (40, 40, 40), thickness,
                    lineType=cv2.LINE_AA)
    return band


def _source_lines(path: Path) -> list[str]:
    
    if path.is_relative_to(DEFAULT_INPUT_DIR):
        folder = Path(DEFAULT_INPUT_DIR.name) / path.parent.relative_to(DEFAULT_INPUT_DIR)
    else:
        folder = path.parent
    return [f"{folder.as_posix()}/", path.name]


def image_choice_labels() -> tuple[str, ...]:
    
    labels: list[str] = []
    for paths in _input_images().values():
        for path in paths:
            if path.is_relative_to(DEFAULT_INPUT_DIR):
                labels.append(
                    (Path(DEFAULT_INPUT_DIR.name) / path.relative_to(DEFAULT_INPUT_DIR)).as_posix()
                )
            else:
                labels.append(path.as_posix())
    return tuple(sorted(labels))


def resolve_named_image(text: str) -> tuple[Path | None, tuple[Path, ...]]:
    
    raw = text.strip().strip('"').strip("'")
    if not raw:
        return None, ()
    direct = Path(raw).expanduser()
    if direct.is_file():
        return direct, ()
    relative = raw.replace("\\", "/").lstrip("/")
    prefix = DEFAULT_INPUT_DIR.name + "/"
    if relative.startswith(prefix):
        relative = relative[len(prefix):]
    under = DEFAULT_INPUT_DIR / relative
    if under.is_file():
        return under, ()
    hits: list[Path] = []
    for paths in _input_images().values():
        for path in paths:
            rel = (
                path.relative_to(DEFAULT_INPUT_DIR).as_posix()
                if path.is_relative_to(DEFAULT_INPUT_DIR) else path.as_posix()
            )
            if path.name == relative or rel == relative or rel.endswith("/" + relative):
                hits.append(path)
    found = tuple(sorted(set(hits)))
    if len(found) == 1:
        return found[0], ()
    return None, found


def _draw_length_arrow(
    big: np.ndarray,
    cx: float,
    cy: float,
    info: dict[str, Any],
    particle: np.ndarray,
) -> None:
    
    length_px = float(info.get("length_px") or 0.0)
    if length_px <= 0 or not particle.any():
        return
    dx, dy = info["dir"]
    mid_across = float(info["mid_across"])
    a_start = float(info["mid_edge_along"]) + 0.5
    x0 = cx + a_start * dx - mid_across * dy + 0.5
    y0 = cy + a_start * dy + mid_across * dx + 0.5
    x1 = x0 + dx * length_px
    y1 = y0 + dy * length_px
    p0 = (int(round(x0 * PNG_SCALE)), int(round(y0 * PNG_SCALE)))
    p1 = (int(round(x1 * PNG_SCALE)), int(round(y1 * PNG_SCALE)))
    color = tuple(int(c) for c in info.get("color", (255, 0, 0)))
    cv2.arrowedLine(big, p0, p1, color, 3, tipLength=0.12, line_type=cv2.LINE_AA)


def _title_band(title: str, width: int) -> np.ndarray:
    
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1
    scale = 0.45
    rows = [row.split("\t") for row in (title.split("\n") if title else [""])]
    two_cols = any(len(cells) > 1 for cells in rows)
    gap = 18

    def _measure(text_scale: float) -> tuple[int, int, int, list[int]]:
        text_h = baseline = 0
        col_w = [0, 0]
        for cells in rows:
            for index, cell in enumerate(cells[:2]):
                (cell_w, cell_h), cell_base = cv2.getTextSize(cell, font, text_scale, thickness)
                col_w[index] = max(col_w[index], cell_w)
                text_h = max(text_h, cell_h)
                baseline = max(baseline, cell_base)
        total = col_w[0] + (gap if two_cols else 0) + col_w[1]
        return total, text_h, baseline, col_w

    total, text_h, baseline, col_w = _measure(scale)
    if total > width - 8 and total > 0:
        scale *= (width - 8) / total
        total, text_h, baseline, col_w = _measure(scale)
    line_h = text_h + baseline + 4
    band = np.full((line_h * len(rows) + 4, width, 3), 255, dtype=np.uint8)
    x_right = 4 + col_w[0] + gap
    for i, cells in enumerate(rows):
        y = line_h * i + text_h + 2
        for index, cell in enumerate(cells[:2]):
            color = DIRECTION_RGB.get(cell[:1], (255, 0, 0))
            x = 4 if index == 0 else x_right
            cv2.putText(band, cell, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)
    return band


def debug_image(
    debug: dict[str, Any], title: str, source: list[str] | None = None
) -> np.ndarray:
    
    patch = debug["patch"]
    particle = np.asarray(debug["particle"], dtype=bool)
    cx, cy = float(debug["cx"]), float(debug["cy"])
    rgb = photo_display_rgb(patch)
    tint = rgb.copy()
    named = debug.get("shadows")
    if isinstance(named, dict):
        for name, mask in named.items():
            tint[np.asarray(mask, dtype=bool)] = DIRECTION_RGB.get(str(name), (255, 0, 0))
    else:
        tint[np.asarray(debug["shadow"], dtype=bool)] = (255, 0, 0)
    rgb = cv2.addWeighted(tint, 0.45, rgb, 0.55, 0)
    big = cv2.resize(rgb, None, fx=PNG_SCALE, fy=PNG_SCALE, interpolation=cv2.INTER_NEAREST)
    outline = cv2.resize(particle.astype(np.uint8), None, fx=PNG_SCALE,
                         fy=PNG_SCALE, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(big, contours, -1, (0, 255, 0), 2)
    direction_info = debug.get("directions")
    if isinstance(direction_info, dict):
        for info in direction_info.values():
            _draw_length_arrow(big, cx, cy, info, particle)
    else:
        dx, dy = debug["dir"]
        light_len = 0.18 * min(big.shape[:2])
        light_origin = (14, 40)
        cv2.arrowedLine(
            big,
            light_origin,
            (int(round(light_origin[0] + dx * light_len)),
             int(round(light_origin[1] + dy * light_len))),
            (255, 255, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA,
        )
        _draw_length_arrow(
            big, cx, cy,
            {
                "dir": (dx, dy),
                "length_px": float(debug.get("length_px") or 0.0),
                "mid_across": float(debug["mid_across"]),
                "mid_edge_along": float(debug["mid_edge_along"]),
                "color": (255, 0, 0),
            },
            particle,
        )
    band = _title_band(title, big.shape[1])
    parts = [band, big]
    if source:
        parts.append(_footer_band(source, big.shape[1]))
    return np.vstack(parts)


def light_distance_mm(
    x: float,
    y: float,
    width: int,
    height: int,
    light_from_deg: float,
    edge_mm: float = RING_EDGE_MM,
    centre_mm: float = RING_CENTRE_MM,
) -> float:
    
    dx, dy = shadow_direction(light_from_deg)
    ux, uy = -dx, -dy
    toward = (x - 0.5 * (width - 1)) * ux + (y - 0.5 * (height - 1)) * uy
    half = abs(ux) * 0.5 * width + abs(uy) * 0.5 * height
    return float(centre_mm - (centre_mm - edge_mm) * toward / max(half, 1e-9))


@lru_cache(maxsize=1)
def _input_images() -> dict[str, tuple[Path, ...]]:
    found: dict[str, list[Path]] = {}
    for path in DEFAULT_INPUT_DIR.rglob("*"):
        if path.suffix.lower() in TILE_SUFFIXES and path.is_file():
            found.setdefault(path.name, []).append(path)
    return {name: tuple(sorted(paths)) for name, paths in found.items()}


def _dir_matches(folder: str, part: str) -> bool:
    folder, part = folder.lower(), part.lower()
    return folder == part or folder.startswith(part + " ")


def _shared_leading_words(a: str, b: str) -> int:
    count = 0
    for left, right in zip(a.lower().split(), b.lower().split()):
        if left != right:
            break
        count += 1
    return count


def find_label_image(source_tile: str, source_csv: object = "") -> tuple[Path | None, int]:
    
    parts = [p for p in str(source_tile).replace("\\", "/").split("/") if p]
    if not parts:
        return None, 0
    name, dirs = parts[-1], parts[:-1]
    matches = [
        path for path in _input_images().get(name, ())
        if len(path.parents) > len(dirs)
        and all(_dir_matches(path.parents[i].name, part)
                for i, part in enumerate(reversed(dirs)))
    ]
    if len(matches) <= 1:
        return (matches[0] if matches else None), len(matches)
    run = crop_source_folder(source_csv, source_tile)

    def score(path: Path) -> tuple[int, int]:
        top = path.relative_to(DEFAULT_INPUT_DIR).parts[0]
        siblings = sum(1 for p in path.parent.iterdir() if p.suffix.lower() in TILE_SUFFIXES)
        return _shared_leading_words(run, top), siblings

    ranked = sorted(matches, key=score, reverse=True)
    best = score(ranked[0])
    return ranked[0], sum(1 for path in ranked if score(path) == best)


def id_column(table: pd.DataFrame) -> str:
    return "particle_id" if "particle_id" in table.columns else "id"


def select_rows(
    table: pd.DataFrame,
    ids: list[float],
    keys: list[str],
    run: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    
    id_col = id_column(table)
    messages: list[str] = []
    if run and "source_csv" in table.columns:
        table = table[table["source_csv"].astype(str).str.contains(run, regex=False)]
    picked: list[pd.DataFrame] = []
    if keys:
        if "key" not in table.columns:
            table = table.assign(key=table.apply(detection_key, axis=1))
        by_key = table[table["key"].astype(str).isin(keys)].drop_duplicates("key")
        picked.append(by_key)
        found = set(by_key["key"].astype(str))
        messages.extend(f"Key {key}: not found." for key in keys if key not in found)
    for pid in ids:
        rows = table[pd.to_numeric(table[id_col], errors="coerce") == float(pid)]
        if rows.empty:
            messages.append(f"ID {pid:g}: not found.")
        elif len(rows) > 1:
            cols = [c for c in ("key", "source_tile", "label", "source_csv") if c in rows]
            messages.append(
                f"ID {pid:g}: {len(rows)} matches, narrow with the run filter or use the key:\n"
                + rows[cols].to_string(index=False)
            )
        else:
            picked.append(rows)
    if not picked:
        return table.iloc[0:0], messages
    return pd.concat(picked).drop_duplicates(), messages


def _edge_scales(
    x_nm: float,
    y_nm: float,
    border: float,
    width: int,
    height: int,
    origin_x: float,
    origin_y: float,
) -> list[float]:
    
    scales: list[float] = []
    for numerator, denominator in (
        (x_nm, border + origin_x),
        (y_nm, border + origin_y),
        (x_nm, (float(width) - 1.0 - border) + origin_x),
        (y_nm, (float(height) - 1.0 - border) + origin_y),
    ):
        if denominator <= 1e-6:
            continue
        scale = numerator / denominator
        if np.isfinite(scale) and scale > 0:
            scales.append(float(scale))
    return scales


def scale_from_borders(
    hits: list[tuple[float, float, float]],
    width: int,
    height: int,
    x0: float = 0.0,
    y0: float = 0.0,
) -> float | None:
    
    usable = [
        (float(x_nm), float(y_nm), float(border))
        for x_nm, y_nm, border in hits
        if np.isfinite(border) and border >= 0
    ]
    if not usable:
        return None
    origin_x, origin_y = float(x0), float(y0)
    candidates: list[float] = []
    for x_nm, y_nm, border in usable:
        candidates.extend(
            _edge_scales(x_nm, y_nm, border, width, height, origin_x, origin_y)
        )
    best: tuple[float, float] | None = None
    for scale in candidates:
        errors: list[float] = []
        for x_nm, y_nm, border in usable:
            x = x_nm / scale - origin_x
            y = y_nm / scale - origin_y
            if not (0.0 <= x < float(width) and 0.0 <= y < float(height)):
                errors = []
                break
            predicted = min(x, y, float(width) - 1.0 - x, float(height) - 1.0 - y)
            errors.append(abs(predicted - border))
        if not errors or max(errors) > 1.0:
            continue
        error = float(np.median(errors))
        if best is None or error < best[0]:
            best = (error, float(scale))
    return None if best is None else best[1]


def detection_borders(
    source_csv: object,
    source_tile: object,
    cache: dict[str, pd.DataFrame | None],
) -> list[tuple[float, float, float]]:
    
    path = Path(str(source_csv or ""))
    key = str(path)
    if key not in cache:
        table: pd.DataFrame | None = None
        if path.is_file():
            try:
                loaded = pd.read_csv(path)
            except (OSError, ValueError, pd.errors.ParserError):
                loaded = None
            if (
                loaded is not None
                and {"x_global", "y_global", "tile_border_dist_px"}.issubset(loaded.columns)
            ):
                table = loaded
        cache[key] = table
    table = cache[key]
    if table is None or table.empty:
        return []
    rows = table
    if "source_tile" in table.columns:
        wanted = Path(str(source_tile or "")).name
        named = table["source_tile"].astype(str).map(lambda name: Path(name).name)
        same = table.loc[named == wanted]
        if not same.empty:
            rows = same
    xs = pd.to_numeric(rows["x_global"], errors="coerce")
    ys = pd.to_numeric(rows["y_global"], errors="coerce")
    borders = pd.to_numeric(rows["tile_border_dist_px"], errors="coerce")
    hits: list[tuple[float, float, float]] = []
    for x_nm, y_nm, border in zip(xs, ys, borders):
        if pd.notna(x_nm) and pd.notna(y_nm) and pd.notna(border):
            hits.append((float(x_nm), float(y_nm), float(border)))
    return hits


def _shadow_size(
    length_px: float,
    found: bool,
    um: float,
    distance_mm: float,
    light_height_mm: float,
) -> tuple[float, float, float]:
    length_um = float(length_px) * um
    elevation_deg = math.degrees(math.atan2(light_height_mm, distance_mm))
    height_um = (
        length_um * light_height_mm / distance_mm
        if found and distance_mm > 0
        else float("nan")
    )
    return length_um, elevation_deg, height_um


def _shadow_table_rows(
    results: dict[str, ShadowResult],
    *,
    key: str,
    pid: object,
    label: object,
    folder: str,
    tile: str,
    x: float,
    y: float,
    width: int,
    height: int,
    particle_um: float,
    um: float,
    light_height_mm: float,
    ring_edge_mm: float,
    ring_centre_mm: float,
) -> list[dict[str, Any]]:
    
    rows: list[dict[str, Any]] = []
    for direction in SHADOW_DIRECTIONS:
        shadow = results[direction]
        light_deg = LIGHT_FROM_DEG[OPPOSITE_DIRECTION[direction]]
        distance_mm = light_distance_mm(
            x, y, width, height, light_deg, ring_edge_mm, ring_centre_mm
        )
        length_um, elevation_deg, height_um = _shadow_size(
            shadow.length_px, shadow.found, um, distance_mm, light_height_mm
        )
        rows.append({
            "key": key,
            "id": pid,
            "direction": direction,
            "label": label,
            "folder": folder,
            "tile": tile,
            "x_px": round(x, 1),
            "y_px": round(y, 1),
            "light_distance_mm": round(distance_mm, 2),
            "elevation_deg": round(elevation_deg, 1),
            "particle_um": round(particle_um, 2),
            "shadow_length_um": round(length_um, 2),
            "shadow_height_um": round(height_um, 2),
            **{
                k: v for k, v in asdict(shadow).items()
                if k in ("found", "particle_type", "contrast", "touches_window_edge")
            },
        })
    return rows


def _write_shadow_run(result: ShadowRun, out: str | Path | None) -> None:
    if not out:
        return
    out_dir = resolve_output_dir(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out_dir / "shadow_check.csv", index=False)
    for name, rgb in result.images:
        cv2.imwrite(str(out_dir / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    result.out_dir = out_dir


def run_shadow_check(
    table: pd.DataFrame,
    config: dict[str, Any],
    *,
    ids: list[float] | None = None,
    keys: list[str] | None = None,
    run: str | None = None,
    input_dir: str | None = None,
    tile: Path | None = None,
    light_height_mm: float = LIGHT_HEIGHT_MM,
    ring_edge_mm: float = RING_EDGE_MM,
    ring_centre_mm: float = RING_CENTRE_MM,
    k_mad: float = 3.0,
    max_length_factor: float = 3.0,
    out: str | Path | None = DEFAULT_OUT,
) -> ShadowRun:
    
    config = deepcopy(config)
    pixel_nm = float(cfg_get(config, "pixel_size_nm", 1000.0))
    rows, messages = select_rows(table, list(ids or []), list(keys or []), run)
    result = ShadowRun(table=pd.DataFrame(), messages=messages)
    if rows.empty:
        return result
    id_col = id_column(rows)
    records: list[dict[str, Any]] = []
    cache: dict[Path, np.ndarray] = {}
    run_configs: dict[str, dict[str, Any]] = {}
    border_cache: dict[str, pd.DataFrame | None] = {}

    for _, row in rows.iterrows():
        pid = row[id_col]
        key = row.get("key")
        name = str(key) if pd.notna(key) and str(key) else f"id {float(pid):g}"
        tile_name = str(row["source_tile"])
        source_csv = row.get("source_csv", "")
        if input_dir:
            try:
                label_path = find_tile_path(tile_name, input_dir=input_dir)
            except FileNotFoundError as exc:
                messages.append(f"{name}: {exc}")
                continue
        else:
            label_path, ties = find_label_image(tile_name, source_csv)
            if label_path is None:
                messages.append(f"{name}: no image named {tile_name} under {DEFAULT_INPUT_DIR}.")
                continue
            if ties > 1:
                messages.append(
                    f"{name}: {ties} folders contain {Path(tile_name).name}; used "
                    f"{label_path.parent.relative_to(DEFAULT_INPUT_DIR)}."
                )
        path = Path(tile) if tile else label_path
        run_key = (str(source_csv), str(label_path.parent))
        if run_key not in run_configs:
            run_config = dict(config, input_dir=str(label_path.parent))
            same_run = (table[table["source_csv"].astype(str) == run_key[0]]
                        if "source_csv" in table.columns else table)
            run_configs[run_key] = config_for_hit_placement(run_config, same_run)
        if path not in cache:
            cache[path] = load_tile_image(path)
        image = cache[path]
        placements = [placement_for_tile(label_path, run_configs[run_key])]
        placements += candidate_placements(label_path, run_configs[run_key])
        scale_nm = pixel_nm
        x = y = 0.0
        landed = False
        for placement in placements:
            x, y = global_nm_to_local_px(float(row["x_global"]), float(row["y_global"]),
                                         placement, scale_nm)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                landed = True
                break
        if not landed:
            hits = detection_borders(source_csv, tile_name, border_cache)
            for placement in placements:
                recovered = scale_from_borders(
                    hits, image.shape[1], image.shape[0], placement.x0, placement.y0
                )
                if recovered is None:
                    continue
                x, y = global_nm_to_local_px(
                    float(row["x_global"]), float(row["y_global"]), placement, recovered
                )
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    scale_nm = recovered
                    landed = True
                    break
        if not landed:
            x, y = global_nm_to_local_px(float(row["x_global"]), float(row["y_global"]),
                                         placements[0], pixel_nm)
            messages.append(f"{name}: ({x:.0f}, {y:.0f}) px is outside {path.name}.")
            continue
        um = scale_nm / 1000.0
        diameter_px = float(row["size"]) / scale_nm
        shadows, debug = measure_nsew_shadows(
            image, x, y, diameter_px, k_mad=k_mad, max_length_factor=max_length_factor
        )
        folder = (
            str(label_path.parent.relative_to(DEFAULT_INPUT_DIR))
            if label_path.is_relative_to(DEFAULT_INPUT_DIR) else str(label_path.parent)
        )
        records.extend(_shadow_table_rows(
            shadows,
            key=name,
            pid=pid,
            label=row.get("label", ""),
            folder=folder,
            tile=path.name,
            x=x,
            y=y,
            width=image.shape[1],
            height=image.shape[0],
            particle_um=diameter_px * um,
            um=um,
            light_height_mm=light_height_mm,
            ring_edge_mm=ring_edge_mm,
            ring_centre_mm=ring_centre_mm,
        ))
        result.images.append(
            (name, debug_image(debug, nsew_shadow_title(shadows, um), _source_lines(path)))
        )

    if not records:
        return result
    result.table = pd.DataFrame(records)
    _write_shadow_run(result, out)
    return result


def _parse_keys(text: str) -> list[str]:
    keys: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(depth - 1, 0)
            current.append(ch)
        elif depth == 0 and (ch == "," or ch.isspace()):
            token = "".join(current).strip()
            if token:
                keys.append(token)
            current = []
        else:
            current.append(ch)
    token = "".join(current).strip()
    if token:
        keys.append(token)
    return keys


def display_preview(
    image: np.ndarray,
    max_side: int = PICK_MAX_SIDE,
) -> tuple[np.ndarray, float]:
    
    rgb = photo_display_rgb(image)
    height, width = rgb.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width, 1)))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return rgb, scale


def zoom_preview(
    image: np.ndarray,
    cx: float,
    cy: float,
    zoom: float,
    max_side: int = PICK_MAX_SIDE,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    
    rgb = photo_display_rgb(image)
    height, width = rgb.shape[:2]
    zoom = max(float(zoom), 1.0)
    if zoom <= 1.0 + 1e-9:
        preview, _ = display_preview(image, max_side=max_side)
        return preview, (0, 0, height, width)
    src_side = max(int(round(max(height, width) / zoom)), 32)
    half = src_side // 2
    cx_i, cy_i = int(round(cx)), int(round(cy))
    x0 = int(min(max(cx_i - half, 0), max(width - src_side, 0)))
    y0 = int(min(max(cy_i - half, 0), max(height - src_side, 0)))
    x1 = min(x0 + src_side, width)
    y1 = min(y0 + src_side, height)
    crop = rgb[y0:y1, x0:x1]
    out_scale = float(max_side) / float(max(crop.shape[0], crop.shape[1], 1))
    out_w = max(1, int(round(crop.shape[1] * out_scale)))
    out_h = max(1, int(round(crop.shape[0] * out_scale)))
    preview = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return preview, (y0, x0, y1, x1)


def paint_kept_mask(
    preview: np.ndarray,
    mask: np.ndarray,
    crop: tuple[int, int, int, int],
) -> np.ndarray:
    
    y0, x0, y1, x1 = crop
    sub = np.asarray(mask, dtype=bool)[y0:y1, x0:x1]
    if sub.shape != preview.shape[:2]:
        sub = cv2.resize(
            sub.astype(np.uint8),
            (preview.shape[1], preview.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    out = preview.copy()
    if sub.any():
        tint = np.array((40, 180, 70), dtype=np.float32)
        out[sub] = (0.45 * out[sub].astype(np.float32) + 0.55 * tint).astype(np.uint8)
    return out


def _canvas_drawing_mask(arr: np.ndarray) -> np.ndarray:
    
    array = np.asarray(arr)
    if array.ndim == 2:
        plane = array.astype(np.float32)
        if plane.max() <= 1.5:
            return plane > 0.04
        return plane > 10.0
    rgb = array[..., :3].astype(np.float32)
    if array.shape[2] >= 4:
        alpha = array[..., 3].astype(np.float32)
    else:
        alpha = np.ones(array.shape[:2], dtype=np.float32)
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    if alpha.max() <= 1.5:
        alpha_on = alpha > 0.04
    else:
        alpha_on = alpha > 10.0
    green = (
        alpha_on
        & (rgb[..., 1] >= rgb[..., 0] + 15.0)
        & (rgb[..., 1] >= rgb[..., 2] + 15.0)
    )
    if green.any():
        return green
    if alpha_on.any() and float(alpha_on.mean()) < 0.55:
        return alpha_on
    ink = alpha_on & (rgb.max(axis=-1) > 8.0)
    if ink.any() and float(ink.mean()) < 0.55:
        return ink
    return np.zeros(array.shape[:2], dtype=bool)


def mask_from_canvas(
    image_data: np.ndarray | None,
    image_shape: tuple[int, int],
    *,
    crop: tuple[int, int, int, int] | None = None,
    close_px: int = 7,
) -> np.ndarray:
    
    height, width = int(image_shape[0]), int(image_shape[1])
    empty = np.zeros((height, width), dtype=bool)
    if image_data is None:
        return empty
    arr = np.asarray(image_data)
    if arr.size == 0:
        return empty
    drawn = _canvas_drawing_mask(arr)
    if not drawn.any():
        return empty
    if crop is None:
        y0, x0, y1, x1 = 0, 0, height, width
    else:
        y0, x0, y1, x1 = (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))
        y0, x0 = max(y0, 0), max(x0, 0)
        y1, x1 = min(y1, height), min(x1, width)
    region_h, region_w = max(y1 - y0, 1), max(x1 - x0, 1)
    scaled = cv2.resize(
        drawn.astype(np.uint8), (region_w, region_h), interpolation=cv2.INTER_NEAREST
    )
    k = max(3, int(close_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    else:
        filled = closed
    empty[y0:y1, x0:x1] = filled.astype(bool)
    return empty


def run_shadow_on_mask(
    image: np.ndarray,
    name: str,
    particle_mask: np.ndarray,
    config: dict[str, Any],
    *,
    source_path: Path | None = None,
    light_height_mm: float = LIGHT_HEIGHT_MM,
    ring_edge_mm: float = RING_EDGE_MM,
    ring_centre_mm: float = RING_CENTRE_MM,
    k_mad: float = 3.0,
    max_length_factor: float = 3.0,
    out: str | Path | None = DEFAULT_OUT,
) -> ShadowRun:
    
    pixel_nm = float(cfg_get(config, "pixel_size_nm", 1000.0))
    um = pixel_nm / 1000.0
    messages: list[str] = []
    result = ShadowRun(table=pd.DataFrame(), messages=messages)
    mask = np.asarray(particle_mask, dtype=bool)
    if not mask.any():
        messages.append("Draw the particle outline on the image first.")
        return result
    ys, xs = np.where(mask)
    x, y = float(xs.mean()), float(ys.mean())
    diameter = max(_mask_thickness_px(mask), 1.0)
    shadows, debug = measure_nsew_shadows(
        image, x, y, diameter,
        k_mad=k_mad, max_length_factor=max_length_factor, particle_mask=mask,
    )
    if not any(item.found for item in shadows.values()):
        messages.append(
            "No shadow found in any direction. Lower the darkness threshold, "
            "draw only the particle (not the shadow), and keep max shadow length large enough."
        )
    folder = (
        str(source_path.parent.relative_to(DEFAULT_INPUT_DIR))
        if source_path is not None and source_path.is_relative_to(DEFAULT_INPUT_DIR)
        else ("upload" if source_path is None else str(source_path.parent))
    )
    result.table = pd.DataFrame(_shadow_table_rows(
        shadows,
        key="upload",
        pid="",
        label="",
        folder=folder,
        tile=name,
        x=x,
        y=y,
        width=image.shape[1],
        height=image.shape[0],
        particle_um=diameter * um,
        um=um,
        light_height_mm=light_height_mm,
        ring_edge_mm=ring_edge_mm,
        ring_centre_mm=ring_centre_mm,
    ))
    if source_path is not None:
        source = _source_lines(source_path)
    else:
        matches = _input_images().get(name, ())
        source = _source_lines(matches[0]) if len(matches) == 1 else ["Uploaded file", name]
    result.images.append(("upload", debug_image(debug, nsew_shadow_title(shadows, um), source)))
    _write_shadow_run(result, out)
    return result


def _show_shadow_run(last: ShadowRun, *, download_key: str) -> None:
    import streamlit as st

    for message in last.messages:
        st.warning(message)
    if last.table.empty:
        return
    st.dataframe(last.table, hide_index=True, width="stretch")
    st.download_button(
        "Download CSV",
        last.table.to_csv(index=False).encode("utf-8"),
        file_name="shadow_check.csv",
        mime="text/csv",
        key=download_key,
    )
    if last.out_dir is not None:
        st.caption(f"Saved to `{last.out_dir}`")
    cols = st.columns(3)
    for i, (name, rgb) in enumerate(last.images):
        cols[i % 3].image(rgb, caption=name, width="stretch")


SHADOW_WIDGET_DEFAULTS: dict[str, Any] = {
    "shadow_keys": "",
    "shadow_light_height": LIGHT_HEIGHT_MM,
    "shadow_ring_edge": RING_EDGE_MM,
    "shadow_ring_centre": RING_CENTRE_MM,
    "shadow_k": 3.0,
    "shadow_max_len": 3.0,
}


def keep_shadow_inputs() -> None:
    
    import streamlit as st

    for key, default in SHADOW_WIDGET_DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)


def _light_controls() -> tuple[float, float, float, float, float]:
    import streamlit as st

    col_f, col_g, col_h = st.columns(3)
    light_height = col_f.number_input(
        "Light height above wafer (mm)", min_value=0.1, max_value=100.0, step=0.1,
        key="shadow_light_height",
    )
    ring_edge = col_g.number_input(
        "Distance to light at image edge (mm)", min_value=0.1, max_value=100.0, step=0.1,
        key="shadow_ring_edge",
        help="A particle on an image edge is this far from the ring on that side. "
             "Each shadow uses the side opposite the direction it falls.",
    )
    ring_centre = col_h.number_input(
        "Distance to light at image centre (mm)", min_value=0.1, max_value=100.0, step=0.1,
        key="shadow_ring_centre",
        help="At the centre every side uses this distance. Between centre and edge "
             "it changes linearly with the particle's position toward that side. "
             "Height = shadow length × light height / distance.",
    )
    col_k, col_l = st.columns(2)
    k_mad = col_k.number_input(
        "Darkness threshold (× noise)", min_value=0.1, max_value=10.0, step=0.01,
        key="shadow_k",
    )
    max_len = col_l.number_input(
        "Max shadow length (× diameter)", min_value=0.5, max_value=10.0, step=0.5,
        key="shadow_max_len",
    )
    return (
        float(light_height),
        float(ring_edge),
        float(ring_centre),
        float(k_mad),
        float(max_len),
    )


def _render_database_shadow(
    config: dict[str, Any],
    *,
    light_height: float,
    ring_edge: float,
    ring_centre: float,
    k_mad: float,
    max_len: float,
) -> None:
    import streamlit as st

    st.caption(
        "Enter keys of particles you already marked (shown under each crop in Database). "
        "The image folder is found from the label. Each key is measured in all four directions."
    )
    keys_text = st.text_input(
        "Particle keys",
        key="shadow_keys",
        placeholder="e.g. (1,1)_1402100_346700",
        help="Separate several keys with commas or spaces. A comma inside (row,col) stays part of the key.",
    )
    keys = _parse_keys(keys_text)
    if st.button("Measure shadows", type="primary", key="shadow_measure_keys"):
        if not keys:
            st.warning("Enter at least one key (press Enter after typing).")
        else:
            with st.spinner("Measuring shadows…"):
                run_result = run_shadow_check(
                    LabelStore(PACKAGE_ROOT / "labels").load(),
                    config,
                    keys=keys,
                    light_height_mm=light_height,
                    ring_edge_mm=ring_edge,
                    ring_centre_mm=ring_centre,
                    k_mad=k_mad,
                    max_length_factor=max_len,
                )
            st.session_state["shadow_last"] = run_result

    last: ShadowRun | None = st.session_state.get("shadow_last")
    if last is not None:
        _show_shadow_run(last, download_key="shadow_keys_csv")


def _image_to_url_shim(
    image: Any,
    width: int,
    clamp: bool,
    channels: str,
    output_format: str,
    image_id: str,
) -> str:
    
    import base64
    import io

    from PIL import Image as PILImage

    pil = image if isinstance(image, PILImage.Image) else PILImage.fromarray(np.asarray(image))
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    data = buf.getvalue()
    try:
        from streamlit.runtime import get_instance

        return get_instance().media_file_mgr.add(data, "image/png", str(image_id))
    except Exception:
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _load_st_canvas() -> Any | None:
    
    import streamlit.elements.image as st_image

    if not hasattr(st_image, "image_to_url"):
        st_image.image_to_url = _image_to_url_shim
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        return None
    return st_canvas


def _render_upload_shadow(
    config: dict[str, Any],
    *,
    light_height: float,
    ring_edge: float,
    ring_centre: float,
    k_mad: float,
    max_len: float,
) -> None:
    import streamlit as st
    from PIL import Image

    st_canvas = _load_st_canvas()

    st.caption(
        "Type a combi image path or file name, draw the particle or hair outline in green, "
        "then measure its shadow toward N, E, S, and W. Trace around the body "
        "(or paint over it); the drawing is the particle."
    )
    choices = list(image_choice_labels())
    current = st.session_state.get("shadow_image")
    if current in (None, ""):
        st.session_state.pop("shadow_image", None)
    elif current not in choices:
        choices.insert(0, current)
    if not choices:
        st.warning("No images found under Inputs.")
        return
    image_text = st.selectbox(
        "Image path or name",
        choices,
        index=None,
        placeholder="Start typing a name or path",
        accept_new_options=True,
        filter_mode="fuzzy",
        key="shadow_image",
        help="Suggestions appear as you type. A pasted path selects that file.",
    )
    if not image_text:
        return
    base_path, hits = resolve_named_image(str(image_text))
    if base_path is None:
        if hits:
            st.warning("Several images match. Pick one from the suggestions.")
        else:
            st.warning("No image found at that path or with that name.")
        return
    shown = base_path
    folder_key = str(base_path.parent)
    if st.session_state.get("shadow_draw_folder") not in (None, folder_key):
        st.session_state.pop("shadow_kept_mask", None)
        st.session_state.pop("shadow_upload_zoom_center", None)
        st.session_state.pop("shadow_zoom_click_stamp", None)
    st.session_state["shadow_draw_folder"] = folder_key
    state = st.session_state.get("shadow_upload")
    if (
        st.session_state.get("shadow_shown_path") != str(shown)
        or not isinstance(state, dict)
    ):
        state = {
            "name": shown.name,
            "image": load_tile_image(shown),
            "path": str(shown),
        }
        st.session_state["shadow_upload"] = state
        st.session_state["shadow_shown_path"] = str(shown)
        st.session_state["shadow_upload_sig"] = str(base_path)
        st.session_state.pop("shadow_upload_last", None)
    image = state["image"]
    name = str(state["name"])
    source_path = Path(str(state["path"]))
    height, width = int(image.shape[0]), int(image.shape[1])
    zoom = float(
        st.slider(
            "Zoom",
            min_value=1.0,
            max_value=8.0,
            value=float(st.session_state.get("shadow_upload_zoom", 1.0)),
            step=0.5,
            key="shadow_upload_zoom",
            help="Zoom in to draw a small particle or hair more accurately. "
                 "When zoomed, click the overview to move the view.",
        )
    )
    center = st.session_state.get("shadow_upload_zoom_center")
    if not (isinstance(center, (tuple, list)) and len(center) == 2):
        center = (0.5 * (width - 1), 0.5 * (height - 1))
        st.session_state["shadow_upload_zoom_center"] = center

    if zoom > 1.0:
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
        except ImportError:
            streamlit_image_coordinates = None
        overview, ov_scale = display_preview(image, max_side=280)
        _preview_tmp, crop_tmp = zoom_preview(
            image, float(center[0]), float(center[1]), zoom, max_side=PICK_MAX_SIDE
        )
        y0, x0, y1, x1 = crop_tmp
        marked = overview.copy()
        cv2.rectangle(
            marked,
            (int(round(x0 * ov_scale)), int(round(y0 * ov_scale))),
            (int(round(x1 * ov_scale) - 1), int(round(y1 * ov_scale) - 1)),
            (40, 180, 70),
            2,
        )
        st.caption("Overview · click to re-center the zoom")
        if streamlit_image_coordinates is None:
            st.image(marked, width=280)
        else:
            click = streamlit_image_coordinates(
                marked,
                key=f"shadow_zoom_overview_{st.session_state.get('shadow_upload_sig')}",
                width=min(280, int(marked.shape[1])),
                cursor="crosshair",
            )
            if isinstance(click, dict) and "x" in click:
                stamp = click.get("unix_time")
                if stamp != st.session_state.get("shadow_zoom_click_stamp"):
                    st.session_state["shadow_zoom_click_stamp"] = stamp
                    shown_w = float(click.get("width") or marked.shape[1])
                    shown_h = float(click.get("height") or marked.shape[0])
                    x_disp = float(click["x"]) * marked.shape[1] / max(shown_w, 1.0)
                    y_disp = float(click["y"]) * marked.shape[0] / max(shown_h, 1.0)
                    st.session_state["shadow_upload_zoom_center"] = (
                        x_disp / max(ov_scale, 1e-9),
                        y_disp / max(ov_scale, 1e-9),
                    )
                    st.rerun()
        center = st.session_state["shadow_upload_zoom_center"]

    kept = st.session_state.get("shadow_kept_mask")
    if not (isinstance(kept, np.ndarray) and kept.shape == image.shape):
        kept = np.zeros(image.shape[:2], dtype=bool)
    preview, crop = zoom_preview(
        image, float(center[0]), float(center[1]), zoom, max_side=PICK_MAX_SIDE
    )
    preview = paint_kept_mask(preview, kept, crop)
    where = ""
    if source_path is not None:
        where = "".join(_source_lines(source_path)) + " · "
    st.caption(
        f"{where}draw on the {'zoomed view' if zoom > 1 else 'image'} "
        f"({crop[3] - crop[1]}×{crop[2] - crop[0]} px)"
    )

    def _clear_drawing() -> None:
        st.session_state.pop("shadow_kept_mask", None)
        st.session_state.pop("shadow_upload_last", None)
        st.session_state["shadow_draw_epoch"] = int(st.session_state.get("shadow_draw_epoch", 0)) + 1

    st.button("Clear drawing", key="shadow_clear_drawing", on_click=_clear_drawing)
    pic_col, _ = st.columns(2)
    with pic_col:
        if st_canvas is None:
            st.image(preview, width="stretch")
            st.warning("Install streamlit-drawable-canvas==0.9.3 to draw on the image.")
            canvas_data = None
        else:
            view_key = (
                f"{st.session_state.get('shadow_upload_sig')}_{name}_"
                f"z{zoom:.1f}_{int(center[0])}_{int(center[1])}_"
                f"e{st.session_state.get('shadow_draw_epoch', 0)}"
            )
            canvas = st_canvas(
                fill_color="rgba(40, 180, 70, 0.35)",
                stroke_width=3,
                stroke_color="rgba(40, 180, 70, 1.0)",
                background_image=Image.fromarray(preview),
                update_streamlit=True,
                height=int(preview.shape[0]),
                width=int(preview.shape[1]),
                drawing_mode="freedraw",
                display_toolbar=True,
                key=f"shadow_upload_canvas_{view_key}",
            )
            canvas_data = None if canvas is None else getattr(canvas, "image_data", None)

        drawn = mask_from_canvas(canvas_data, image.shape, crop=crop)
        mask = kept | drawn
        if drawn.any():
            st.session_state["shadow_kept_mask"] = mask
        if mask.any():
            st.caption(f"Outline covers {int(mask.sum())} px")
        if st.button("Measure shadows", type="primary", key="shadow_measure_pick"):
            if not mask.any():
                st.warning("Draw the particle outline on the image first.")
            else:
                with st.spinner("Measuring shadow…"):
                    st.session_state["shadow_upload_last"] = run_shadow_on_mask(
                        image,
                        name,
                        mask,
                        config,
                        source_path=source_path,
                        light_height_mm=light_height,
                        ring_edge_mm=ring_edge,
                        ring_centre_mm=ring_centre,
                        k_mad=k_mad,
                        max_length_factor=max_len,
                    )

    last: ShadowRun | None = st.session_state.get("shadow_upload_last")
    if last is not None:
        _show_shadow_run(last, download_key="shadow_upload_csv")


def render_shadow_tab(config: dict[str, Any]) -> None:
    
    import streamlit as st

    keep_shadow_inputs()
    mode = st.session_state.get("shadow_mode")
    if mode not in ("database", "upload"):
        mode = None

    def _set_mode(next_mode: str) -> None:
        st.session_state["shadow_mode"] = (
            None if st.session_state.get("shadow_mode") == next_mode else next_mode
        )

    st.caption(
        "On an NSEW combi the ring lights all four sides, so each particle is measured "
        "for a shadow toward N, E, S, and W. Pixel size comes from Adjust parameters "
        "on the Run tab."
    )
    db_col, upload_col = st.columns(2)
    db_col.button(
        "Hide database" if mode == "database" else "From database",
        key="shadow_mode_database",
        type="primary" if mode == "database" else "secondary",
        on_click=_set_mode,
        args=("database",),
        width="stretch",
    )
    upload_col.button(
        "Hide upload" if mode == "upload" else "From uploaded image",
        key="shadow_mode_upload",
        type="primary" if mode == "upload" else "secondary",
        on_click=_set_mode,
        args=("upload",),
        width="stretch",
    )
    if mode is None:
        return

    with st.container(border=True, key="block-shadow-light"):
        (
            light_height,
            ring_edge,
            ring_centre,
            k_mad,
            max_len,
        ) = _light_controls()
    params = dict(
        light_height=light_height,
        ring_edge=ring_edge,
        ring_centre=ring_centre,
        k_mad=k_mad,
        max_len=max_len,
    )
    with st.container(border=True, key="block-shadow-measure"):
        if mode == "database":
            _render_database_shadow(config, **params)
        else:
            _render_upload_shadow(config, **params)
