"""Summary statistics and downsampled overlay images for reporting."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.overall.config import cfg_get
from src.main_pipeline.stitching.stitcher import LazyMosaic, downsample_for_target


SIZE_COUNT_THRESHOLDS_UM = (10, 20, 25, 50, 75)
SIZE_BIN_STEP_UM = 5.0
NM_PER_UM = 1000.0


def summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Return count and size-distribution summary for a particle table."""
    if df.empty:
        return {
            "particle_count": 0,
            "size_mean": 0.0,
            "size_std": 0.0,
            "size_min": 0.0,
            "size_max": 0.0,
        }
    sizes = df["size"].astype(float)
    return {
        "particle_count": int(len(df)),
        "size_mean": float(sizes.mean()),
        "size_std": float(sizes.std(ddof=0)),
        "size_min": float(sizes.min()),
        "size_max": float(sizes.max()),
    }


def particles_at_least(
    df: pd.DataFrame | None,
    thresholds_um: tuple[float, ...] = SIZE_COUNT_THRESHOLDS_UM,
) -> pd.DataFrame:
    """One row per size: how many particles are at least that large.

    ``size`` is equivalent circular diameter in nm.
    """
    labels = [
        f"≥{int(threshold) if float(threshold).is_integer() else threshold} µm"
        for threshold in thresholds_um
    ]
    if df is None or df.empty or "size" not in df.columns:
        counts = [0] * len(thresholds_um)
    else:
        sizes_um = df["size"].astype(float).to_numpy() / NM_PER_UM
        counts = [int(np.sum(sizes_um >= float(threshold))) for threshold in thresholds_um]
    return pd.DataFrame({"Size": labels, "Particles": counts})


def size_bin_edges(
    min_um: float,
    max_um: float,
    step_um: float = SIZE_BIN_STEP_UM,
) -> np.ndarray:
    """``[min, min+step, ...]`` covering ``max_um``.

    The last edge is the first step boundary at or above the largest size.
    A size that lands on ``min`` still gets one bin.
    """
    start = float(min_um)
    stop = max(float(max_um), start)
    step = float(step_um)
    steps = int(np.ceil((stop - start) / step - 1e-9))
    if steps < 1:
        steps = 1
    return start + np.arange(steps + 1, dtype=float) * step


def _size_bin_label(left: float, right: float) -> str:
    def fmt(value: float) -> str:
        rounded = round(value)
        if abs(value - rounded) < 1e-6:
            return str(int(rounded))
        return f"{value:.2f}"

    return f"{fmt(left)}–{fmt(right)}"


def size_distribution(
    df: pd.DataFrame | None,
    min_um: float,
    step_um: float = SIZE_BIN_STEP_UM,
) -> pd.DataFrame:
    """Particle counts in 5 µm bins starting at the minimum size.

    ``size`` is equivalent circular diameter in nm. Bins are
    ``[min, min+5)``, ``[min+5, min+10)``, and so on through the largest size.
    """
    if df is None or df.empty or "size" not in df.columns:
        edges = size_bin_edges(min_um, min_um, step_um)
        counts = np.zeros(len(edges) - 1, dtype=int)
    else:
        sizes_um = df["size"].astype(float).to_numpy() / NM_PER_UM
        edges = size_bin_edges(min_um, float(np.max(sizes_um)), step_um)
        counts, edges = np.histogram(sizes_um, bins=edges)
    labels = [_size_bin_label(edges[i], edges[i + 1]) for i in range(len(counts))]
    return pd.DataFrame({"count": counts}, index=labels)


def overlay_downsample(
    mosaic: LazyMosaic,
    config: dict[str, Any],
    crop: tuple[int, int, int, int] | None = None,
) -> int:
    """Pick the integer factor used to downsample tiles before stitching.

    ``report.downsample`` > 0 is an explicit override. Otherwise the factor is
    chosen so the stitched RGB overlay is about ``report.target_mb`` megabytes.
    """
    configured = int(cfg_get(config, "report.downsample", 0) or 0)
    if configured > 0:
        return configured
    if crop is None:
        height, width = mosaic.full_height, mosaic.full_width
    else:
        height, width = int(crop[2]), int(crop[3])
    target_mb = float(cfg_get(config, "report.target_mb", 20.0))
    return downsample_for_target(height, width, target_mb=target_mb, channels=3)


def overlay_markers(
    mosaic: LazyMosaic,
    particles: pd.DataFrame,
    config: dict[str, Any],
    crop: tuple[int, int, int, int] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> np.ndarray:
    """Downsampled mosaic with particle markers. Coordinates are global nm or px.

    Marker positions are converted back to mosaic pixels using ``pixel_size_nm``.
    ``crop`` is ``(y, x, h, w)`` in full-resolution mosaic pixels. ``crop=None``
    stitches every tile into one overview.
    """
    downsample = overlay_downsample(mosaic, config, crop=crop)
    preview = mosaic.preview(downsample=downsample, crop=crop, progress_cb=progress_cb)
    rgb = _to_display_rgb(preview)
    if particles.empty or rgb.size == 0:
        return rgb

    placed_names = {placement.name for placement in mosaic.placements}
    if placed_names and "source_tile" in particles.columns:
        names = particles["source_tile"].astype(str).map(
            lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1]
        )
        particles = particles.loc[names.isin(placed_names)]
        if particles.empty:
            return rgb

    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    if pixel_size <= 0:
        pixel_size = 1.0
    y_off = 0 if crop is None else crop[0]
    x_off = 0 if crop is None else crop[1]

    for _, row in particles.iterrows():
        x_px = float(row["x_global"]) / pixel_size
        y_px = float(row["y_global"]) / pixel_size
        x_d = int(round((x_px - x_off) / downsample))
        y_d = int(round((y_px - y_off) / downsample))
        if not (0 <= x_d < rgb.shape[1] and 0 <= y_d < rgb.shape[0]):
            continue
        particle_r = float(row["size"]) / pixel_size / downsample / 2.0
        radius = max(int(round(particle_r)) + 10, 14)
        cv2.circle(
            rgb, (x_d, y_d), radius, (255, 140, 40), thickness=3, lineType=cv2.LINE_AA
        )
    return rgb


def _to_display_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    low, high = np.percentile(array, (1.0, 99.5))
    if high <= low:
        high = low + 1e-6
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    gray = (scaled * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


_PAGE_W = 1240
_PAGE_H = 1754
_MARGIN = 64
_INK = (28, 32, 38)
_MUTED = (90, 98, 110)
_BAR = (31, 111, 184)
_RULE = (220, 224, 230)


def write_last_run_pdf(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    path: str | Path,
    mosaic_rgb: np.ndarray | None = None,
) -> Path:
    """Write the Last run summary, size histogram, and mosaic with hits."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pages = _last_run_pages(table, config, mosaic_rgb)
    pages[0].save(
        destination,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )
    return destination


def _last_run_pages(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    mosaic_rgb: np.ndarray | None,
) -> list[Image.Image]:
    pages = [_summary_page(table, config)]
    if mosaic_rgb is not None and np.asarray(mosaic_rgb).size > 0:
        pages.append(_mosaic_page(np.asarray(mosaic_rgb)))
    return pages


_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
_FONT_FILE: str | None = None


def _font_file() -> str | None:
    global _FONT_FILE
    if _FONT_FILE is not None:
        return _FONT_FILE or None
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            _FONT_FILE = candidate
            return candidate
    _FONT_FILE = ""
    return None


def _font(size: int) -> ImageFont.ImageFont:
    path = _font_file()
    if path:
        return ImageFont.truetype(path, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _run_title(config: Mapping[str, Any]) -> str:
    raw = str(config.get("input_dir") or config.get("output_dir") or "").strip()
    name = Path(raw).name if raw else "unknown"
    return f"Run on {name}"


def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    box = font.getbbox(text)
    return int(box[2] - box[0])


def _summary_page(table: pd.DataFrame, config: Mapping[str, Any]) -> Image.Image:
    image = Image.new("RGB", (_PAGE_W, _PAGE_H), "white")
    draw = ImageDraw.Draw(image)
    metric_label = _font(18)
    metric_value = _font(36)
    body = _font(20)
    small = _font(16)

    y = _MARGIN
    heading = _run_title(config)
    title_size = 42
    title = _font(title_size)
    while title_size > 22 and _text_width(title, heading) > _PAGE_W - 2 * _MARGIN:
        title_size -= 2
        title = _font(title_size)
    draw.text((_MARGIN, y), heading, fill=_INK, font=title)
    y += 72

    stats = summary_stats(table)
    metrics = (
        ("Particles", f"{stats['particle_count']}"),
        ("Mean size (µm)", f"{stats['size_mean'] / NM_PER_UM:.2f}"),
        ("Min size (µm)", f"{stats['size_min'] / NM_PER_UM:.2f}"),
        ("Max size (µm)", f"{stats['size_max'] / NM_PER_UM:.2f}"),
    )
    slot = (_PAGE_W - 2 * _MARGIN) // len(metrics)
    for index, (label, value) in enumerate(metrics):
        x = _MARGIN + index * slot
        draw.text((x, y), label, fill=_MUTED, font=metric_label)
        draw.text((x, y + 28), value, fill=_INK, font=metric_value)
    y += 100
    draw.line([(_MARGIN, y), (_PAGE_W - _MARGIN, y)], fill=_RULE, width=2)
    y += 28

    caption = "Detected particles at least this large (equivalent circular diameter)."
    draw.text((_MARGIN, y), caption, fill=_INK, font=body)
    y += 40
    counts = particles_at_least(table)
    y = _draw_count_table(draw, counts, y, body, small)
    y += 36

    draw.text((_MARGIN, y), "Size distribution (µm)", fill=_INK, font=_font(28))
    y += 48
    min_size_nm = float((config.get("detection") or {}).get("min_size_nm") or 0.0)
    distribution = size_distribution(table, min_size_nm / NM_PER_UM)
    _draw_histogram(image, distribution, y, y + 640, small)
    return image


def _draw_count_table(
    draw: ImageDraw.ImageDraw,
    counts: pd.DataFrame,
    y: int,
    body: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> int:
    left = _MARGIN
    right = _PAGE_W - _MARGIN
    mid = left + 280
    row_h = 36
    draw.text((left, y), "Size", fill=_MUTED, font=small)
    draw.text((mid, y), "Particles", fill=_MUTED, font=small)
    y += 28
    draw.line([(left, y), (right, y)], fill=_RULE, width=1)
    y += 8
    for _, row in counts.iterrows():
        draw.text((left, y), str(row["Size"]), fill=_INK, font=body)
        draw.text((mid, y), str(int(row["Particles"])), fill=_INK, font=body)
        y += row_h
    return y


def _axis_ticks(peak: float, count: int = 4) -> list[float]:
    if peak <= 0:
        return [0.0]
    raw = peak / max(count, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = 10 * power
    for factor in (1, 2, 2.5, 5, 10):
        if factor * power >= raw - 1e-9:
            step = factor * power
            break
    limit = math.ceil((peak - 1e-9) / step) * step
    ticks: list[float] = []
    value = 0.0
    while value <= limit + step * 0.01:
        ticks.append(value)
        value += step
        if len(ticks) > 8:
            break
    return ticks


def _tick_label(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.1f}"


def _paste_vertical(
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
) -> None:
    """Draw ``text`` rotated so it reads upward beside the y-axis."""
    bbox = font.getbbox(text)
    width = int(bbox[2] - bbox[0]) + 4
    height = int(bbox[3] - bbox[1]) + 4
    plate = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(plate).text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=_INK + (255,))
    rotated = plate.rotate(90, expand=True)
    image.paste(rotated, (x, y), rotated)


def _draw_histogram(
    image: Image.Image,
    distribution: pd.DataFrame,
    top: int,
    bottom: int,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    label_w = 52
    left = _MARGIN + 28 + label_w
    right = _PAGE_W - _MARGIN
    plot_top = top + 8
    plot_bottom = bottom - 56
    counts = [int(value) for value in distribution["count"].tolist()]
    labels = [str(label) for label in distribution.index]
    peak = max(counts) if counts else 0
    ticks = _axis_ticks(float(peak))
    scale = ticks[-1] if ticks[-1] > 0 else 1.0
    for tick in ticks:
        if tick <= 0:
            continue
        y = plot_bottom - (tick / scale) * (plot_bottom - plot_top)
        draw.line([(left, y), (right, y)], fill=_RULE, width=1)
    draw.line([(left, plot_top), (left, plot_bottom)], fill=_INK, width=2)
    draw.line([(left, plot_bottom), (right, plot_bottom)], fill=_INK, width=2)
    for tick in ticks:
        y = plot_bottom - (tick / scale) * (plot_bottom - plot_top)
        draw.line([(left - 6, y), (left, y)], fill=_INK, width=2)
        label = _tick_label(tick)
        draw.text((left - 10 - _text_width(font, label), y - 8), label, fill=_INK, font=font)
    axis_y = "Amount of particles"
    _paste_vertical(image, axis_y, font, _MARGIN - 8, plot_top + 70)
    if not counts:
        return
    slot = (right - left) / len(counts)
    gap = min(10, max(2, slot * 0.25))
    for index, (count, label) in enumerate(zip(counts, labels)):
        bar_left = left + index * slot + gap / 2
        bar_right = left + (index + 1) * slot - gap / 2
        height = (count / scale) * (plot_bottom - plot_top)
        bar_top = plot_bottom - height
        draw.rectangle([bar_left, bar_top, bar_right, plot_bottom], fill=_BAR)
        center = (bar_left + bar_right) / 2
        draw.line([(center, plot_bottom), (center, plot_bottom + 6)], fill=_INK, width=2)
        text_w = _text_width(font, label)
        draw.text((center - text_w / 2, plot_bottom + 10), label, fill=_INK, font=font)
    axis = "Size (µm)"
    draw.text(
        ((left + right) / 2 - _text_width(font, axis) / 2, bottom - 22),
        axis,
        fill=_INK,
        font=font,
    )


def _mosaic_page(mosaic_rgb: np.ndarray) -> Image.Image:
    image = Image.new("RGB", (_PAGE_W, _PAGE_H), "white")
    draw = ImageDraw.Draw(image)
    draw.text((_MARGIN, _MARGIN), "Stitched mosaic", fill=_INK, font=_font(42))
    draw.text(
        (_MARGIN, _MARGIN + 56),
        "Downsampled mosaic with detected particles marked.",
        fill=_MUTED,
        font=_font(20),
    )
    rgb = np.asarray(mosaic_rgb)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    tile = Image.fromarray(np.ascontiguousarray(rgb[:, :, :3].astype(np.uint8)), mode="RGB")
    max_w = _PAGE_W - 2 * _MARGIN
    max_h = _PAGE_H - _MARGIN - 140
    scale = min(max_w / tile.width, max_h / tile.height, 1.0)
    resized = tile.resize(
        (max(int(tile.width * scale), 1), max(int(tile.height * scale), 1)),
        Image.Resampling.LANCZOS,
    )
    x = (_PAGE_W - resized.width) // 2
    y = _MARGIN + 110
    image.paste(resized, (x, y))
    return image


def encode_overlay_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    """JPEG bytes for download; ``rgb`` is uint8 RGB."""
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("Failed to encode mosaic JPEG.")
    return buffer.tobytes()


def write_overlay_image(
    mosaic: LazyMosaic,
    particles: pd.DataFrame,
    config: dict[str, Any],
    path: str | Path,
    crop: tuple[int, int, int, int] | None = None,
) -> Path:
    """Stitch a size-capped mosaic overlay and write it to ``path``."""
    rgb = overlay_markers(mosaic, particles, config, crop=crop)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        destination.write_bytes(encode_overlay_jpeg(rgb))
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(destination), bgr):
            raise RuntimeError(f"Failed to write mosaic image to {destination}")
    return destination
