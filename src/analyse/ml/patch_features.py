from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.morphology import skeletonize

from src.analyse.detection.detector import CANDIDATE_FEATURE_FIELDS, ParticleCandidate
from src.overall.io.tile_loader import DEFAULT_FILENAME_PATTERN
from src.analyse.ml.features import candidate_feature_vector
from src.analyse.ml.patches import tile_row_col

HOG_SIZE = 64
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2
HOG_ORIENTATIONS = 8
HOG_PIXELS = 16
HOG_CELLS = 2

TABULAR_LAYOUT_FIELDS = ("size_um", "confidence") + CANDIDATE_FEATURE_FIELDS + (
    "tile_row",
    "tile_col",
)


def _hog_length() -> int:
    cells = HOG_SIZE // HOG_PIXELS
    blocks = max(cells - HOG_CELLS + 1, 1)
    return int((blocks**2) * (HOG_CELLS**2) * HOG_ORIENTATIONS)


GEOM_NAMES = (
    "solidity",
    "eccentricity",
    "extent",
    "skeleton_frac",
    "orient_entropy",
    "cc_aspect",
    "cc_holes",
    "ring_contrast",
)

PATCH_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"hog_raw_{i}" for i in range(_hog_length()))
    + tuple(f"hog_res_{i}" for i in range(_hog_length()))
    + tuple(f"lbp_raw_{i}" for i in range(LBP_BINS))
    + tuple(f"lbp_res_{i}" for i in range(LBP_BINS))
    + tuple(f"hu_raw_{i}" for i in range(7))
    + tuple(f"hu_res_{i}" for i in range(7))
    + GEOM_NAMES
    + TABULAR_LAYOUT_FIELDS
)


def _resize(channel: np.ndarray, size: int = HOG_SIZE) -> np.ndarray:
    array = np.asarray(channel, dtype=np.float32)
    if array.shape == (size, size):
        return array
    return cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)


def _hog_vec(channel: np.ndarray) -> np.ndarray:
    image = _resize(channel)
    return hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
        cells_per_block=(HOG_CELLS, HOG_CELLS),
        block_norm="L2-Hys",
        feature_vector=True,
        transform_sqrt=True,
    ).astype(np.float64)


def _uint8(channel: np.ndarray) -> np.ndarray:
    image = _resize(channel)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _lbp_hist(channel: np.ndarray) -> np.ndarray:
    codes = local_binary_pattern(_uint8(channel), P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(codes, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    return hist.astype(np.float64)


def _hu(channel: np.ndarray) -> np.ndarray:
    moments = cv2.moments(_uint8(channel))
    hu = cv2.HuMoments(moments).reshape(-1)
    return np.sign(hu) * np.log1p(np.abs(hu))


def _binary(channel: np.ndarray) -> np.ndarray:
    image = _resize(channel)
    thresh = float(np.percentile(image, 70.0))
    return (image >= max(thresh, 0.15)).astype(np.uint8)


def _geometry(raw: np.ndarray, residual: np.ndarray) -> np.ndarray:
    residual_small = _resize(residual)
    mask = _binary(residual_small)
    area = float(np.count_nonzero(mask))
    total = float(mask.size)
    solidity = 0.0
    eccentricity = 0.0
    extent = area / total if total else 0.0
    skeleton_frac = 0.0
    cc_aspect = 1.0
    cc_holes = 0.0
    if area >= 4:
        ys, xs = np.nonzero(mask)
        yy = ys.astype(np.float64)
        xx = xs.astype(np.float64)
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
        hull_area = float(cv2.contourArea(hull)) if hull is not None and len(hull) >= 3 else area
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        cov = np.cov(np.vstack([xx, yy]))
        eig = np.linalg.eigvalsh(cov)
        eig = np.clip(eig, 1e-9, None)
        eccentricity = float(np.sqrt(1.0 - eig[0] / eig[-1])) if eig[-1] > 0 else 0.0
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if nlab > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            w = max(float(stats[idx, cv2.CC_STAT_WIDTH]), 1.0)
            h = max(float(stats[idx, cv2.CC_STAT_HEIGHT]), 1.0)
            cc_aspect = float(max(w, h) / min(w, h))
        inv = 1 - mask
        n_holes, _ = cv2.connectedComponents(inv, 8)
        cc_holes = float(max(n_holes - 1, 0))
        skeleton_frac = float(np.count_nonzero(skeletonize(mask.astype(bool)))) / area
    gy, gx = np.gradient(residual_small.astype(np.float64))
    angles = np.arctan2(gy, gx)
    mag = np.hypot(gx, gy)
    weights = mag.reshape(-1)
    if float(weights.sum()) > 0:
        hist, _ = np.histogram(
            angles.reshape(-1), bins=8, range=(-np.pi, np.pi), weights=weights
        )
        hist = hist / hist.sum()
        orient_entropy = float(-(hist[hist > 0] * np.log(hist[hist > 0] + 1e-12)).sum())
    else:
        orient_entropy = 0.0
    h, w = residual_small.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(yy - cy, xx - cx)
    inner = residual_small[dist <= 8]
    ring = residual_small[(dist > 8) & (dist <= 20)]
    ring_contrast = float(inner.mean() - ring.mean()) if inner.size and ring.size else 0.0
    return np.array(
        [
            solidity,
            eccentricity,
            extent,
            skeleton_frac,
            orient_entropy,
            cc_aspect,
            cc_holes,
            ring_contrast,
        ],
        dtype=np.float64,
    )


def patch_descriptor(channels: np.ndarray) -> np.ndarray:
    
    raw = np.asarray(channels[0], dtype=np.float32)
    residual = np.asarray(channels[1], dtype=np.float32)
    parts = (
        _hog_vec(raw),
        _hog_vec(residual),
        _lbp_hist(raw),
        _lbp_hist(residual),
        _hu(raw),
        _hu(residual),
        _geometry(raw, residual),
    )
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])


def layout_from_candidate(
    candidate: ParticleCandidate,
    pixel_size_nm: float,
    source_tile: str = "",
    pattern: str = DEFAULT_FILENAME_PATTERN,
) -> np.ndarray:
    tile_row, tile_col = tile_row_col(source_tile, pattern)
    base = candidate_feature_vector(candidate, pixel_size_nm)
    return np.concatenate([base, [float(tile_row), float(tile_col)]])


def patch_feature_vector(
    channels: np.ndarray,
    layout: np.ndarray | None = None,
) -> np.ndarray:
    image_part = patch_descriptor(channels)
    extra = (
        np.asarray(layout, dtype=np.float64)
        if layout is not None
        else np.zeros(len(TABULAR_LAYOUT_FIELDS), dtype=np.float64)
    )
    vector = np.concatenate([image_part, extra])
    if vector.size != len(PATCH_FEATURE_NAMES):
        raise ValueError(
            f"Patch feature length {vector.size} != {len(PATCH_FEATURE_NAMES)}"
        )
    return vector


def patches_to_matrix(
    channels: np.ndarray,
    layouts: Sequence[np.ndarray] | np.ndarray | None = None,
) -> np.ndarray:
    
    n = int(channels.shape[0])
    if n == 0:
        return np.empty((0, len(PATCH_FEATURE_NAMES)), dtype=np.float64)
    rows = []
    for i in range(n):
        layout = None if layouts is None else np.asarray(layouts[i], dtype=np.float64)
        rows.append(patch_feature_vector(channels[i], layout))
    return np.vstack(rows)
