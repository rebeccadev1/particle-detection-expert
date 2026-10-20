"""Combine N/S/E/W directional images and write results next to them."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.overall.config import DEFAULT_INPUT_DIR, resolve_output_dir
from src.main_pipeline.measurement.measurer import (
    is_combi_tile,
    lighting_role,
    quad_direction,
)

DIRECTIONS = ("N", "S", "E", "W")
# Capture names one ring side by grid cell: ``(1,3)_N.png``.
_POSITIONAL_DIRECTION = re.compile(
    r"^\((?P<row>\d+),(?P<col>\d+)\)_(?P<direction>[NSEW])$",
    re.IGNORECASE,
)
_POSITIONAL_COMBI = re.compile(r"^\((?P<row>\d+),(?P<col>\d+)\)$")
# Adjacent compass pairs only (not opposite N–S or E–W).
ORTHOGONAL_PAIRS = (("N", "E"), ("N", "W"), ("S", "E"), ("S", "W"))
IMAGE_EXTS = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}
MIN_COUNTS = (2, 3, 4)
KIND_NAMES = ("combi", "symmetry_map")
# White point for a saved combi. The single hottest pixel is a speck; stretching
# to it leaves real particles in a few gray levels and the detector then lifts the noise.
COMBI_SCALE_PERCENTILE = 99.9
KIND_FOLDERS = {
    "combi": "Combi",
    "symmetry_map": "Symmetry",
}
# Prefer matching the input suffix (usually .bmp); fall back to .png.
OUTPUT_NAMES = tuple(
    f"{kind}_{min_count}of4" for min_count in MIN_COUNTS for kind in KIND_NAMES
)


def _direction_name(path: Path) -> str | None:
    """N/S/E/W from a stem, or from a quad shot (quad0=N, quad1=E, quad2=S, quad3=W)."""
    stem = path.stem.upper()
    if stem in DIRECTIONS:
        return stem
    return quad_direction(path.name)


def list_direction_images(folder: str | Path) -> dict[str, Path]:
    """Images in ``folder`` named N, S, E, or W, including Groundup quad shots."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")

    found: dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = _direction_name(path)
        if name not in DIRECTIONS:
            continue
        if name in found:
            raise FileExistsError(
                f"Multiple images named {name} in {folder}: "
                f"{found[name].name} and {path.name}"
            )
        found[name] = path
    return {d: found[d] for d in DIRECTIONS if d in found}


def find_nsew_images(folder: str | Path) -> dict[str, Path]:
    """Locate four images named N, S, E, W in ``folder`` (any common image suffix)."""
    found = list_direction_images(folder)
    missing = [d for d in DIRECTIONS if d not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing directional images in {folder}: {', '.join(missing)}. "
            "Expected files named N, S, E, W (e.g. N.bmp, S.png), "
            "or wafer_shot quad files (quad0=N, quad1=E, quad2=S, quad3=W)."
        )
    return found


def nsew_output_dir(input_folder: str | Path) -> Path:
    """``Outputs/Output {input folder name}`` (or a sibling folder in tests)."""
    folder = Path(input_folder).resolve()
    name = f"Output {folder.name}"
    try:
        folder.relative_to(DEFAULT_INPUT_DIR.resolve())
    except ValueError:
        return folder.parent / name
    return resolve_output_dir(name)


def kind_output_dir(output_dir: str | Path, kind: str) -> Path:
    """Subfolder under the NSEW output folder: Combi or Symmetry."""
    name = KIND_FOLDERS.get(kind, kind)
    folder = Path(output_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _usable_dpi(dpi: Any) -> tuple[float, float] | None:
    """Return DPI only when both axes are positive (BMP often stores 0,0)."""
    if not dpi or len(dpi) < 2:
        return None
    x_dpi, y_dpi = float(dpi[0]), float(dpi[1])
    if x_dpi <= 0 or y_dpi <= 0:
        return None
    return (x_dpi, y_dpi)


def output_suffix(reference_path: str | Path) -> str:
    """Match the input image format so viewers treat size the same way."""
    suffix = Path(reference_path).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    return ".png"


def save_same_resolution(path: str | Path, array: np.ndarray, reference_path: str | Path) -> None:
    """Write ``array`` with the same pixel size (and DPI when known) as ``reference_path``."""
    path = Path(path)
    h, w = array.shape[:2]
    with Image.open(reference_path) as ref:
        ref_w, ref_h = ref.size
        if (w, h) != (ref_w, ref_h):
            raise ValueError(
                f"{path}: output is {w}x{h} px but input is {ref_w}x{ref_h} px"
            )
        dpi = _usable_dpi(ref.info.get("dpi"))
    image = Image.fromarray(array)
    save_kw: dict[str, Any] = {}
    if dpi is not None:
        save_kw["dpi"] = dpi
    # Uncompressed BMP matches input size; PNG uses no palette downsample.
    if path.suffix.lower() == ".png":
        save_kw["compress_level"] = 1
    image.save(path, **save_kw)


def flatten_illumination(img: np.ndarray, sigma: float = 120) -> np.ndarray:
    """Remove the large-scale brightness gradient caused by directional lighting."""
    img_f = img.astype(np.float32)
    background = cv2.GaussianBlur(img_f, (0, 0), sigma)
    background = np.clip(background, 1.0, None)
    return img_f / background


def symmetry_signal_4way(
    sorted_stack: np.ndarray,
    sum_val: np.ndarray,
    min_count: int,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Score pixels that are bright in at least ``min_count`` of the four angles.

    ``sorted_stack`` is brightness sorted low→high. Index ``4 - min_count`` is the
    darkest of those ``min_count`` brightest angles (2nd-brightest for 2-of-4,
    2nd-darkest for 3-of-4, darkest for 4-of-4).
    """
    n_dirs = sorted_stack.shape[0]
    if min_count < 1 or min_count > n_dirs:
        raise ValueError(f"min_count must be between 1 and {n_dirs}, got {min_count}")
    min_val = sorted_stack[n_dirs - min_count]
    sym = (n_dirs * min_val) / (sum_val + eps)
    combined_brightness = sum_val / float(n_dirs)
    isotropic = sym * combined_brightness
    return isotropic, sym


def symmetry_signal_orthogonal_2of4(
    stack_nsew: np.ndarray,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """2-of-4 score that only lights when an orthogonal pair agrees.

    ``stack_nsew`` is ``(4, H, W)`` in ``DIRECTIONS`` order (N, S, E, W).
    Opposite pairs (N–S, E–W) alone do not contribute; only N–E, N–W, S–E, S–W.
    """
    if stack_nsew.shape[0] != 4:
        raise ValueError(f"expected 4 directions, got {stack_nsew.shape[0]}")
    by_dir = {name: stack_nsew[i] for i, name in enumerate(DIRECTIONS)}
    pair_mins = [
        np.minimum(by_dir[a], by_dir[b]) for a, b in ORTHOGONAL_PAIRS
    ]
    min_val = pair_mins[0]
    for other in pair_mins[1:]:
        min_val = np.maximum(min_val, other)
    sum_val = np.sum(stack_nsew, axis=0)
    n_dirs = 4
    sym = (n_dirs * min_val) / (sum_val + eps)
    isotropic = sym * (sum_val / float(n_dirs))
    return isotropic, sym


def scale_combi_uint8(
    values: np.ndarray,
    percentile: float = COMBI_SCALE_PERCENTILE,
) -> np.ndarray:
    """Scale a non-negative combi score to 8-bit, clipping the top tail.

    The white point is ``percentile`` (default 99.9), not the maximum. Values
    above that clip to 255 so one hot pixel does not crush the rest of the tile.
    """
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    hi = float(np.percentile(finite, percentile))
    if hi <= 0:
        hi = float(np.max(finite))
    if hi <= 0:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip(array / np.float32(hi), 0.0, 1.0)
    return (scaled * np.float32(255.0)).astype(np.uint8)


def load_flattened_nsew(
    path1: str | Path,
    path2: str | Path,
    path3: str | Path,
    path4: str | Path,
    flatten_sigma: float = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sorted_stack, sum_val, stack)`` with stack in path order."""
    paths = [path1, path2, path3, path4]
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    for i, img in enumerate(imgs):
        if img is None:
            raise FileNotFoundError(f"Could not read image: {paths[i]}")
        if img.shape != imgs[0].shape:
            raise ValueError("All images must be the same size and pixel-aligned")
    f_imgs = [flatten_illumination(img, flatten_sigma) for img in imgs]
    stack = np.array(f_imgs)
    return np.sort(stack, axis=0), np.sum(stack, axis=0), stack


def combine_nsew_folder(
    folder: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write 2-of-4, 3-of-4, and 4-of-4 combined images into an output folder."""
    folder = Path(folder)
    dest = Path(output_dir) if output_dir is not None else nsew_output_dir(folder)
    dest.mkdir(parents=True, exist_ok=True)
    nsew = find_nsew_images(folder)
    paths = [nsew[d] for d in DIRECTIONS]
    sorted_stack, sum_val, flat_stack = load_flattened_nsew(*paths)

    suffix = output_suffix(paths[0])
    variants: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for min_count in MIN_COUNTS:
        isotropic, sym = symmetry_signal_4way(sorted_stack, sum_val, min_count)
        iso_u8 = scale_combi_uint8(isotropic)
        sym_u8 = (sym * 255).astype(np.uint8)
        named: list[tuple[str, str, np.ndarray]] = [
            ("combi", f"combi_{min_count}of4{suffix}", iso_u8),
            ("symmetry_map", f"symmetry_map_{min_count}of4{suffix}", sym_u8),
        ]
        if min_count == 2:
            named.extend(
                (
                    ("combi", f"combi{suffix}", iso_u8),
                    ("symmetry_map", f"symmetry_map{suffix}", sym_u8),
                )
            )
            ortho_iso, ortho_sym = symmetry_signal_orthogonal_2of4(flat_stack)
            ortho_iso_u8 = scale_combi_uint8(ortho_iso)
            ortho_sym_u8 = (ortho_sym * 255).astype(np.uint8)
            named.extend(
                (
                    ("combi", f"combi_2of4_ortho{suffix}", ortho_iso_u8),
                    ("symmetry_map", f"symmetry_map_2of4_ortho{suffix}", ortho_sym_u8),
                )
            )
        for kind, filename, array in named:
            out_path = kind_output_dir(dest, kind) / filename
            save_same_resolution(out_path, array, paths[0])
            outputs.append(out_path)
        variants.append(
            {
                "min_count": min_count,
                "outputs": [
                    str(kind_output_dir(dest, "combi") / f"combi_{min_count}of4{suffix}"),
                    str(
                        kind_output_dir(dest, "symmetry_map")
                        / f"symmetry_map_{min_count}of4{suffix}"
                    ),
                ],
            }
        )

    height, width = sorted_stack.shape[1], sorted_stack.shape[2]
    return {
        "folder": folder,
        "output_dir": dest,
        "nsew": nsew,
        "width": width,
        "height": height,
        "outputs": outputs,
        "variants": variants,
    }


def is_uniform_image(path: str | Path) -> bool:
    """True for a Groundup ``all`` shot or a file named ``all`` / ``uniform``."""
    file_path = Path(path)
    if is_combi_tile(file_path.name):
        return False
    if lighting_role(file_path.name) == "all":
        return True
    return file_path.stem.lower() in {"all", "uniform"}


def uniform_image(folder: str | Path) -> Path | None:
    """The uniform-lighting image in ``folder``, if there is one."""
    folder = Path(folder)
    chosen: Path | None = None
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if not is_uniform_image(path):
            continue
        if chosen is not None:
            raise FileExistsError(
                f"Multiple uniform images in {folder}: {chosen.name} and {path.name}"
            )
        chosen = path
    return chosen


def _combi_2of4_uint8(paths: list[Path]) -> np.ndarray:
    sorted_stack, sum_val, _flat = load_flattened_nsew(*paths)
    isotropic, _sym = symmetry_signal_4way(sorted_stack, sum_val, 2)
    return scale_combi_uint8(isotropic)


def write_combi_2of4(folder: str | Path) -> Path:
    """Write the 2-of-4 combi into ``folder`` itself and return that path."""
    folder = Path(folder)
    nsew = find_nsew_images(folder)
    paths = [nsew[d] for d in DIRECTIONS]
    out = folder / f"combi_2of4{output_suffix(paths[0])}"
    save_same_resolution(out, _combi_2of4_uint8(paths), paths[0])
    return out


def positional_direction(path: str | Path) -> tuple[int, int, str] | None:
    """``(row, col, direction)`` for a capture named ``(1,3)_N.png``."""
    file_path = Path(path)
    if file_path.suffix.lower() not in IMAGE_EXTS:
        return None
    match = _POSITIONAL_DIRECTION.fullmatch(file_path.stem)
    if match is None:
        return None
    return (
        int(match.group("row")),
        int(match.group("col")),
        match.group("direction").upper(),
    )


def positional_cell_sides(tile_path: str | Path) -> dict[str, Path]:
    """N, S, E, and W files for ``(row,col).png`` or ``(row,col)_N.png``.

    Missing sides are left out. An unrelated filename returns an empty dict.
    """
    path = Path(tile_path)
    parsed = positional_direction(path)
    if parsed is not None:
        row, col, _direction = parsed
    else:
        match = _POSITIONAL_COMBI.fullmatch(path.stem)
        if match is None:
            return {}
        row, col = int(match.group("row")), int(match.group("col"))
    found: dict[str, Path] = {}
    suffixes = (path.suffix.lower(), *(ext for ext in IMAGE_EXTS if ext != path.suffix.lower()))
    for direction in DIRECTIONS:
        for suffix in suffixes:
            candidate = path.parent / f"({int(row)},{int(col)})_{direction}{suffix}"
            if candidate.is_file():
                found[direction] = candidate
                break
    return found


def positional_combi_path(
    folder: str | Path,
    row: int,
    col: int,
    by_direction: dict[str, Path],
) -> Path:
    """``(row,col).png`` written beside that cell's side images."""
    reference = by_direction[DIRECTIONS[0]]
    return Path(folder) / f"({int(row)},{int(col)}){output_suffix(reference)}"


def combi_is_current(path: str | Path, sources: list[Path]) -> bool:
    """True when ``path`` exists, opens, and is newer than every source image."""
    file_path = Path(path)
    if not file_path.is_file():
        return False
    try:
        built = file_path.stat().st_mtime
        if any(built < source.stat().st_mtime for source in sources):
            return False
        with Image.open(file_path) as image:
            image.verify()
    except (OSError, ValueError, SyntaxError):
        return False
    return True


def write_positional_combi(
    folder: str | Path,
    row: int,
    col: int,
    by_direction: dict[str, Path],
) -> Path:
    """Write ``(row,col).png`` from that cell's four side images."""
    folder = Path(folder)
    paths = [by_direction[direction] for direction in DIRECTIONS]
    out = positional_combi_path(folder, row, col, by_direction)
    save_same_resolution(out, _combi_2of4_uint8(paths), paths[0])
    return out


def _positional_groups(paths: list[Path]) -> tuple[list[Path], dict[tuple[Path, int, int], dict[str, Path]]]:
    """Split ``(row,col)_N`` shots out of the files detection would otherwise load."""
    groups: dict[tuple[Path, int, int], dict[str, Path]] = {}
    rest: list[Path] = []
    for path in paths:
        parsed = positional_direction(path)
        if parsed is None:
            rest.append(path)
            continue
        row, col, direction = parsed
        slot = groups.setdefault((path.parent.resolve(), row, col), {})
        if direction in slot:
            raise FileExistsError(
                f"Multiple {direction} images for ({row},{col}) in {path.parent}: "
                f"{slot[direction].name} and {path.name}"
            )
        slot[direction] = path
    return rest, groups


def prepare_detection_paths(
    paths: list[Path],
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    """Combine directional shots, then return the images detection should load.

    A folder with N, S, E, and W (or Groundup quads) gets ``combi_2of4`` written
    beside those files. Detection runs on the combi as well as the original
    shots, including the uniform ``all`` image when the folder has one.

    A map of ``(row,col)_N`` shots is different: each cell's four sides are
    combined into ``(row,col).png`` first. Detection then loads that combi
    and the four side files. A cell missing a side is left out. Folders
    without a full set of plain N/S/E/W names are otherwise unchanged.
    """
    rest, groups = _positional_groups(paths)
    extras: list[Path] = []
    side_files: list[Path] = []
    pending = [
        (folder, row, col, found)
        for (folder, row, col), found in groups.items()
        if all(direction in found for direction in DIRECTIONS)
    ]
    n_cells = len(pending)
    for index, (folder, row, col, found) in enumerate(pending, start=1):
        # The count has to change on every cell. A repeat of 0/N leaves the bar stuck.
        if progress_cb is not None:
            progress_cb(index, n_cells, f"Combining ({row},{col})")
        sources = [found[direction] for direction in DIRECTIONS]
        out = positional_combi_path(folder, row, col, found)
        side_files.extend(found[direction] for direction in DIRECTIONS)
        if combi_is_current(out, sources):
            extras.append(out)
            continue
        extras.append(write_positional_combi(folder, row, col, found))

    folders = {path.parent.resolve() for path in rest}
    for folder in folders:
        if len(list_direction_images(folder)) < 4:
            continue
        if progress_cb is not None:
            progress_cb(0, 1, f"Combining {folder.name}")
        extras.append(write_combi_2of4(folder))
        if progress_cb is not None:
            progress_cb(1, 1, f"Combining {folder.name}")
    kept: list[Path] = []
    seen: set[Path] = set()
    for path in list(rest) + extras + side_files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        kept.append(path)
    return kept
