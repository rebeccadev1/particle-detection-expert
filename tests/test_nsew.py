"""Tests for N/S/E/W folder discovery and combined output size."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.main_pipeline.nsew import (
    MIN_COUNTS,
    combine_nsew_folder,
    find_nsew_images,
    list_direction_images,
    nsew_output_dir,
    prepare_detection_paths,
    scale_combi_uint8,
    symmetry_signal_4way,
    symmetry_signal_orthogonal_2of4,
    write_combi_2of4,
)
from src.verification.queue import detection_key


def _write_gray(path: Path, value: int, size: tuple[int, int] = (24, 32), dpi=(254, 254)) -> None:
    array = np.full(size, value, dtype=np.uint8)
    Image.fromarray(array).save(path, dpi=dpi)


def test_find_nsew_images_matches_stems(tmp_path: Path) -> None:
    _write_gray(tmp_path / "N.bmp", 10)
    _write_gray(tmp_path / "S.bmp", 20)
    _write_gray(tmp_path / "E.png", 30)
    _write_gray(tmp_path / "W.png", 40)
    found = find_nsew_images(tmp_path)
    assert found["N"].name == "N.bmp"
    assert found["S"].name == "S.bmp"
    assert found["E"].name == "E.png"
    assert found["W"].name == "W.png"


def test_find_nsew_images_accepts_groundup_quad_shots(tmp_path: Path) -> None:
    _write_gray(tmp_path / "wafer_shot_0_quad0_exp200000.png", 10)
    _write_gray(tmp_path / "wafer_shot_1_quad1_exp200000.png", 20)
    _write_gray(tmp_path / "wafer_shot_2_quad2_exp200000.png", 30)
    _write_gray(tmp_path / "wafer_shot_3_quad3_exp200000.png", 40)
    _write_gray(tmp_path / "wafer_shot_4_all_exp200000.png", 50)
    found = find_nsew_images(tmp_path)
    assert found["N"].name.startswith("wafer_shot_0_quad0")
    assert found["E"].name.startswith("wafer_shot_1_quad1")
    assert found["S"].name.startswith("wafer_shot_2_quad2")
    assert found["W"].name.startswith("wafer_shot_3_quad3")


def test_list_direction_images_allows_partial_nwes(tmp_path: Path) -> None:
    _write_gray(tmp_path / "N.bmp", 10)
    _write_gray(tmp_path / "W.bmp", 20)
    found = list_direction_images(tmp_path)
    assert set(found) == {"N", "W"}


def test_combine_writes_same_size_in_folder(tmp_path: Path) -> None:
    height, width = 24, 32
    for name, value in (("N", 40), ("S", 50), ("E", 60), ("W", 70)):
        _write_gray(tmp_path / f"{name}.bmp", value, size=(height, width))
    result = combine_nsew_folder(tmp_path)
    dest = nsew_output_dir(tmp_path)
    assert result["output_dir"] == dest
    assert dest.name == f"Output {tmp_path.name}"
    assert result["width"] == width
    assert result["height"] == height
    for min_count in MIN_COUNTS:
        for kind, folder in (
            ("combi", "Combi"),
            ("symmetry_map", "Symmetry"),
        ):
            path = dest / folder / f"{kind}_{min_count}of4.bmp"
            assert path.is_file()
            with Image.open(path) as image:
                assert image.size == (width, height)
                # BMP inputs often have dpi (0,0); we must not copy that.
                dpi = image.info.get("dpi")
                if dpi is not None:
                    assert dpi[0] > 0 and dpi[1] > 0
    assert not (dest / "Particles only").exists()
    for folder, name in (
        ("Combi", "combi.bmp"),
        ("Symmetry", "symmetry_map.bmp"),
    ):
        assert (dest / folder / name).is_file()
        assert not (tmp_path / name).is_file()
        assert not (dest / name).is_file()


def test_combine_matches_png_inputs(tmp_path: Path) -> None:
    height, width = 16, 20
    for name, value in (("N", 40), ("S", 50), ("E", 60), ("W", 70)):
        array = np.full((height, width), value, dtype=np.uint8)
        Image.fromarray(array).save(tmp_path / f"{name}.png", dpi=(254, 254))
    result = combine_nsew_folder(tmp_path)
    dest = nsew_output_dir(tmp_path)
    assert result["output_dir"] == dest
    path = dest / "Combi" / "combi_2of4.png"
    assert path.is_file()
    with Image.open(path) as image:
        assert image.size == (width, height)
        assert image.info.get("dpi") == (254.0, 254.0)


def test_nsew_output_dir_uses_workspace_outputs(tmp_path: Path, monkeypatch) -> None:
    from src.main_pipeline import nsew as nsew_mod

    inputs = tmp_path / "Inputs"
    outputs = tmp_path / "Outputs"
    folder = inputs / "Groundup v3"
    folder.mkdir(parents=True)
    monkeypatch.setattr(nsew_mod, "DEFAULT_INPUT_DIR", inputs)
    monkeypatch.setattr(
        nsew_mod, "resolve_output_dir", lambda name: (outputs / name).resolve()
    )
    dest = nsew_mod.nsew_output_dir(folder)
    assert dest == (outputs / "Output Groundup v3").resolve()


def test_min_count_uses_matching_rank() -> None:
    dark = np.zeros((3, 3), dtype=np.float32)
    bright = np.ones((3, 3), dtype=np.float32)
    stack = np.stack([bright, bright, dark, dark], axis=0)
    sorted_stack = np.sort(stack, axis=0)
    sum_val = np.sum(stack, axis=0)
    iso2, _ = symmetry_signal_4way(sorted_stack, sum_val, 2)
    iso3, _ = symmetry_signal_4way(sorted_stack, sum_val, 3)
    iso4, _ = symmetry_signal_4way(sorted_stack, sum_val, 4)
    assert float(iso2.mean()) > 0.5
    assert float(iso3.mean()) == 0.0
    assert float(iso4.mean()) == 0.0


def test_orthogonal_2of4_rejects_opposite_pair() -> None:
    dark = np.zeros((2, 2), dtype=np.float32)
    bright = np.ones((2, 2), dtype=np.float32)
    # DIRECTIONS order: N, S, E, W — only opposite N–S bright.
    opposite = np.stack([bright, bright, dark, dark], axis=0)
    ortho_iso, _ = symmetry_signal_orthogonal_2of4(opposite)
    classic = np.sort(opposite, axis=0)
    classic_iso, _ = symmetry_signal_4way(classic, np.sum(opposite, axis=0), 2)
    assert float(classic_iso.mean()) > 0.5
    assert float(ortho_iso.mean()) < 0.05

    # Adjacent N–E bright.
    adjacent = np.stack([bright, dark, bright, dark], axis=0)
    adjacent_iso, _ = symmetry_signal_orthogonal_2of4(adjacent)
    assert float(adjacent_iso.mean()) > 0.5


def test_scale_combi_uint8_keeps_a_hot_pixel_from_crushing_the_tile() -> None:
    image = np.full((200, 200), 2.0, dtype=np.float32)
    image[50:60, 50:60] = 40.0
    image[0, 0] = 10_000.0
    scaled = scale_combi_uint8(image)
    assert scaled[0, 0] == 255
    assert scaled[55, 55] == 255
    assert int(scaled[10, 10]) < 30


def test_write_combi_2of4_lands_in_the_input_folder(tmp_path: Path) -> None:
    for name, value in (("N", 40), ("S", 50), ("E", 60), ("W", 70)):
        _write_gray(tmp_path / f"{name}.png", value)
    out = write_combi_2of4(tmp_path)
    assert out.parent == tmp_path
    assert out.name == "combi_2of4.png"
    assert out.is_file()


def test_prepare_detection_paths_keeps_combi_and_uniform(tmp_path: Path) -> None:
    folder = tmp_path / "v4"
    folder.mkdir()
    for name in (
        "wafer_shot_0_quad0_exp1.png",
        "wafer_shot_1_quad1_exp1.png",
        "wafer_shot_2_quad2_exp1.png",
        "wafer_shot_3_quad3_exp1.png",
        "wafer_shot_4_all_exp1.png",
    ):
        _write_gray(folder / name, 40)
    other = tmp_path / "plain.png"
    _write_gray(other, 10)
    paths = prepare_detection_paths([*(folder.iterdir()), other])
    names = sorted(path.name for path in paths)
    assert names == [
        "combi_2of4.png",
        "plain.png",
        "wafer_shot_0_quad0_exp1.png",
        "wafer_shot_1_quad1_exp1.png",
        "wafer_shot_2_quad2_exp1.png",
        "wafer_shot_3_quad3_exp1.png",
        "wafer_shot_4_all_exp1.png",
    ]
    assert (folder / "combi_2of4.png").is_file()


def test_prepare_detection_paths_leaves_a_partial_set(tmp_path: Path) -> None:
    _write_gray(tmp_path / "N.png", 10)
    _write_gray(tmp_path / "S.png", 20)
    paths = prepare_detection_paths(list(tmp_path.iterdir()))
    assert sorted(path.name for path in paths) == ["N.png", "S.png"]
    assert not (tmp_path / "combi_2of4.png").exists()


def test_combi_beside_quads_uses_light_keys(tmp_path: Path) -> None:
    folder = tmp_path / "v4"
    folder.mkdir()
    _write_gray(folder / "wafer_shot_0_quad0_exp1.png", 10)
    combi = folder / "combi_2of4.png"
    _write_gray(combi, 20)
    key = detection_key({"source_tile": str(combi), "x_global": 10.2, "y_global": 20.4})
    assert key == "v4_LIGHT_10_20"


def test_combi_beside_letters_uses_nsew_keys(tmp_path: Path) -> None:
    folder = tmp_path / "v3"
    folder.mkdir()
    _write_gray(folder / "N.png", 10)
    combi = folder / "combi_2of4.png"
    _write_gray(combi, 20)
    key = detection_key({"source_tile": str(combi), "x_global": 10.2, "y_global": 20.4})
    assert key == "v3_NSEW_10_20"
