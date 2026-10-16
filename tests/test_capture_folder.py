"""Capture folders are named by grid location, inside a daily run folder."""

from datetime import date

import json

from src.capture.camera_automation_v2 import (
    CAPTURES_DIR,
    allocate_capture_folder,
    allocate_run_folder,
    folder_location,
    image_filename,
    next_capture_location,
    write_manifest,
)


def test_locations_follow_the_serpentine() -> None:
    assert next_capture_location([], 7) == (1, 1)
    assert next_capture_location([(1, 1)], 7) == (1, 2)
    assert next_capture_location([(1, 6)], 7) == (1, 7)
    assert next_capture_location([(1, 7)], 7) == (2, 7)
    assert next_capture_location([(2, 7)], 7) == (2, 6)
    assert next_capture_location([(2, 2)], 7) == (2, 1)
    assert next_capture_location([(2, 1)], 7) == (3, 1)
    assert next_capture_location([(3, 1)], 7) == (3, 2)
    assert next_capture_location([(1, 1)], 1) == (2, 1)
    assert next_capture_location([(2, 1)], 1) == (3, 1)


def test_latest_stop_wins_when_folders_already_exist() -> None:
    assert next_capture_location([(1, 1), (1, 2), (1, 3)], 3) == (2, 3)
    assert next_capture_location([(2, 3), (2, 2), (1, 3)], 3) == (2, 1)


def test_allocate_capture_folder_uses_location_names(tmp_path) -> None:
    first = allocate_capture_folder(tmp_path, columns=2)
    second = allocate_capture_folder(tmp_path, columns=2)
    third = allocate_capture_folder(tmp_path, columns=2)
    fourth = allocate_capture_folder(tmp_path, columns=2)
    assert first.name == "(1,1)"
    assert second.name == "(1,2)"
    assert third.name == "(2,2)"
    assert fourth.name == "(2,1)"
    assert folder_location(third) == (2, 2)


def test_old_run_folders_are_not_locations(tmp_path) -> None:
    (tmp_path / "run_001").mkdir()
    folder = allocate_capture_folder(tmp_path, columns=7)
    assert folder.name == "(1,1)"


def test_picture_names_are_the_grid_position() -> None:
    assert image_filename(1, 1) == "(1,1).png"
    assert image_filename(2, 7, "N") == "(2,7)_N.png"


def test_manifest_lives_in_the_run_folder(tmp_path) -> None:
    write_manifest(
        tmp_path,
        {"mode": "uniform", "row": 1, "col": 1, "files": ["(1,1).png"], "shutter_ms": 25.0},
    )
    write_manifest(
        tmp_path,
        {
            "mode": "nsew",
            "row": 1,
            "col": 2,
            "files": ["(1,2)_N.png"],
            "shutter_ms": 100.0,
        },
    )
    write_manifest(
        tmp_path,
        {"mode": "nsew", "row": 1, "col": 2, "files": ["(1,2)_E.png"], "shutter_ms": 100.0},
    )
    data = json.loads((tmp_path / "capture.json").read_text())
    assert data["stops"] == [
        {"row": 1, "col": 1, "files": ["(1,1).png"]},
        {"row": 1, "col": 2, "files": ["(1,2)_N.png", "(1,2)_E.png"]},
    ]
    assert data["shutter_ms"] == 100.0


def test_captures_sit_beside_inputs(tmp_path) -> None:
    assert CAPTURES_DIR.name == "captures"
    assert (CAPTURES_DIR.parent / "Inputs").is_dir()
    assert (CAPTURES_DIR.parent / "Outputs").is_dir()


def test_run_folders_count_only_that_day(tmp_path) -> None:
    today = date(2026, 10, 9)
    yesterday = date(2026, 10, 8)
    (tmp_path / "v3 2026-10-08").mkdir()
    (tmp_path / "(1,1)").mkdir()
    first = allocate_run_folder(tmp_path, on=today)
    second = allocate_run_folder(tmp_path, on=today)
    earlier = allocate_run_folder(tmp_path, on=yesterday)
    assert first.name == "v1 2026-10-09"
    assert second.name == "v2 2026-10-09"
    assert earlier.name == "v4 2026-10-08"
