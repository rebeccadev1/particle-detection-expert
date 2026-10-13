"""Serpentine stage program used by the Capture tab."""

from pathlib import Path

import pytest

from src.capture.camera_automation_v2 import MOTION_PAUSE_S, NSEW_GAP_S, SETTLE_S
from src.capture.stage import (
    PAUSE_MARGIN_MS,
    build_relative_move_gcode,
    build_scan_gcode,
    build_z_move_gcode,
    move_axis,
    move_z,
    next_scan_row,
    pause_plan,
    scan_stops,
    scan_summary,
)


def test_old_scan_40x60_still_builds() -> None:
    path = Path("/Users/rebeccajekel/Desktop/scan_40x60.gcode")
    if not path.is_file():
        pytest.skip("reference gcode is not on this machine")
    assert build_scan_gcode(step_mm=6.3, columns=7, rows=10, dwell_ms=1000) == path.read_text()


def test_standard_grid_covers_40_by_60_with_10_percent_overlap() -> None:
    from src.capture.stage import DEFAULT_COLUMNS, DEFAULT_ROWS, DEFAULT_STEP_MM, FOV_MM

    assert DEFAULT_STEP_MM == pytest.approx(6.174)
    assert DEFAULT_COLUMNS == 7
    assert DEFAULT_ROWS == 10
    covered_x = FOV_MM + (DEFAULT_COLUMNS - 1) * DEFAULT_STEP_MM
    covered_y = FOV_MM + (DEFAULT_ROWS - 1) * DEFAULT_STEP_MM
    assert covered_x >= 40
    assert covered_y >= 60
    assert covered_x == pytest.approx(43.904)
    assert covered_y == pytest.approx(62.426)


def test_one_row_has_no_y_move() -> None:
    text = build_scan_gcode(step_mm=1, columns=3, rows=1, feed=100, dwell_ms=50)
    assert "Y" not in text
    assert text.splitlines() == [
        "G21",
        "G91",
        "",
        "; Start at top-left position",
        "G4 P50",
        "G1 X1 F100",
        "G4 P50",
        "G1 X1 F100",
        "G4 P50",
        "",
        "G90",
    ]


def test_second_row_reverses_x() -> None:
    text = build_scan_gcode(step_mm=2.5, columns=2, rows=2, feed=800, dwell_ms=10)
    assert "G1 X2.5 F800" in text
    assert "G1 Y-2.5 F800" in text
    assert "G1 X-2.5 F800" in text


def test_summary_counts_stops() -> None:
    summary = scan_summary(6.3, 7, 10)
    assert summary["columns"] == 7
    assert summary["rows"] == 10
    assert summary["stops"] == 70
    assert summary["x_mm"] == pytest.approx(37.8)
    assert summary["y_mm"] == pytest.approx(56.7)


def test_uniform_pause_fits_one_frame() -> None:
    plan = pause_plan(100.0, "Uniform")
    assert plan["frames"] == 1
    assert plan["shutter_ms"] == pytest.approx(25.0)
    assert plan["motion_ms"] == pytest.approx(MOTION_PAUSE_S * 1000)
    assert plan["dwell_ms"] == int(
        MOTION_PAUSE_S * 1000 + SETTLE_S * 1000 + 25 + PAUSE_MARGIN_MS
    )


def test_nsew_pause_fits_four_frames() -> None:
    plan = pause_plan(100.0, "NSEW")
    assert plan["frames"] == 4
    assert plan["shutter_ms"] == pytest.approx(100.0)
    assert plan["gaps"] == 3
    assert plan["gap_ms"] == pytest.approx(NSEW_GAP_S * 1000)
    assert plan["dwell_ms"] == int(
        MOTION_PAUSE_S * 1000
        + 4 * (SETTLE_S * 1000 + 100 + PAUSE_MARGIN_MS)
        + 3 * NSEW_GAP_S * 1000
    )
    assert plan["dwell_ms"] > pause_plan(100.0, "Uniform")["dwell_ms"]


def test_rejects_a_zero_step() -> None:
    with pytest.raises(ValueError):
        build_scan_gcode(step_mm=0)


def test_z_move_is_relative_and_waits() -> None:
    text = build_z_move_gcode(-0.05, 300)
    assert text.splitlines() == ["G21", "G91", "G1 Z-0.05 F300", "M400", "G90"]


def test_z_move_needs_a_printer() -> None:
    with pytest.raises(RuntimeError, match="Connect the printer"):
        move_z(0.1, 300)


def test_xy_jog_is_one_relative_millimetre() -> None:
    assert build_relative_move_gcode("X", -1, 1000).splitlines() == [
        "G21",
        "G91",
        "G1 X-1 F1000",
        "M400",
        "G90",
    ]
    assert build_relative_move_gcode("Y", 1, 1000).splitlines() == [
        "G21",
        "G91",
        "G1 Y1 F1000",
        "M400",
        "G90",
    ]


def test_scan_stops_follow_the_serpentine() -> None:
    stops = scan_stops(3, 2, 1.5)
    assert [(row, col) for row, col, _axis, _delta in stops] == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 2),
        (2, 1),
    ]
    assert stops[0][2] is None
    assert stops[1][2:] == ("X", 1.5)
    assert stops[3][2:] == ("Y", -1.5)
    assert stops[4][2:] == ("X", -1.5)


def test_next_scan_starts_below_existing_rows() -> None:
    assert next_scan_row([]) == 1
    assert next_scan_row([(1, 1), (1, 7), (2, 3)]) == 3


def test_xy_jog_needs_a_printer() -> None:
    with pytest.raises(RuntimeError, match="Connect the printer"):
        move_axis("X", 1, 1000)
