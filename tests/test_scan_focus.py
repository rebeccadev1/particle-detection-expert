"""When scan and capture refocuses, and the sharpness floor."""

from src.capture.scan_capture import (
    FOCUS_EVERY_STEPS,
    MIN_SCAN_SHARPNESS,
    needs_another_focus,
    refocus_due,
)
from src.capture.stage import scan_stops


def test_refocus_lands_on_every_seventh_step() -> None:
    assert FOCUS_EVERY_STEPS == 7
    assert [step for step in range(0, 22) if refocus_due(step)] == [7, 14, 21]


def test_serpentine_refocuses_after_seven_moves() -> None:
    stops = scan_stops(8, 1, 1.0)
    taken = 0
    focused = []
    for _row, _col, axis, _delta in stops:
        if axis is None:
            continue
        taken += 1
        if refocus_due(taken):
            focused.append(taken)
    assert focused == [7]


def test_soft_focus_runs_again() -> None:
    assert MIN_SCAN_SHARPNESS == 900
    assert needs_another_focus(899.9)
    assert needs_another_focus(0)
    assert not needs_another_focus(900)
    assert not needs_another_focus(1200.4)
    assert not needs_another_focus(100, stopped=True)
