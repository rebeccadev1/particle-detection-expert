"""When scan and capture refocuses, and the sharpness floor."""

from src.capture.scan_capture import (
    FOCUS_EVERY_STEPS,
    FOCUS_MIN_RATIO,
    needs_another_focus,
    refocus_due,
)
from src.capture.stage import scan_stops


def test_refocus_lands_on_every_seventh_step() -> None:
    assert FOCUS_EVERY_STEPS == 7
    assert [step for step in range(0, 22) if refocus_due(step)] == [7, 14, 21]
    assert [step for step in range(0, 22) if refocus_due(step, enabled=False)] == []


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
    assert FOCUS_MIN_RATIO == 0.8
    assert needs_another_focus(800, starting=1000)
    assert needs_another_focus(0, starting=1000)
    assert not needs_another_focus(800.1, starting=1000)
    assert not needs_another_focus(1200.4, starting=1000)
    assert not needs_another_focus(400, starting=400)
    assert not needs_another_focus(100, stopped=True, starting=1000)
