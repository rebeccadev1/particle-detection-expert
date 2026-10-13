"""Autofocus hill climb. No printer and no camera."""

import math

import cv2
import numpy as np
import pytest

from src.capture.focus import focus_search, sharpness


def test_sharp_image_scores_above_a_blurred_one() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    assert sharpness(sharp) > sharpness(blurred)


def test_rgb_frame_is_scored() -> None:
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
    assert sharpness(sharp) > sharpness(blurred)


def _stage(score_at):
    position = {"z": 0.0}
    seen: list[float] = []

    def measure() -> float:
        return score_at(position["z"])

    def move(delta: float) -> None:
        position["z"] += delta
        seen.append(position["z"])

    return position, seen, measure, move


def test_search_finds_a_peak_within_one_fine_step() -> None:
    peak = 0.22
    sigma = 0.08

    def score_at(z: float) -> float:
        return math.exp(-((z - peak) / sigma) ** 2)

    position, seen, measure, move = _stage(score_at)
    result = focus_search(
        measure,
        move,
        max_travel_mm=1.0,
        coarse_mm=0.05,
        fine_mm=0.01,
        min_gain=0.02,
    )
    assert abs(result.offset_mm - peak) <= 0.01 + 1e-9
    assert result.offset_mm == pytest.approx(position["z"])
    assert max(abs(z) for z in seen) <= 1.0 + 1e-6
    assert not result.stopped


def test_search_finds_a_downward_peak() -> None:
    peak = -0.12
    sigma = 0.08

    def score_at(z: float) -> float:
        return math.exp(-((z - peak) / sigma) ** 2)

    position, _seen, measure, move = _stage(score_at)
    result = focus_search(
        measure,
        move,
        max_travel_mm=1.0,
        coarse_mm=0.05,
        fine_mm=0.01,
        min_gain=0.02,
    )
    assert abs(result.offset_mm - peak) <= 0.01 + 1e-9
    assert result.offset_mm == pytest.approx(position["z"])


def test_search_never_passes_max_travel() -> None:
    def score_at(z: float) -> float:
        return math.exp(z * 20.0)

    position, seen, measure, move = _stage(score_at)
    result = focus_search(
        measure,
        move,
        max_travel_mm=0.1,
        coarse_mm=0.05,
        fine_mm=0.01,
        min_gain=0.02,
    )
    assert seen
    assert max(abs(z) for z in seen) <= 0.1 + 1e-6
    assert result.offset_mm == pytest.approx(0.1)
    assert position["z"] == pytest.approx(0.1)


def test_stop_finishes_on_the_best_height_so_far() -> None:
    def score_at(z: float) -> float:
        return math.exp(z * 40.0)

    position, seen, measure, move = _stage(score_at)

    def should_stop() -> bool:
        return len(seen) >= 1

    result = focus_search(
        measure,
        move,
        max_travel_mm=1.0,
        coarse_mm=0.05,
        fine_mm=0.01,
        min_gain=0.02,
        should_stop=should_stop,
    )
    assert result.stopped
    assert result.offset_mm == pytest.approx(0.05)
    assert position["z"] == pytest.approx(0.05)
    assert max(abs(z) for z in seen) == pytest.approx(0.05)
