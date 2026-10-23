"""Tests for the per-tile inspect overlay (no Streamlit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.verification.crops import local_px_to_global_nm
from src.verification.inspect import (
    OverlayView,
    cell_bounds,
    MISSED_PARTICLE_COLOR,
    MISS_SOURCE,
    pointer_miss_rows,
    circles_from_rows,
    detections_on_this_file,
    display_xy_to_local,
    fresh_inspect_click,
    expand_submap_lighting_names,
    extra_grid_placements,
    last_run_class_counts,
    last_run_class_counts_by_tile,
    std_of_tile_scores,
    subfolder_score_rows,
    local_xy_on_tile,
    missed_particle_record,
    preview_click_crop,
    undetected_size_floor_nm,
)
from src.overall.io.placement import TilePlacement


def test_cell_bounds_splits_tile_into_equal_grid() -> None:
    y0, x0, y1, x1 = cell_bounds(90, 120, 1, 2, grid=3)
    assert (y0, x0) == (30, 80)
    assert (y1, x1) == (60, 120)


def test_detections_on_this_file_stays_on_the_source_quad() -> None:
    table = pd.DataFrame(
        {
            "source_tile": ["v4/wafer_shot_0_quad0_exp200000.png"],
            "size": [4.0],
            "key": ["v4_LIGHT_4_4"],
            "nsew_dirs": ["QUAD0,QUAD1"],
        }
    )
    on_quad0 = detections_on_this_file(table, "v4/wafer_shot_0_quad0_exp200000.png")
    on_quad1 = detections_on_this_file(table, "v4/wafer_shot_1_quad1_exp200000.png")
    on_quad2 = detections_on_this_file(table, "v4/wafer_shot_2_quad2_exp200000.png")
    on_other = detections_on_this_file(table, "v5/wafer_shot_1_quad1_exp200000.png")
    assert list(on_quad0["key"]) == ["v4_LIGHT_4_4"]
    assert on_quad1.empty
    assert on_quad2.empty
    assert on_other.empty
    both = last_run_class_counts(
        table,
        None,
        tile_names=["v4/wafer_shot_0_quad0_exp200000.png", "v4/wafer_shot_1_quad1_exp200000.png"],
    )
    assert both["n_detected"] == 1
    assert last_run_class_counts(
        table, None, tile_names=["v4/wafer_shot_1_quad1_exp200000.png"]
    )["n_detected"] == 0


def test_pointer_miss_stays_on_the_tile_it_was_marked() -> None:
    labels = pd.DataFrame(
        {
            "key": ["miss-a", "found", "other-set"],
            "source_tile": [
                "v4/wafer_shot_0_quad0_exp200000.png",
                "v4/wafer_shot_1_quad1_exp200000.png",
                "v5/wafer_shot_0_quad0_exp200000.png",
            ],
            "label": ["particle", "particle", "particle"],
            "source_csv": [MISS_SOURCE, MISS_SOURCE, MISS_SOURCE],
            "x_global": [1.0, 2.0, 3.0],
            "y_global": [1.0, 2.0, 3.0],
            "size": [20_000.0, 20_000.0, 20_000.0],
        }
    )
    detections = pd.DataFrame(
        {
            "source_tile": ["v4/wafer_shot_1_quad1_exp200000.png"],
            "key": ["found"],
            "x_global": [2.0],
            "y_global": [2.0],
            "size": [20_000.0],
        }
    )
    on_quad0 = pointer_miss_rows(
        labels, detections, "v4/wafer_shot_0_quad0_exp200000.png"
    )
    on_quad1 = pointer_miss_rows(
        labels, detections, "v4/wafer_shot_1_quad1_exp200000.png"
    )
    on_combi = pointer_miss_rows(labels, detections, "v4/combi_2of4.png")
    on_uniform = pointer_miss_rows(
        labels, detections, "v4/wafer_shot_4_all_exp200000.png"
    )
    assert list(on_quad0["key"]) == ["miss-a"]
    assert bool(on_quad0["missed"].iloc[0])
    assert on_quad1.empty
    assert on_combi.empty
    assert on_uniform.empty


def test_grid_inspect_marks_stay_in_their_folder() -> None:
    labels = pd.DataFrame(
        {
            "key": ["here", "other-wafer", "no-folder"],
            "source_tile": [
                "Groundup/wafer-a/(1,1).png",
                "Groundup/wafer-b/(1,1).png",
                "(1,1).png",
            ],
            "label": ["particle", "particle", "particle"],
            "source_csv": [MISS_SOURCE, MISS_SOURCE, MISS_SOURCE],
            "x_global": [1.0, 2.0, 3.0],
            "y_global": [1.0, 2.0, 3.0],
            "size": [20_000.0, 20_000.0, 20_000.0],
        }
    )
    detections = pd.DataFrame(
        columns=["source_tile", "key", "x_global", "y_global", "size"]
    )
    shown = pointer_miss_rows(
        labels, detections, "(1,1).png", run_dir="Groundup/wafer-a"
    )
    assert list(shown["key"]) == ["here"]


def test_particle_detected_on_combi_and_uniform_is_marked_on_both() -> None:
    from src.verification.inspect import detections_on_review_picture

    table = pd.DataFrame(
        {
            "source_tile": [
                "v6/combi_2of4.png",
                "v6/wafer_shot_0_quad0_exp200000.png",
                "v6/wafer_shot_1_quad1_exp200000.png",
                "v5/wafer_shot_4_all_exp200000.png",
            ],
            "nsew_dirs": ["COMBI", "ALL,QUAD0", "QUAD1", "ALL"],
            "key": ["combi-hit", "both-uniform", "quad-only", "other-field"],
            "x_global": [1.0, 2.0, 3.0, 2.0],
            "y_global": [1.0, 2.0, 3.0, 2.0],
            "size": [10.0, 10.0, 10.0, 10.0],
        }
    )
    combi = detections_on_review_picture(table, "v6/combi_2of4.png")
    uniform = detections_on_review_picture(table, "v6/wafer_shot_4_all_exp200000.png")
    quad = detections_on_review_picture(table, "v6/wafer_shot_1_quad1_exp200000.png")
    assert set(combi["key"]) == {"combi-hit"}
    assert set(uniform["key"]) == {"both-uniform"}
    assert list(quad["key"]) == ["quad-only"]


def test_expand_submap_lighting_names_includes_combi() -> None:
    names = expand_submap_lighting_names(
        ["v4/wafer_shot_0_quad0_exp200000.png"],
        [
            "v4/wafer_shot_0_quad0_exp200000.png",
            "v4/wafer_shot_4_all_exp200000.png",
            "v4/combi_2of4.png",
            "v5/combi_2of4.png",
        ],
    )
    assert "v4/combi_2of4.png" in names
    assert "v4/wafer_shot_4_all_exp200000.png" in names
    assert "v5/combi_2of4.png" not in names


def test_last_run_class_counts_splits_real_fake_and_inspect_misses() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [10.0, 20.0, 30.0],
            "y_global": [11.0, 21.0, 31.0],
            "size": [16000.0, 17000.0, 18000.0],
            "confidence": [0.9, 0.8, 0.7],
            "source_tile": ["a.tif", "a.tif", "a.tif"],
            "key": ["real", "fake", "open"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["real", "fake", "miss", "old_miss"],
            "label": ["particle", "not_particle", "particle", "particle"],
            "source_tile": ["a.tif", "a.tif", "a.tif", "a.tif"],
            "source_csv": ["run.csv", "run.csv", "tile_inspect", "older.csv"],
        }
    )
    stats = last_run_class_counts(detections, labels, tile_names=["a.tif"])
    assert stats["n_detected"] == 3
    assert stats["n_real"] == 1
    assert stats["n_fake"] == 2
    assert stats["n_unlabeled"] == 1
    assert stats["n_undetected"] == 2
    assert stats["n_real_total"] == stats["n_real"] + stats["n_undetected"]
    assert stats["n_real_total"] == 3
    assert stats["precision"] == 1 / 3
    assert stats["recall"] == 1 / 3


def test_last_run_counts_pointer_clicks_on_the_matching_field() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [20000.0],
            "confidence": [0.9],
            "source_tile": ["v5_2of4.png"],
            "key": ["v5_2of4_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": [
                "v5_NSEW_10_11",
                "v5_NSEW_80_80",
                "v5_NSEW_90_90",
                "v6_NSEW_70_70",
                "v6_2of4_111_111",
            ],
            "label": ["particle", "particle", "particle", "particle", "particle"],
            "source_tile": ["v5/N.png", "v5/S.png", "v5/E.png", "v6/N.png", "v6_2of4.bmp"],
            "source_csv": ["tile_inspect"] * 5,
            "size": [25000.0, 30000.0, 15000.0, 40000.0, 50000.0],
            "x_global": [10.0, 80.0, 90.0, 70.0, 111.0],
            "y_global": [11.0, 80.0, 90.0, 70.0, 111.0],
        }
    )
    stats = last_run_class_counts(
        detections,
        labels,
        tile_names=["v5_2of4.png"],
        min_size_nm=20000.0,
        folder_tiles=["v5_2of4.png", "v6_2of4.png"],
    )
    assert stats["n_real"] == 1
    assert stats["n_undetected"] == 1
    assert stats["n_undetected_below"] == 1
    assert stats["n_real_total"] == 2
    both = last_run_class_counts(
        detections,
        labels,
        tile_names=["v5_2of4.png", "v6_2of4.png"],
        min_size_nm=20000.0,
    )
    assert both["n_undetected"] == 2
    assert both["n_real_total"] == 3


def test_submap_combi_does_not_inherit_letter_field_misses() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [20000.0],
            "confidence": [0.9],
            "source_tile": ["v5/combi_2of4.png"],
            "key": ["v5_LIGHT_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["v5_NSEW_80_80", "v5_2of4_90_90", "v5_LIGHT_400_400"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["v5/N.png", "v5_2of4.png", "v5/combi_2of4.png"],
            "source_csv": ["tile_inspect", "old.csv", "tile_inspect"],
            "size": [25000.0, 30000.0, 22000.0],
            "x_global": [80.0, 90.0, 400.0],
            "y_global": [80.0, 90.0, 400.0],
        }
    )
    stats = last_run_class_counts(
        detections,
        labels,
        tile_names=["v5/combi_2of4.png", "v5/wafer_shot_4_all_exp200000.png"],
        min_size_nm=20000.0,
        folder_tiles=["v5/combi_2of4.png", "v5/wafer_shot_4_all_exp200000.png"],
    )
    assert stats["n_undetected"] == 1
    assert stats["n_real_total"] == 1


def test_submap_tiles_are_counted_separately() -> None:
    """A quad fake is not a fake of the combi, and a quad hit is not a combi miss."""
    combi = "v6/combi_2of4.png"
    quad = "v6/wafer_shot_3_quad3_exp200000.png"
    uniform = "v6/wafer_shot_4_all_exp200000.png"
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [10.0, 20.0, 30.0],
            "y_global": [11.0, 21.0, 31.0],
            "size": [20_000.0, 20_000.0, 20_000.0],
            "confidence": [0.9, 0.4, 0.8],
            "source_tile": [combi, quad, quad],
            "nsew_dirs": ["COMBI", "QUAD3", "QUAD3"],
            "key": ["v6_LIGHT_10_11", "v6_LIGHT_20_21", "v6_LIGHT_30_31"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": [
                "v6_LIGHT_10_11",
                "v6_LIGHT_20_21",
                "v6_LIGHT_30_31",
                "v6_LIGHT_400_400",
            ],
            "label": ["particle", "particle", "not_particle", "particle"],
            "source_tile": [combi, quad, quad, combi],
            "source_csv": ["run.csv", "run.csv", "run.csv", "tile_inspect"],
            "size": [20_000.0, 20_000.0, 20_000.0, 22_000.0],
            "x_global": [10.0, 20.0, 30.0, 400.0],
            "y_global": [11.0, 21.0, 31.0, 400.0],
        }
    )
    tiles = [combi, quad, uniform]
    on_combi = last_run_class_counts(
        detections,
        labels,
        tile_names=[combi],
        min_size_nm=20_000.0,
        folder_tiles=tiles,
    )
    on_quad = last_run_class_counts(
        detections,
        labels,
        tile_names=[quad],
        min_size_nm=20_000.0,
        folder_tiles=tiles,
    )
    assert on_combi["n_detected"] == 1
    assert on_combi["n_real"] == 1
    assert on_combi["n_fake"] == 0
    assert on_combi["n_undetected"] == 1
    assert on_quad["n_detected"] == 2
    assert on_quad["n_fake"] == 1
    assert on_quad["n_real"] == 1
    assert on_quad["n_undetected"] == 0


def test_last_run_class_counts_drops_undetected_below_run_size_floor() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [20000.0],
            "confidence": [0.9],
            "source_tile": ["a.tif"],
            "key": ["hit"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["big_miss", "small_miss"],
            "label": ["particle", "particle"],
            "source_tile": ["a.tif", "a.tif"],
            "size": [25000.0, 15000.0],
        }
    )
    stats = last_run_class_counts(
        detections, labels, tile_names=["a.tif"], min_size_nm=20000.0
    )
    assert stats["n_undetected"] == 1
    assert stats["n_undetected_below"] == 1
    assert stats["n_real_total"] == 1
    assert stats["precision"] == 0.0
    assert stats["recall"] == 0.0


def test_subfolder_score_rows_combines_tiles_and_splits_vn() -> None:
    per_tile = pd.DataFrame(
        [
            {
                "tile": "(1,1).png",
                "n_detected": 2,
                "n_real": 1,
                "n_undetected": 1,
            },
            {
                "tile": "(3,3).png",
                "n_detected": 2,
                "n_real": 2,
                "n_undetected": 0,
            },
            {
                "tile": "v5/quad.png",
                "n_detected": 1,
                "n_real": 1,
                "n_undetected": 0,
            },
            {
                "tile": "v7/quad.png",
                "n_detected": 4,
                "n_real": 0,
                "n_undetected": 2,
            },
        ]
    )
    rows = subfolder_score_rows(
        per_tile,
        folder_name="Groundup/Final test 21-10/v44 2026-10-20 N40 (10%) copy",
        beta=1.0,
    )
    by_name = {row["name"]: row for row in rows}
    assert list(by_name) == [
        "v44 2026-10-20 N40 (10%) copy",
        "v5",
        "v7",
    ]
    folder = by_name["v44 2026-10-20 N40 (10%) copy"]
    assert folder["n_detected"] == 4
    assert folder["n_real"] == 3
    assert folder["n_undetected"] == 1
    assert folder["precision"] == pytest.approx(0.75)
    assert folder["recall"] == pytest.approx(0.75)
    assert folder["precision_std"] == pytest.approx(float(np.std([0.5, 1.0], ddof=1)))
    assert folder["recall_std"] == pytest.approx(float(np.std([0.5, 1.0], ddof=1)))
    assert folder["f_beta_std"] == pytest.approx(float(np.std([0.5, 1.0], ddof=1)))
    assert by_name["v5"]["precision_std"] is None
    assert by_name["v7"]["n_detected"] == 4
    assert by_name["v7"]["precision"] == 0.0


def test_last_run_class_counts_by_tile_splits_folder() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [10.0, 20.0, 30.0],
            "y_global": [11.0, 21.0, 31.0],
            "size": [20000.0, 20000.0, 20000.0],
            "confidence": [0.9, 0.8, 0.7],
            "source_tile": ["a.tif", "a.tif", "b.tif"],
            "key": ["a_real", "a_fake", "b_real"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_real", "a_fake", "a_miss", "b_real"],
            "label": ["particle", "not_particle", "particle", "particle"],
            "source_tile": ["a.tif", "a.tif", "a.tif", "b.tif"],
            "size": [25000.0, 20000.0, 25000.0, 20000.0],
        }
    )
    table = last_run_class_counts_by_tile(
        detections, labels, ["a.tif", "b.tif"], min_size_nm=20000.0
    )
    by_tile = table.set_index("tile")
    assert list(table["tile"]) == ["a.tif", "b.tif"]
    assert int(by_tile.loc["a.tif", "n_detected"]) == 2
    assert int(by_tile.loc["a.tif", "n_real"]) == 1
    assert int(by_tile.loc["a.tif", "n_undetected"]) == 1
    assert float(by_tile.loc["a.tif", "precision"]) == 0.5
    assert float(by_tile.loc["a.tif", "recall"]) == 0.5
    assert int(by_tile.loc["b.tif", "n_detected"]) == 1
    assert float(by_tile.loc["b.tif", "precision"]) == 1.0
    assert float(by_tile.loc["b.tif", "recall"]) == 1.0


def test_last_run_class_counts_by_tile_keeps_submap_folder() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2],
            "x_global": [10.0, 20.0],
            "y_global": [11.0, 21.0],
            "size": [20000.0, 20000.0],
            "confidence": [0.9, 0.8],
            "source_tile": [
                "v4/wafer_shot_0_quad0_exp200000.png",
                "v4/wafer_shot_1_quad1_exp200000.png",
            ],
            "key": ["quad0", "quad1"],
        }
    )
    table = last_run_class_counts_by_tile(
        detections,
        None,
        [
            "v4/wafer_shot_0_quad0_exp200000.png",
            "v4/wafer_shot_1_quad1_exp200000.png",
        ],
    )
    by_tile = table.set_index("tile")
    assert int(by_tile.loc["v4/wafer_shot_0_quad0_exp200000.png", "n_detected"]) == 1
    assert int(by_tile.loc["v4/wafer_shot_1_quad1_exp200000.png", "n_detected"]) == 1


def test_shared_combi_uniform_hit_counts_on_both_pictures() -> None:
    combi = "v6/combi_2of4.png"
    uniform = "v6/wafer_shot_4_all_exp200000.png"
    detections = pd.DataFrame(
        {
            "id": [7, 21],
            "x_global": [100.0, 2_000.0],
            "y_global": [200.0, 200.0],
            "size": [20_000.0, 20_000.0],
            "confidence": [0.9, 0.8],
            "source_tile": [combi, uniform],
            "nsew_dirs": ["ALL,COMBI", "ALL"],
            "key": ["shared", "uniform-only"],
        }
    )
    both = last_run_class_counts(detections, None, tile_names=[combi, uniform])
    on_combi = last_run_class_counts(detections, None, tile_names=[combi])
    on_uniform = last_run_class_counts(detections, None, tile_names=[uniform])
    assert both["n_detected"] == 2
    assert on_combi["n_detected"] == 1
    assert on_uniform["n_detected"] == 2


def test_std_of_tile_scores_uses_every_tile() -> None:
    table = pd.DataFrame(
        [
            {"tile": "(1,1).png", "n_detected": 2, "n_real": 1, "n_undetected": 1},
            {"tile": "(1,2).png", "n_detected": 2, "n_real": 2, "n_undetected": 0},
            {"tile": "(1,3).png", "n_detected": 0, "n_real": 0, "n_undetected": 0},
        ]
    )
    spread = std_of_tile_scores(table, beta=1.0)
    expected = float(np.std([0.5, 1.0, 0.0], ddof=1))
    assert spread["precision"] == pytest.approx(expected)
    assert spread["recall"] == pytest.approx(expected)
    assert spread["f_beta"] == pytest.approx(expected)
    assert std_of_tile_scores(table.iloc[:1], beta=1.0) == {
        "precision": None,
        "recall": None,
        "f_beta": None,
    }

    counted = table.iloc[:2]
    rows = subfolder_score_rows(
        counted,
        folder_name="folder",
        beta=1.0,
        std_tiles=table,
    )
    assert rows[0]["n_detected"] == 4
    assert rows[0]["precision_std"] == pytest.approx(expected)


def test_last_run_class_counts_by_tile_splits_particles_only() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [10.0, 20.0, 30.0],
            "y_global": [11.0, 21.0, 31.0],
            "size": [20000.0, 20000.0, 20000.0],
            "confidence": [0.9, 0.8, 0.7],
            "source_tile": [
                "particles_only.png",
                "particles_only.png",
                "particles_only_4of4.png",
            ],
            "key": ["NSEW_10_11", "NSEW_20_21", "NSEW_30_31"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["NSEW_10_11", "NSEW_30_31", "NSEW_99_99"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["N.png", "N.png", "E.png"],
            "size": [20000.0, 20000.0, 20000.0],
            "x_global": [10.0, 30.0, 99.0],
            "y_global": [11.0, 31.0, 99.0],
        }
    )
    table = last_run_class_counts_by_tile(
        detections,
        labels,
        ["particles_only.png", "particles_only_4of4.png"],
        min_size_nm=20000.0,
        folder_tiles=["particles_only.png", "particles_only_4of4.png", "N.png", "E.png"],
    )
    by_tile = table.set_index("tile")
    assert int(by_tile.loc["particles_only.png", "n_detected"]) == 2
    assert int(by_tile.loc["particles_only.png", "n_real"]) == 1
    assert int(by_tile.loc["particles_only_4of4.png", "n_detected"]) == 1
    assert int(by_tile.loc["particles_only_4of4.png", "n_real"]) == 1


def test_last_run_ignores_nsew_labels_from_another_folder() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [20000.0],
            "confidence": [0.9],
            "source_tile": ["N.png"],
            "key": ["NSEW_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["NSEW_10_11", "NSEW_old_v2", "NSEW_miss_v3"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["N.png", "N.bmp", "E.png"],
            "size": [20000.0, 25000.0, 25000.0],
            "x_global": [10.0, 99.0, 50.0],
            "y_global": [11.0, 98.0, 51.0],
        }
    )
    stats = last_run_class_counts(
        detections,
        labels,
        tile_names=["N.png", "E.png", "S.png", "W.png"],
        min_size_nm=20000.0,
    )
    assert stats["n_real"] == 1
    assert stats["n_undetected"] == 1
    assert stats["n_real_total"] == 2


def test_grid_labels_stay_in_their_own_folder() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [20000.0],
            "confidence": [0.9],
            "source_tile": ["(1,1).png"],
            "key": ["(1,1)_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["(1,1)_10_11", "(1,1)_40_41", "(1,1)_99_98", "(1,1)_50_51"],
            "label": ["particle", "particle", "particle", "particle"],
            "source_tile": [
                "Groundup/wafer-a/(1,1).png",
                "Groundup/wafer-a/(1,1).png",
                "Groundup/wafer-b/(1,1).png",
                "(1,1).png",
            ],
            "size": [20000.0, 22000.0, 25000.0, 25000.0],
            "x_global": [10.0, 40.0, 99.0, 50.0],
            "y_global": [11.0, 41.0, 98.0, 51.0],
        }
    )
    stats = last_run_class_counts(
        detections,
        labels,
        tile_names=["(1,1).png"],
        min_size_nm=20000.0,
        run_dir="Groundup/wafer-a",
    )
    assert stats["n_real"] == 1
    assert stats["n_undetected"] == 1
    assert stats["n_real_total"] == 2


def test_tinder_saves_a_grid_label_in_this_folder() -> None:
    from src.verification.inspect import label_on_run_tiles, with_grid_folder

    saved = with_grid_folder(
        {"source_tile": "(1,1).png", "key": "(1,1)_10_11"},
        "Groundup/wafer-a",
    )
    assert saved["source_tile"] == "Groundup/wafer-a/(1,1).png"
    assert with_grid_folder({"source_tile": "N.png"}, "Groundup/wafer-a")["source_tile"] == "N.png"
    assert label_on_run_tiles(saved["source_tile"], {"(1,1).png"}, "Groundup/wafer-a")
    assert not label_on_run_tiles(saved["source_tile"], {"(1,1).png"}, "Groundup/wafer-b")


def test_grid_source_name_keeps_the_folder() -> None:
    from src.overall.io.naming import lighting_source_name
    from src.overall.config import DEFAULT_INPUT_DIR

    path = DEFAULT_INPUT_DIR / "Groundup" / "wafer-a" / "(1,1).png"
    assert lighting_source_name(path) == "Groundup/wafer-a/(1,1).png"
    assert lighting_source_name(DEFAULT_INPUT_DIR / "v5" / "N.png") == "N.png"


def test_last_run_particles_only_uses_same_scene_nsew_labels() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [20000.0],
            "confidence": [0.9],
            "source_tile": ["particles_only_2of4.png"],
            "key": ["NSEW_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["NSEW_10_11", "NSEW_50_51", "NSEW_99_98"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["N.png", "E.png", "N.bmp"],
            "size": [20000.0, 25000.0, 25000.0],
            "x_global": [10.0, 50.0, 99.0],
            "y_global": [11.0, 51.0, 98.0],
        }
    )
    stats = last_run_class_counts(
        detections,
        labels,
        tile_names=["particles_only_2of4.png", "particles_only_4of4.png"],
        min_size_nm=20000.0,
        folder_tiles=[
            "particles_only_2of4.png",
            "particles_only_4of4.png",
            "N.png",
            "E.png",
            "S.png",
            "W.png",
        ],
    )
    assert stats["n_real"] == 1
    assert stats["n_undetected"] == 1
    assert stats["n_real_total"] == 2


def test_last_run_snaps_particles_only_to_nearby_nsew_label() -> None:
    from src.verification.queue import snap_detection_keys_to_labels, tag_table

    detections = tag_table(
        pd.DataFrame(
            {
                "id": [1],
                "x_global": [18.0],
                "y_global": [11.0],
                "size": [20000.0],
                "confidence": [0.9],
                "source_tile": ["particles_only_2of4.png"],
            }
        )
    )
    labels = pd.DataFrame(
        {
            "key": ["NSEW_10_11"],
            "label": ["particle"],
            "source_tile": ["N.png"],
            "size": [20000.0],
            "x_global": [10.0],
            "y_global": [11.0],
        }
    )
    snapped = snap_detection_keys_to_labels(
        detections, labels, merge_radius_nm=24.0, size_match_fraction=0.0
    )
    stats = last_run_class_counts(
        snapped,
        labels,
        tile_names=["particles_only_2of4.png"],
        min_size_nm=20000.0,
        folder_tiles=["particles_only_2of4.png", "N.png"],
    )
    assert stats["n_real"] == 1
    assert stats["n_unlabeled"] == 0
    assert stats["n_undetected"] == 0


def test_related_scene_tile_names_keeps_groundup_v3_suffix(tmp_path: Path) -> None:
    from src.verification.inspect import related_scene_tile_names

    v3 = tmp_path / "Groundup v3"
    v3.mkdir()
    (v3 / "N.png").write_bytes(b"n")
    (v3 / "S.png").write_bytes(b"s")
    po = v3 / "Output Groundup v3" / "Particles only"
    po.mkdir(parents=True)
    (po / "particles_only_2of4.png").write_bytes(b"p")
    (po / "particles_only_4of4.png").write_bytes(b"q")
    v2 = tmp_path / "Groundup v2"
    v2.mkdir()
    (v2 / "N.bmp").write_bytes(b"old")

    names = related_scene_tile_names(po, ["particles_only_2of4.png", "particles_only_4of4.png"])
    assert "N.png" in names
    assert "S.png" in names
    assert "particles_only_4of4.png" in names
    assert "N.bmp" not in names

    relative = related_scene_tile_names(
        "Groundup v3/Output Groundup v3/Particles only",
        ["particles_only_2of4.png"],
    )
    assert "N.png" in relative
    assert "N.bmp" not in relative


def test_undetected_size_floor_uses_this_run_minimum() -> None:
    detections = pd.DataFrame({"size": [20000.0, 31000.0]})
    floor = undetected_size_floor_nm(detections, {"detection": {"min_size_nm": 10000.0}})
    assert floor == 20000.0


def _area_line_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels of the #0F238C area outline, allowing antialiasing."""
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    return (blue > 80) & (blue > red + 25) & (blue > green + 25)


def test_outline_particle_area_follows_a_green_marker() -> None:
    import cv2

    from src.verification.inspect import outline_particle_area

    image = np.full((200, 200), 20, dtype=np.uint8)
    cv2.circle(image, (60, 70), 16, 220, thickness=-1)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.circle(rgb, (60, 70), 26, (40, 180, 70), thickness=3)
    out = outline_particle_area(rgb, size_nm=32.0, pixel_size_nm=1.0, config={})
    line = _area_line_mask(out)
    ys, xs = np.where(line)
    assert ys.size > 30
    assert abs(float(np.median(ys)) - 70.0) < 10.0
    assert abs(float(np.median(xs)) - 60.0) < 10.0


def test_outline_particle_area_is_a_thin_line() -> None:
    import cv2

    from src.verification.inspect import outline_particle_area

    image = np.full((200, 200), 20, dtype=np.uint8)
    cv2.circle(image, (100, 100), 18, 220, thickness=-1)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.circle(rgb, (100, 100), 28, (255, 140, 40), thickness=3)
    out = outline_particle_area(
        rgb,
        size_nm=36.0,
        pixel_size_nm=1.0,
        config={"detection": {"size_mass_fraction": 0.4, "particles_bright": True}},
    )
    line = _area_line_mask(out)
    assert 40 < int(line.sum()) < 500
    ys, xs = np.where(line)
    dist = np.hypot(ys - 100.0, xs - 100.0)
    assert 14.0 < float(np.median(dist)) < 24.0
    orange = (out[:, :, 0] > 200) & (out[:, :, 1] > 80) & (out[:, :, 2] < 80)
    assert int(orange.sum()) > 20


def test_circles_from_rows_uses_label_color() -> None:
    rows = pd.DataFrame(
        {
            "key": ["a", "b"],
            "size": [10.0, 12.0],
        }
    )
    circles = circles_from_rows(rows, [1.0, 2.0], [3.0, 4.0], {"a": "particle"})
    assert circles[0]["color"] == (40, 180, 70)
    assert circles[1]["color"] == (255, 140, 40)
    missed = rows.copy()
    missed["missed"] = [True, False]
    missed_circles = circles_from_rows(missed, [1.0, 2.0], [3.0, 4.0], {"a": "particle"})
    assert missed_circles[0]["color"] == MISSED_PARTICLE_COLOR
    assert missed_circles[1]["color"] == (255, 140, 40)


def test_display_xy_to_local_inverts_overlay_scale() -> None:
    view = OverlayView(rgb=np.zeros((50, 80, 3), dtype=np.uint8), crop_x0=10, crop_y0=20, scale=0.5)
    x_local, y_local = display_xy_to_local(15.0, 8.0, view)
    assert x_local == 10.0 + 15.0 / 0.5
    assert y_local == 20.0 + 8.0 / 0.5


def test_fresh_inspect_click_ignores_the_press_already_seen() -> None:
    press = {"x": 12, "y": 40, "unix_time": 1_700_000_000_000}
    assert fresh_inspect_click(press, None) == 1_700_000_000_000
    assert fresh_inspect_click(press, 1_700_000_000_000) is None
    assert fresh_inspect_click(press, "1700000000000") is None
    assert fresh_inspect_click({"x": 12, "y": 40}, None) is None
    assert fresh_inspect_click(None, None) is None


def test_verify_missed_photo_sizes_replaces_assumed_diameter(tmp_path: Path) -> None:
    import cv2

    from src.verification.inspect import verify_missed_photo_sizes
    from src.verification.store import LabelStore

    image = np.zeros((81, 81), dtype=np.uint8)
    yy, xx = np.ogrid[:81, :81]
    image[(yy - 40) ** 2 + (xx - 40) ** 2 <= 10.0**2] = 220
    tile = tmp_path / "v5"
    tile.mkdir()
    assert cv2.imwrite(str(tile / "N.png"), image)
    other = tmp_path / "other.png"
    assert cv2.imwrite(str(other), image)

    pixel = 1000.0
    record = missed_particle_record("v5/N.png", 40.0 * pixel, 40.0 * pixel, 15_000.0)
    elsewhere = missed_particle_record("other.png", 40.0 * pixel, 40.0 * pixel, 15_000.0)
    store = LabelStore(tmp_path / "labels")
    store.apply_label(record, "particle", np.zeros((8, 8, 3), dtype=np.uint8))
    store.apply_label(elsewhere, "particle", np.zeros((8, 8, 3), dtype=np.uint8))

    report = verify_missed_photo_sizes(
        store.load(),
        {"v5/N.png": tile / "N.png"},
        {"detection": {"max_size_nm": 80_000.0, "size_mass_fraction": 0.4, "particles_bright": True}},
        pixel_size_nm=pixel,
    )
    assert len(report) == 1
    assert report[0]["source_tile"] == "v5/N.png"
    assert abs(report[0]["photo_um"] - 20.0) < 1.5
    assert report[0]["note"] == "updated"
    written = store.update_sizes({report[0]["key"]: report[0]["photo_nm"]})
    assert written == 1
    saved = store.load()
    row = saved.loc[saved["source_tile"].astype(str) == "v5/N.png"].iloc[-1]
    assert abs(float(row["size"]) / 1000.0 - 20.0) < 1.5
    other_row = saved.loc[saved["source_tile"].astype(str) == "other.png"].iloc[-1]
    assert float(other_row["size"]) == 15_000.0


def test_verify_drops_a_miss_under_20_um(tmp_path: Path) -> None:
    import cv2

    from src.verification.inspect import verify_missed_photo_sizes
    from src.verification.store import LabelStore

    image = np.zeros((81, 81), dtype=np.uint8)
    yy, xx = np.ogrid[:81, :81]
    image[(yy - 40) ** 2 + (xx - 40) ** 2 <= 3.0**2] = 220
    tile = tmp_path / "v5"
    tile.mkdir()
    assert cv2.imwrite(str(tile / "combi_2of4.png"), image)
    pixel = 1000.0
    record = missed_particle_record(
        "v5/combi_2of4.png", 40.0 * pixel, 40.0 * pixel, 15_000.0
    )
    store = LabelStore(tmp_path / "labels")
    store.apply_label(record, "particle", np.zeros((8, 8, 3), dtype=np.uint8))
    report = verify_missed_photo_sizes(
        store.load(),
        {"v5/combi_2of4.png": tile / "combi_2of4.png"},
        {"detection": {"max_size_nm": 80_000.0, "size_mass_fraction": 0.4, "particles_bright": True}},
        pixel_size_nm=pixel,
    )
    assert len(report) == 1
    assert report[0]["photo_um"] < 20.0
    assert report[0]["note"] == "removed"
    assert store.unlabel(report[0]["key"]) is not None
    assert store.load().empty


def test_verify_keeps_a_grown_mark_above_20_um(tmp_path: Path) -> None:
    import cv2

    from src.verification.inspect import verify_missed_photo_sizes
    from src.verification.store import LabelStore

    image = np.zeros((81, 81), dtype=np.uint8)
    yy, xx = np.ogrid[:81, :81]
    image[(yy - 40) ** 2 + (xx - 40) ** 2 <= 4.0**2] = 220
    tile = tmp_path / "v5"
    tile.mkdir()
    assert cv2.imwrite(str(tile / "combi_2of4.png"), image)
    pixel = 1000.0
    record = missed_particle_record(
        "v5/combi_2of4.png", 40.0 * pixel, 40.0 * pixel, 28_000.0
    )
    store = LabelStore(tmp_path / "labels")
    store.apply_label(record, "particle", np.zeros((8, 8, 3), dtype=np.uint8))
    report = verify_missed_photo_sizes(
        store.load(),
        {"v5/combi_2of4.png": tile / "combi_2of4.png"},
        {"detection": {"max_size_nm": 80_000.0, "size_mass_fraction": 0.4, "particles_bright": True}},
        pixel_size_nm=pixel,
    )
    assert report[0]["photo_um"] < 20.0
    assert report[0]["note"] == "kept stored size"


def test_missed_particle_record_uses_tile_inspect_source() -> None:
    record = missed_particle_record("R3_5_17_5X.tif", 100.0, 200.0, 15000.0)
    assert record["source_csv"] == "tile_inspect"
    assert record["key"] == "R3_5_17_5X_100_200"
    assert record["size"] == 15000.0


def test_n2_pattern_keeps_detector_and_inspect_circles_on_tile() -> None:
    """R4 n2_row_col names need that pattern or mosaic globals fall off the tile."""
    base = TilePlacement(
        path=Path("n2_1_17.tif"),
        name="n2_1_17.tif",
        y0=0,
        x0=0,
        height=2076,
        width=3088,
    )
    config = {
        "filename_pattern": r"n2_(?P<row>\d+)_(?P<col>\d+)\.tiff?",
        "overlap_fraction": 0.0,
    }
    extras = extra_grid_placements(base, config)
    placements = [base, *extras]
    x0 = 16 * 3088
    det_x, det_y = local_xy_on_tile(
        (100.0 + x0) * 960.0, 200.0 * 960.0, placements, 960.0, 3088, 2076
    )
    miss_x, miss_y = local_xy_on_tile(
        100.0 * 960.0, 200.0 * 960.0, placements, 960.0, 3088, 2076
    )
    assert 0.0 <= det_x < 3088.0
    assert 0.0 <= det_y < 2076.0
    assert 0.0 <= miss_x < 3088.0
    assert 0.0 <= miss_y < 2076.0


def test_extra_grid_placement_maps_last_run_csv_onto_tile() -> None:
    base = TilePlacement(
        path=Path("R3_1_26_5X.tif"),
        name="R3_1_26_5X.tif",
        y0=0,
        x0=0,
        height=2076,
        width=3088,
    )
    extras = extra_grid_placements(base, {"filename_pattern": r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?", "overlap_fraction": 0.0})
    x_local, y_local = local_xy_on_tile(
        68686080.0, 219840.0, [base, *extras], 960.0, 3088, 2076
    )
    assert 0.0 <= x_local < 3088.0
    assert 0.0 <= y_local < 2076.0


def test_preview_click_crop_keeps_file_gray_levels() -> None:
    image = np.full((200, 200), 80, dtype=np.uint8)
    image[98:103, 98:103] = 100
    rgb = preview_click_crop(image, 100.0, 100.0, 10.0, 1.0, color=(0, 0, 0))
    assert int(rgb[10, 10, 0]) == 80
    assert int(rgb[10, 10, 1]) == 80
    assert int(rgb[10, 10, 2]) == 80


def test_preview_click_crop_zoom_magnifies_the_speck() -> None:
    image = np.zeros((400, 400), dtype=np.float32)
    image[198:203, 198:203] = 1.0
    plain = preview_click_crop(image, 200.0, 200.0, 15_000.0, 3350.0)
    zoomed = preview_click_crop(image, 200.0, 200.0, 15_000.0, 3350.0, zoom=8.0)
    assert zoomed.shape[0] > plain.shape[0]
    assert zoomed.shape[0] <= 720


def test_preview_click_crop_keeps_full_circle_near_tile_edge() -> None:
    image = np.full((80, 90), 0.2, dtype=np.float32)
    image[2:10, 2:10] = 1.0
    rgb = preview_click_crop(image, 6.0, 6.0, 20.0, 1.0, color=(40, 180, 70))
    assert rgb.shape[0] >= 80
    assert rgb.shape[1] >= 90
    green = (rgb[:, :, 1] > 150) & (rgb[:, :, 0] < 80) & (rgb[:, :, 2] < 100)
    assert int(green.sum()) > 20


def test_local_xy_on_tile_prefers_in_frame_placement() -> None:
    folder = TilePlacement(
        path=Path("R3_2_31_5X.tif"),
        name="R3_2_31_5X.tif",
        y0=0,
        x0=38909,
        height=2076,
        width=3088,
    )
    wafer = TilePlacement(
        path=Path("R3_2_31_5X.tif"),
        name="R3_2_31_5X.tif",
        y0=1868,
        x0=83376,
        height=2076,
        width=3088,
    )
    x_global, y_global = local_px_to_global_nm(100.0, 80.0, wafer, 960.0)
    x_local, y_local = local_xy_on_tile(
        x_global, y_global, [folder, wafer], 960.0, 3088, 2076
    )
    assert x_local == 100.0
    assert y_local == 80.0

    x_click, y_click = local_px_to_global_nm(50.0, 60.0, folder, 960.0)
    x_local, y_local = local_xy_on_tile(
        x_click, y_click, [folder, wafer], 960.0, 3088, 2076
    )
    assert x_local == 50.0
    assert y_local == 60.0


def test_local_px_roundtrip_to_global_nm() -> None:
    placement = TilePlacement(
        path=Path("R3_1_2_5X.tif"),
        name="R3_1_2_5X.tif",
        y0=100,
        x0=80,
        height=128,
        width=128,
    )
    x_global, y_global = local_px_to_global_nm(5.0, 4.0, placement, pixel_size_nm=2.0)
    from src.verification.crops import global_nm_to_local_px

    x_local, y_local = global_nm_to_local_px(x_global, y_global, placement, pixel_size_nm=2.0)
    assert x_local == 5.0
    assert y_local == 4.0


def test_pointer_uses_directional_photos_and_shares_one_set(tmp_path: Path) -> None:
    import cv2

    from src.verification.inspect import pointer_direction_tiles, pointer_set_hits

    staged = tmp_path / "Inputs" / "Groundup v4 ML off"
    staged.mkdir(parents=True)
    photo = np.zeros((8, 8), dtype=np.uint8)
    for set_id in ("v1", "v2"):
        cv2.imwrite(str(staged / f"{set_id}_2of4.bmp"), photo)
        folder = tmp_path / "Inputs" / "Groundup v4" / set_id
        folder.mkdir(parents=True)
        for direction in ("N", "S", "E", "W"):
            cv2.imwrite(str(folder / f"{direction}.bmp"), photo)

    tiles = pointer_direction_tiles(staged)
    labels = [tile.label for tile in tiles]
    assert labels[:4] == ["v1/N.bmp", "v1/S.bmp", "v1/E.bmp", "v1/W.bmp"]
    assert all(tile.path.parent.name == tile.set_id for tile in tiles)
    assert tiles[0].path.name == "N.bmp"

    detections = pd.DataFrame(
        [
            {
                "id": 1,
                "source_tile": "v1_2of4.bmp",
                "x_global": 100.0,
                "y_global": 200.0,
                "size": 20_000.0,
                "confidence": 0.8,
                "key": "v1_2of4_100_200",
            }
        ]
    )
    marks = pd.DataFrame(
        [
            {
                "key": "v1_NSEW_300_400",
                "source_tile": "v1/N.bmp",
                "x_global": 300.0,
                "y_global": 400.0,
                "size": 15_000.0,
                "confidence": 0.0,
                "label": "particle",
            },
            {
                "key": "v2_NSEW_300_400",
                "source_tile": "v2/N.bmp",
                "x_global": 300.0,
                "y_global": 400.0,
                "size": 15_000.0,
                "confidence": 0.0,
                "label": "particle",
            },
        ]
    )
    v1 = pointer_set_hits(detections, marks, "v1")
    keys = set(v1["key"].astype(str))
    assert keys == {"v1_2of4_100_200", "v1_NSEW_300_400"}
    missed = v1.loc[v1["key"].astype(str) == "v1_NSEW_300_400", "missed"]
    assert bool(missed.iloc[0]) is True
    assert set(pointer_set_hits(detections, marks, "v2")["key"].astype(str)) == {"v2_NSEW_300_400"}


def test_growing_the_click_includes_the_dim_edge() -> None:
    from src.verification.inspect import footprint_grow_mass, photo_particle_edge

    image = np.full((81, 81), 40.0, dtype=np.float32)
    yy, xx = np.ogrid[:81, :81]
    radius = np.hypot(yy - 40, xx - 40)
    image[radius <= 6] = 220.0
    image[(radius > 6) & (radius <= 14)] = 90.0
    config = {
        "detection": {
            "max_size_nm": 80_000.0,
            "size_mass_fraction": 0.4,
            "particles_bright": True,
        }
    }
    tight_nm, _tight = photo_particle_edge(
        image, 40.0, 40.0, pixel_size_nm=1000.0, config=config, mass_fraction=0.4
    )
    wide_nm, _wide = photo_particle_edge(
        image,
        40.0,
        40.0,
        pixel_size_nm=1000.0,
        config=config,
        mass_fraction=footprint_grow_mass(config, 300),
    )
    assert footprint_grow_mass(config, 1) == 0.4
    assert footprint_grow_mass(config, 300) == 0.0
    assert footprint_grow_mass(config, 1.5) < footprint_grow_mass(config, 1)
    assert wide_nm > tight_nm + 4_000.0


def test_mosaic_uses_this_folders_review_color() -> None:
    from src.main_pipeline.report.report_generator import draw_overlay_markers
    from src.main_pipeline.stitching.stitcher import LazyMosaic
    from src.verification.inspect import LABEL_CIRCLE_COLORS, label_colors_for_hits

    tile = "Groundup/wafer-a/(1,1).png"
    hits = pd.DataFrame(
        [
            {
                "source_tile": tile,
                "x_global": 1000.0,
                "y_global": 1000.0,
                "size": 20_000.0,
                "key": "(1,1)_1000_1000",
            },
            {
                "source_tile": tile,
                "x_global": 3000.0,
                "y_global": 1000.0,
                "size": 20_000.0,
                "key": "(1,1)_3000_1000",
            },
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "source_tile": tile,
                "x_global": 1000.0,
                "y_global": 1000.0,
                "size": 20_000.0,
                "key": "(1,1)_1000_1000",
                "label": "particle",
            },
            {
                "source_tile": "Groundup/wafer-b/(1,1).png",
                "x_global": 3000.0,
                "y_global": 1000.0,
                "size": 20_000.0,
                "key": "(1,1)_3000_1000",
                "label": "not_particle",
            },
        ]
    )
    colors = label_colors_for_hits(hits, labels, run_dir="Groundup/wafer-a")
    assert colors["(1,1)_1000_1000"] == LABEL_CIRCLE_COLORS["particle"]
    assert "(1,1)_3000_1000" not in colors

    mosaic = LazyMosaic(
        [TilePlacement(path=Path("(1,1).png"), name="(1,1).png", y0=0, x0=0, height=40, width=40)]
    )
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    marked = draw_overlay_markers(
        rgb,
        mosaic,
        hits.iloc[:1].assign(x_global=20_000.0, y_global=20_000.0, size=1_000.0),
        {"pixel_size_nm": 1000.0, "report": {"downsample": 1}},
        colors=colors,
        scale=1.0,
    )
    ring = marked[6, 20]
    assert int(ring[1]) > int(ring[0])


def test_write_marked_mosaic_saves_review_and_inspect_circles(tmp_path: Path) -> None:
    import cv2

    from src.main_pipeline.stitching.stitcher import LazyMosaic
    from src.verification.inspect import (
        LABEL_CIRCLE_COLORS,
        MARKED_MOSAIC_NAME,
        write_marked_mosaic,
    )

    out = tmp_path / "out"
    out.mkdir()
    plain = np.zeros((50, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(out / "mosaic_plain.png"), plain)
    mosaic = LazyMosaic(
        [
            TilePlacement(
                path=Path("field.png"),
                name="field.png",
                y0=0,
                x0=0,
                height=50,
                width=100,
            )
        ]
    )
    hits = pd.DataFrame(
        [
            {
                "source_tile": "field.png",
                "x_global": 20_000.0,
                "y_global": 25_000.0,
                "size": 1_000.0,
                "key": "hit-real",
            },
            {
                "source_tile": "field.png",
                "x_global": 50_000.0,
                "y_global": 25_000.0,
                "size": 1_000.0,
                "key": "hit-open",
            },
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "source_tile": "field.png",
                "x_global": 20_000.0,
                "y_global": 25_000.0,
                "size": 1_000.0,
                "key": "hit-real",
                "label": "particle",
                "source_csv": "particles.csv",
            },
            {
                "source_tile": "field.png",
                "x_global": 80_000.0,
                "y_global": 25_000.0,
                "size": 1_000.0,
                "key": "miss-1",
                "label": "particle",
                "source_csv": MISS_SOURCE,
            },
        ]
    )
    path = write_marked_mosaic(
        {
            "output_dir": str(out),
            "input_dir": str(tmp_path),
            "pixel_size_nm": 1000.0,
            "report": {"downsample": 1},
        },
        hits,
        labels,
        mosaic=mosaic,
    )
    assert path == out / MARKED_MOSAIC_NAME
    assert path.is_file()
    assert not (out / "mosaic_overlay.png").is_file()
    saved = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert saved is not None

    def ring(x: int) -> np.ndarray:
        return saved[11, x]

    particle = ring(20)
    unlabeled = ring(50)
    missed = ring(80)
    assert int(particle[1]) > int(particle[2])
    assert int(particle[1]) > 100
    assert tuple(int(v) for v in LABEL_CIRCLE_COLORS["particle"])[1] == 180
    assert int(unlabeled[2]) > int(unlabeled[1])
    assert int(missed[1]) > int(missed[2])
    assert int(missed[1]) < int(particle[1])


def test_write_marked_mosaic_skips_a_run_with_no_tiles(tmp_path: Path) -> None:
    from src.main_pipeline.stitching.stitcher import LazyMosaic
    from src.verification.inspect import MARKED_MOSAIC_NAME, write_marked_mosaic

    out = tmp_path / "out"
    out.mkdir()
    path = write_marked_mosaic(
        {"output_dir": str(out), "input_dir": str(tmp_path), "pixel_size_nm": 1000.0},
        pd.DataFrame(),
        pd.DataFrame(),
        mosaic=LazyMosaic([]),
    )
    assert path is None
    assert not (out / MARKED_MOSAIC_NAME).is_file()
