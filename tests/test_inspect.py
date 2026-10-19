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
    detections_on_tile,
    display_xy_to_local,
    expand_direction_tile_names,
    expand_submap_lighting_names,
    extra_grid_placements,
    unhappiness_score,
    labeled_detections,
    labeled_particle_recovery,
    crop_metric_cells,
    crop_metric_std,
    last_run_class_counts,
    last_run_class_counts_by_tile,
    local_xy_on_tile,
    missed_particle_record,
    overlay_circles,
    preview_click_crop,
    tile_names_for_hits,
    undetected_size_floor_nm,
)
from src.main_pipeline.stitching.stitcher import TilePlacement


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


def test_detections_on_this_file_keeps_only_that_quad() -> None:
    table = pd.DataFrame(
        {
            "source_tile": [
                "v4/wafer_shot_0_quad0_exp200000.png",
                "v4/wafer_shot_1_quad1_exp200000.png",
                "v4/combi_2of4.png",
            ],
            "size": [1.0, 2.0, 3.0],
            "key": ["v4_LIGHT_1_1", "v4_LIGHT_2_2", "v4_LIGHT_3_3"],
        }
    )
    quad = detections_on_this_file(table, "v4/wafer_shot_1_quad1_exp200000.png")
    shared = detections_on_tile(table, "v4/wafer_shot_1_quad1_exp200000.png")
    assert list(quad["size"]) == [2.0]
    assert list(shared["size"]) == [1.0, 2.0]


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


def test_detections_on_tile_matches_basename() -> None:
    table = pd.DataFrame(
        {
            "source_tile": ["R3_5_17_5X.tif", "other/R3_2_31_5X.tif", "R3_5_17_5X.tif"],
            "size": [1.0, 2.0, 3.0],
        }
    )
    hits = detections_on_tile(table, "R3_5_17_5X.tif")
    assert list(hits["size"]) == [1.0, 3.0]


def test_detections_on_tile_share_nsew_hits() -> None:
    table = pd.DataFrame(
        {
            "source_tile": ["N.bmp", "loose.tif"],
            "size": [10.0, 11.0],
            "x_global": [100.0, 1.0],
            "y_global": [200.0, 2.0],
            "id": [1, 2],
            "confidence": [0.9, 0.8],
            "nsew_dirs": ["N,S,E", ""],
        }
    )
    hits = detections_on_tile(table, "S.bmp")
    assert list(hits["size"]) == [10.0]
    assert list(hits["source_tile"]) == ["S.bmp"]
    assert expand_direction_tile_names(["N.bmp"], ["N.bmp", "S.bmp", "E.bmp", "W.bmp"]) == [
        "N.bmp",
        "S.bmp",
        "E.bmp",
        "W.bmp",
    ]


def test_detections_on_tile_does_not_share_particles_only_versions() -> None:
    table = pd.DataFrame(
        {
            "source_tile": [
                "particles_only.png",
                "particles_only_4of4.png",
            ],
            "size": [10.0, 11.0],
            "x_global": [100.0, 100.0],
            "y_global": [200.0, 200.0],
            "id": [1, 2],
            "confidence": [0.9, 0.8],
            "key": ["NSEW_100_200", "NSEW_100_201"],
        }
    )
    two = detections_on_tile(table, "particles_only.png")
    four = detections_on_tile(table, "particles_only_4of4.png")
    assert list(two["id"]) == [1]
    assert list(four["id"]) == [2]


def test_labeled_detections_keeps_particle_keys_and_crops() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [10.0, 20.0, 30.0],
            "y_global": [11.0, 21.0, 31.0],
            "size": [16000.0, 17000.0, 18000.0],
            "confidence": [0.9, 0.8, 0.7],
            "source_tile": ["a.tif", "b.tif", "a.tif"],
            "key": ["a_10_11", "b_20_21", "a_30_31"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_10_11", "b_20_21", "a_30_31"],
            "label": ["particle", "not_particle", "particle"],
            "crop_path": ["crops/a.jpg", "crops/b.jpg", "crops/c.jpg"],
            "particle_id": [1, 2, 3],
        }
    )
    hits = labeled_detections(detections, labels, label="particle")
    assert list(hits["key"]) == ["a_10_11", "a_30_31"]
    assert list(hits["crop_path"]) == ["crops/a.jpg", "crops/c.jpg"]
    assert tile_names_for_hits(hits) == ["a.tif"]


def test_labeled_detections_empty_when_no_overlap() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [16000.0],
            "confidence": [0.9],
            "source_tile": ["a.tif"],
            "key": ["a_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["other"],
            "label": ["particle"],
            "crop_path": ["crops/x.jpg"],
            "particle_id": [9],
        }
    )
    hits = labeled_detections(detections, labels)
    assert hits.empty
    assert tile_names_for_hits(hits) == []


def test_labeled_particle_recovery_counts_last_run_keys_only() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2],
            "x_global": [10_000.0, 50_000.0],
            "y_global": [11_000.0, 51_000.0],
            "size": [16000.0, 17000.0],
            "confidence": [0.9, 0.8],
            "source_tile": ["a.tif", "a.tif"],
            "key": ["a_10000_11000", "a_50000_51000"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_10000_11000", "a_90000_90000", "b_1_1"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["a.tif", "a.tif", "other.tif"],
            "x_global": [10_000.0, 90_000.0, 1.0],
            "y_global": [11_000.0, 90_000.0, 1.0],
        }
    )
    stats = labeled_particle_recovery(detections, labels, tile_names=["a.tif"])
    assert stats == {"n_labeled": 1, "n_found": 1, "pct": 100.0}


def test_labeled_particle_recovery_ignores_older_keys_not_in_run() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10_000.0],
            "y_global": [11_000.0],
            "size": [16000.0],
            "confidence": [0.9],
            "source_tile": ["a.tif"],
            "key": ["a_10000_11000"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_10000_11000", "old_tophat", "old_ml"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["a.tif", "a.tif", "a.tif"],
            "x_global": [10_000.0, 10_005.0, 90_000.0],
            "y_global": [11_000.0, 11_002.0, 90_000.0],
        }
    )
    stats = labeled_particle_recovery(detections, labels, tile_names=["a.tif"])
    assert stats == {"n_labeled": 1, "n_found": 1, "pct": 100.0}


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
    assert stats["unhappiness"] == unhappiness_score(1 / 3, 1 / 3)
    assert stats["unhappiness"] == 100.0 * (
        (1 / 3 - 0.95) ** 2 + (1 / 3 - 0.95) ** 2
    )


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
    assert float(by_tile.loc["a.tif", "unhappiness"]) == unhappiness_score(0.5, 0.5)
    assert int(by_tile.loc["b.tif", "n_detected"]) == 1
    assert float(by_tile.loc["b.tif", "precision"]) == 1.0
    assert float(by_tile.loc["b.tif", "recall"]) == 1.0
    assert float(by_tile.loc["b.tif", "unhappiness"]) == unhappiness_score(1.0, 1.0)


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


def test_crop_metric_std_uses_defined_cells_only() -> None:
    placement = TilePlacement(
        path=Path("a.tif"),
        name="a.tif",
        y0=0,
        x0=0,
        height=90,
        width=90,
    )
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "x_global": [10.0, 12.0, 40.0, 70.0],
            "y_global": [10.0, 12.0, 10.0, 70.0],
            "size": [20000.0, 20000.0, 20000.0, 20000.0],
            "confidence": [0.9, 0.8, 0.7, 0.6],
            "source_tile": ["a.tif", "a.tif", "a.tif", "a.tif"],
            "key": ["real_a", "fake_a", "real_b", "fake_c"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["real_a", "fake_a", "real_b", "miss_a", "fake_c"],
            "label": ["particle", "not_particle", "particle", "particle", "not_particle"],
            "source_tile": ["a.tif", "a.tif", "a.tif", "a.tif", "a.tif"],
            "size": [20000.0, 20000.0, 20000.0, 25000.0, 20000.0],
            "x_global": [10.0, 12.0, 40.0, 15.0, 70.0],
            "y_global": [10.0, 12.0, 10.0, 15.0, 70.0],
        }
    )
    spread = crop_metric_std(
        detections,
        labels,
        height=90,
        width=90,
        placements=[placement],
        pixel_size_nm=1.0,
        tile_names=["a.tif"],
        min_size_nm=20000.0,
        beta=1.0,
    )
    pair = float(pd.Series([0.5, 1.0]).std(ddof=1))
    three = float(pd.Series([0.5, 1.0, 0.0]).std(ddof=1))
    assert spread["precision"] == pytest.approx(three)
    assert spread["recall"] == pytest.approx(pair)
    assert spread["f_beta"] == pytest.approx(pair)
    including_empty = float(pd.Series([0.5, 1.0, 0.0, *([0.0] * 6)]).std(ddof=1))
    assert spread["precision"] != pytest.approx(including_empty)

    one_cell = detections.loc[detections["key"] == "real_b"]
    one_label = labels.loc[labels["key"] == "real_b"]
    single = crop_metric_std(
        one_cell,
        one_label,
        height=90,
        width=90,
        placements=[placement],
        pixel_size_nm=1.0,
        tile_names=["a.tif"],
        beta=1.0,
    )
    assert single == {"precision": None, "recall": None, "f_beta": None}
    cells = crop_metric_cells(
        detections,
        labels,
        height=90,
        width=90,
        placements=[placement],
        pixel_size_nm=1.0,
        tile_names=["a.tif"],
        min_size_nm=20000.0,
        beta=1.0,
    )
    assert len(cells) == 9
    assert cells[0]["precision"] == pytest.approx(0.5)
    assert cells[0]["recall"] == pytest.approx(0.5)
    assert cells[1]["precision"] == pytest.approx(1.0)
    assert cells[1]["recall"] == pytest.approx(1.0)
    assert cells[8]["precision"] == pytest.approx(0.0)
    assert cells[8]["recall"] is None


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
    # Even without resolving the real Inputs tree, png particles-only
    # still keeps N.png labels (not v2 N.bmp).
    assert "N.png" in relative
    assert "N.bmp" not in relative


def test_unhappiness_score_is_zero_at_target() -> None:
    assert unhappiness_score(0.95, 0.95) == 0.0
    assert unhappiness_score(1.0, 1.0) == 100.0 * (
        (1.0 - 0.95) ** 2 + (1.0 - 0.95) ** 2
    )


def test_undetected_size_floor_uses_this_run_minimum() -> None:
    detections = pd.DataFrame({"size": [20000.0, 31000.0]})
    floor = undetected_size_floor_nm(detections, {"detection": {"min_size_nm": 10000.0}})
    assert floor == 20000.0


def test_labeled_particle_recovery_ignores_non_particle_labels() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [16000.0],
            "confidence": [0.9],
            "source_tile": ["a.tif"],
            "key": ["a_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_10_11"],
            "label": ["not_particle"],
            "source_tile": ["a.tif"],
            "x_global": [10.0],
            "y_global": [11.0],
        }
    )
    stats = labeled_particle_recovery(detections, labels)
    assert stats == {"n_labeled": 0, "n_found": 0, "pct": 0.0}


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


def test_overlay_draws_labeled_circle_in_crop() -> None:
    image = np.full((80, 90), 0.2, dtype=np.float32)
    image[20:28, 30:38] = 1.0
    rgb = overlay_circles(
        image,
        [
            {
                "x_local": 34.0,
                "y_local": 24.0,
                "size": 20.0,
                "color": (40, 180, 70),
            }
        ],
        pixel_size_nm=1.0,
        crop=(0, 0, 80, 90),
        max_side=None,
    )
    assert rgb.shape == (80, 90, 3)
    green = (rgb[:, :, 1] > 150) & (rgb[:, :, 0] < 80) & (rgb[:, :, 2] < 100)
    assert int(green.sum()) > 20


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


def test_own_tile_only_keeps_misses_on_that_file() -> None:
    from src.verification.ui import _label_rows_missing_from_hits

    labels = pd.DataFrame(
        {
            "key": ["v4_LIGHT_1_1", "v4_LIGHT_2_2", "v4_LIGHT_9_9", "v4_LIGHT_3_3"],
            "source_tile": [
                "v4/wafer_shot_4_all_exp200000.png",
                "v4/wafer_shot_0_quad0_exp200000.png",
                "v4/wafer_shot_4_all_exp200000.png",
                "v4/wafer_shot_4_all_exp200000.png",
            ],
            "label": ["particle", "particle", "particle", "not_particle"],
            "x_global": [1.0, 2.0, 9.0, 3.0],
            "y_global": [1.0, 2.0, 9.0, 3.0],
            "size": [20_000.0, 20_000.0, 20_000.0, 20_000.0],
        }
    )
    hits = pd.DataFrame(
        {
            "key": ["v4_LIGHT_1_1"],
            "source_tile": ["v4/wafer_shot_4_all_exp200000.png"],
        }
    )
    tile = "v4/wafer_shot_4_all_exp200000.png"
    missing = _label_rows_missing_from_hits(labels, tile, hits, own_tile_only=True)
    assert missing is not None
    assert list(missing["key"]) == ["v4_LIGHT_9_9"]
    shared = _label_rows_missing_from_hits(labels, tile, hits, own_tile_only=False)
    assert shared is not None
    assert "v4_LIGHT_2_2" in set(shared["key"].astype(str))


def test_display_xy_to_local_inverts_overlay_scale() -> None:
    view = OverlayView(rgb=np.zeros((50, 80, 3), dtype=np.uint8), crop_x0=10, crop_y0=20, scale=0.5)
    x_local, y_local = display_xy_to_local(15.0, 8.0, view)
    assert x_local == 10.0 + 15.0 / 0.5
    assert y_local == 20.0 + 8.0 / 0.5


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


def test_pointer_groundup_folder_lists_combi_and_uniform_only(tmp_path: Path) -> None:
    from src.verification.inspect import combi_uniform_paths

    folder = tmp_path / "v5"
    folder.mkdir()
    names = [
        "wafer_shot_0_quad0_exp200000.png",
        "wafer_shot_1_quad1_exp200000.png",
        "wafer_shot_4_all_exp200000.png",
        "combi_2of4.png",
        "notes.txt",
    ]
    paths = []
    for name in names:
        path = folder / name
        path.write_bytes(b"x")
        if path.suffix == ".png":
            paths.append(path)
    chosen = combi_uniform_paths(paths)
    assert chosen is not None
    assert [path.name for path in chosen] == [
        "combi_2of4.png",
        "wafer_shot_4_all_exp200000.png",
    ]
    letters = [tmp_path / "N.png", tmp_path / "S.png"]
    for path in letters:
        path.write_bytes(b"x")
    assert combi_uniform_paths(letters) is None


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
