"""Tests for labeling crops, queue filtering, and the on-disk store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from src.verification.crops import (
    CIRCLE_COLOR,
    TileImageCache,
    config_for_hit_placement,
    crop_direction_views,
    crop_particle,
    crop_window,
    find_tile_path,
    global_nm_to_local_px,
)
from src.verification.inspect import detections_on_tile
from src.verification.queue import (
    MIN_SIZE_NM,
    detection_key,
    filter_min_size,
    tag_table,
    unlabeled_queue,
    nsew_key_aliases,
    share_exposure_keys,
    use_last_pipeline_queue,
)
from src.verification.store import (
    LabelStore,
    crop_source_folder,
    label_campaign,
    newest_first,
    rows_for_campaign,
)
from src.main_pipeline.stitching.stitcher import TilePlacement
from tests.fixtures.synthetic import write_tile_tiff

PATTERN = r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?"


def _config(folder: Path, pixel_size_nm: float = 1.0) -> dict:
    return {
        "input_dir": str(folder),
        "filename_pattern": PATTERN,
        "overlap_fraction": 0.0,
        "pixel_size_nm": pixel_size_nm,
    }


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels close to the unlabeled orange marker (legacy name)."""
    return (
        (rgb[:, :, 0] > 200)
        & (rgb[:, :, 1] > 80)
        & (rgb[:, :, 1] < 180)
        & (rgb[:, :, 2] < 80)
    )


def test_global_nm_to_local_px_subtracts_origin() -> None:
    placement = TilePlacement(
        path=Path("R3_1_2_5X.tif"),
        name="R3_1_2_5X.tif",
        y0=100,
        x0=80,
        height=128,
        width=128,
    )
    x_local, y_local = global_nm_to_local_px(960.0, 480.0, placement, pixel_size_nm=2.0)
    assert x_local == 960.0 / 2.0 - 80
    assert y_local == 480.0 / 2.0 - 100


def test_find_tile_path_accepts_bmp_for_tif_name(tmp_path: Path) -> None:
    import cv2

    gray = np.zeros((8, 8), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "R3_1_1_5X.bmp"), gray)
    found = find_tile_path("R3_1_1_5X.tif", input_dir=tmp_path)
    assert found.name == "R3_1_1_5X.bmp"


def test_crop_window_clamps_to_tile_edges() -> None:
    y0, x0, y1, x1 = crop_window(5.0, 4.0, diameter_px=10.0, height=80, width=90)
    assert (y0, x0) == (0, 0)
    assert 0 < y1 <= 80
    assert 0 < x1 <= 90
    assert y1 - y0 <= 80
    assert x1 - x0 <= 90


def test_hit_placement_overlap_puts_the_circle_on_the_crop(tmp_path: Path) -> None:
    image = np.zeros((80, 100), dtype=np.float32)
    write_tile_tiff(tmp_path / "R3_1_1_5X.tif", image)
    write_tile_tiff(tmp_path / "R3_1_2_5X.tif", image)
    row = {
        "id": 1,
        "source_tile": "R3_1_2_5X.tif",
        "x_global": 95.0,
        "y_global": 40.0,
        "size": 10.0,
        "confidence": 1.0,
    }
    config = _config(tmp_path, pixel_size_nm=1.0)
    config["overlap_fraction"] = 0.0
    placed = config_for_hit_placement(config, pd.DataFrame([row]))
    assert float(placed["overlap_fraction"]) > 0.0
    crop = crop_particle(row, placed)
    assert int(_red_mask(crop.rgb).sum()) > 20


def test_crop_window_stays_local_when_point_is_outside_the_tile() -> None:
    y0, x0, y1, x1 = crop_window(-5000.0, 229.0, diameter_px=40.0, height=2076, width=3088)
    assert x1 - x0 == 192
    assert y1 - y0 == 192
    assert x0 == 0
    assert y0 <= 229 < y1


def test_crop_keeps_file_gray_levels(tmp_path: Path) -> None:
    image = np.full((80, 90), 80, dtype=np.uint8)
    image[2:8, 3:8] = 110
    path = tmp_path / "R3_1_1_5X.tif"
    tifffile.imwrite(path, image)
    row = {
        "id": 1,
        "source_tile": path.name,
        "x_global": 40.0,
        "y_global": 40.0,
        "size": 10.0,
        "confidence": 1.0,
    }
    crop = crop_particle(row, _config(tmp_path))
    assert int(crop.rgb[0, 0, 0]) == 80
    assert int(crop.rgb[0, 0, 1]) == 80


def test_crop_contains_blob_and_clamps_at_edge(tmp_path: Path) -> None:
    image = np.zeros((80, 90), dtype=np.float32)
    image[2:8, 3:8] = 1.0
    path = write_tile_tiff(tmp_path / "R3_1_1_5X.tif", image)
    row = {
        "id": 1,
        "source_tile": path.name,
        "x_global": 5.0,
        "y_global": 5.0,
        "size": 10.0,
        "confidence": 1.0,
    }
    crop = crop_particle(row, _config(tmp_path))
    assert crop.crop_x0 == 0
    assert crop.crop_y0 == 0
    assert abs(crop.x_local - 5.0) < 1.0
    assert abs(crop.y_local - 5.0) < 1.0
    assert crop.rgb.shape[0] == 80
    assert crop.rgb.shape[1] == 90
    local_x = int(round(crop.x_local)) - crop.crop_x0
    local_y = int(round(crop.y_local)) - crop.crop_y0
    assert 0 <= local_x < crop.rgb.shape[1]
    assert 0 <= local_y < crop.rgb.shape[0]


def test_overlay_draws_one_circle(tmp_path: Path) -> None:
    image = np.zeros((128, 128), dtype=np.float32)
    image[38:43, 38:43] = 1.0
    image[88:93, 88:93] = 1.0
    path = write_tile_tiff(tmp_path / "R3_1_1_5X.tif", image)
    row = {
        "id": 2,
        "source_tile": path.name,
        "x_global": 40.0,
        "y_global": 40.0,
        "size": 20.0,
        "confidence": 1.0,
    }
    crop = crop_particle(row, _config(tmp_path))
    marker = _red_mask(crop.rgb)
    assert int(marker.sum()) > 20
    yy, xx = np.nonzero(marker)
    assert np.mean(np.hypot(yy - 40.0, xx - 40.0)) < 30.0
    other = marker[85:96, 85:96]
    assert int(other.sum()) == 0


def test_tile_cache_reuses_decoded_array(tmp_path: Path) -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[10:14, 10:14] = 1.0
    path = write_tile_tiff(tmp_path / "R3_1_1_5X.tif", image)
    cache = TileImageCache()
    row = {
        "id": 1,
        "source_tile": path.name,
        "x_global": 12.0,
        "y_global": 12.0,
        "size": 8.0,
        "confidence": 1.0,
    }
    crop_particle(row, _config(tmp_path), cache=cache)
    first = cache.image
    crop_particle(row, _config(tmp_path), cache=cache)
    assert cache.image is first


def test_min_size_filter_drops_below_15_um() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [1.0, 2.0, 3.0],
            "y_global": [1.0, 2.0, 3.0],
            "size": [9_999.0, 10_000.0, 20_000.0],
            "confidence": [1.0, 1.0, 1.0],
            "source_tile": ["A.tif", "B.tif", "C.tif"],
        }
    )
    kept = filter_min_size(df, MIN_SIZE_NM)
    assert list(kept["id"]) == [2, 3]
    assert float(kept["size"].min()) >= MIN_SIZE_NM


def test_unlabeled_queue_skips_labeled_keys() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "x_global": [100.0, 200.0],
            "y_global": [10.0, 20.0],
            "size": [20_000.0, 21_000.0],
            "confidence": [1.0, 1.0],
            "source_tile": ["R3_1_1_5X.tif", "R3_1_1_5X.tif"],
        }
    )
    tagged = tag_table(df, "run.csv")
    skip = {detection_key(tagged.iloc[0])}
    queue = unlabeled_queue([tagged], skip)
    assert len(queue) == 1
    assert int(queue.iloc[0]["id"]) == 2


def test_nsew_label_covers_all_four_directions(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    south = {
        "id": 1,
        "source_tile": "S.bmp",
        "x_global": 100.4,
        "y_global": 200.4,
        "size": 20_000.0,
        "confidence": 0.8,
        "source_csv": "particles.csv",
    }
    store.apply_label(south, "particle", rgb)
    assert detection_key(south) == "NSEW_100_200"
    assert store.labeled_keys() == {"NSEW_100_200"}
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [100.4, 100.4, 50.0],
            "y_global": [200.4, 200.4, 50.0],
            "size": [20_000.0, 21_000.0, 22_000.0],
            "confidence": [0.9, 0.7, 0.6],
            "source_tile": ["N.bmp", "E.bmp", "loose.tif"],
        }
    )
    queue = unlabeled_queue([tag_table(detections)], store.labeled_keys())
    assert list(queue["source_tile"]) == ["loose.tif"]
    hits = detections_on_tile(tag_table(detections), "W.bmp")
    assert "NSEW_100_200" in set(hits["key"].astype(str))
    assert set(hits["source_tile"].astype(str)) == {"W.bmp"}


def test_particles_only_shares_nsew_key_and_skips_labeled_location(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    north = {
        "id": 1,
        "source_tile": "N.png",
        "x_global": 100.4,
        "y_global": 200.4,
        "size": 20_000.0,
        "confidence": 0.8,
        "source_csv": "particles.csv",
    }
    store.apply_label(north, "particle", rgb)
    only = {
        "id": 2,
        "source_tile": "particles_only_2of4.png",
        "x_global": 100.4,
        "y_global": 200.4,
        "size": 21_000.0,
        "confidence": 0.9,
    }
    four = {
        "id": 3,
        "source_tile": "particles_only_4of4.png",
        "x_global": 100.4,
        "y_global": 200.4,
        "size": 22_000.0,
        "confidence": 0.7,
    }
    other = {
        "id": 4,
        "source_tile": "particles_only_2of4.png",
        "x_global": 1_000_000.0,
        "y_global": 50.0,
        "size": 23_000.0,
        "confidence": 0.6,
    }
    assert detection_key(only) == "NSEW_100_200"
    assert detection_key(four) == detection_key(north)
    detections = pd.DataFrame([only, four, other])
    queue = unlabeled_queue(
        [tag_table(detections)],
        store.labeled_keys(),
        nsew_merge_radius_nm=24.0,
        nsew_size_match_fraction=0.4,
    )
    assert list(queue["source_tile"]) == ["particles_only_2of4.png"]
    assert float(queue.iloc[0]["x_global"]) == 1_000_000.0


def test_snap_keys_reuses_nearby_nsew_label() -> None:
    from src.verification.queue import snap_detection_keys_to_labels

    labels = pd.DataFrame(
        {
            "key": ["NSEW_100_200"],
            "label": ["particle"],
            "source_tile": ["N.png"],
            "x_global": [100.0],
            "y_global": [200.0],
            "size": [20_000.0],
        }
    )
    detections = pd.DataFrame(
        {
            "id": [1, 2],
            "source_tile": ["particles_only_2of4.png", "particles_only_2of4.png"],
            "x_global": [118.0, 1_000_000.0],
            "y_global": [204.0, 50.0],
            "size": [21_000.0, 22_000.0],
            "confidence": [0.9, 0.8],
        }
    )
    snapped = snap_detection_keys_to_labels(
        tag_table(detections), labels, merge_radius_nm=24.0, size_match_fraction=0.0
    )
    assert str(snapped.iloc[0]["key"]) == "NSEW_100_200"
    assert str(snapped.iloc[1]["key"]) != "NSEW_100_200"
    queue = unlabeled_queue(
        [tag_table(detections)],
        labels["key"],
        nsew_merge_radius_nm=24.0,
        nsew_size_match_fraction=0.0,
        labels=labels,
    )
    assert list(queue["x_global"]) == [1_000_000.0]


def test_tinder_queue_merges_nwes_offset_hits() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "x_global": [100.0, 118.0, 108.0, 95.0, 800.0],
            "y_global": [200.0, 204.0, 198.0, 205.0, 200.0],
            "size": [20_000.0, 21_000.0, 22_000.0, 19_000.0, 23_000.0],
            "confidence": [0.9, 0.5, 0.7, 0.8, 0.6],
            "source_tile": ["N.bmp", "W.bmp", "E.bmp", "S.bmp", "N.bmp"],
        }
    )
    queue = unlabeled_queue(
        [tag_table(detections)],
        labeled_keys=(),
        nsew_merge_radius_nm=24.0,
        nsew_size_match_fraction=0.4,
    )
    assert len(queue) == 2
    same = queue.loc[queue["x_global"].astype(float) < 400]
    other = queue.loc[queue["x_global"].astype(float) > 400]
    assert len(same) == 1
    assert len(other) == 1
    assert str(same.iloc[0]["key"]).startswith("NSEW_")
    assert int(same.iloc[0]["nsew_count"]) == 4
    assert set(str(same.iloc[0]["nsew_dirs"]).split(",")) == {"N", "W", "E", "S"}


def test_groundup_tinder_keeps_combi_and_uniform(tmp_path: Path) -> None:
    from src.verification.crops import nsew_direction_crops, reviewed_on_combi_or_uniform

    folder = tmp_path / "v5"
    folder.mkdir()
    for name in (
        "wafer_shot_0_quad0_exp200000.png",
        "wafer_shot_1_quad1_exp200000.png",
        "wafer_shot_2_quad2_exp200000.png",
        "wafer_shot_3_quad3_exp200000.png",
        "wafer_shot_4_all_exp200000.png",
        "combi_2of4.png",
    ):
        (folder / name).write_bytes(b"png")
    combi = str(folder / "combi_2of4.png")
    uniform = str(folder / "wafer_shot_4_all_exp200000.png")
    quad = str(folder / "wafer_shot_0_quad0_exp200000.png")
    assert reviewed_on_combi_or_uniform(combi)
    assert reviewed_on_combi_or_uniform(uniform)
    assert not reviewed_on_combi_or_uniform(quad)
    row = {
        "id": 1,
        "x_global": 10.0,
        "y_global": 20.0,
        "size": 8.0,
        "confidence": 0.9,
    }
    assert nsew_direction_crops({**row, "source_tile": combi}, _config(folder)) == {}
    assert nsew_direction_crops({**row, "source_tile": uniform}, _config(folder)) == {}


def test_nsew_direction_crops_shows_nsew_for_combined(tmp_path: Path) -> None:
    import cv2

    from src.verification.crops import nsew_direction_crops

    angles = tmp_path / "Inputs" / "Groundup v5" / "v5 (10^5)"
    stage = tmp_path / "Inputs" / "Groundup v5 combi v5 v6"
    angles.mkdir(parents=True)
    stage.mkdir()
    for name in ("N", "S", "E", "W"):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[28:36, 28:36] = 180
        assert cv2.imwrite(str(angles / f"{name}.png"), gray)
    assert cv2.imwrite(str(stage / "v5_2of4.png"), np.zeros((64, 64), dtype=np.uint8))
    row = {
        "id": 1,
        "source_tile": "v5_2of4.png",
        "x_global": 32.0,
        "y_global": 32.0,
        "size": 8.0,
        "confidence": 0.9,
    }
    views = nsew_direction_crops(row, _config(stage))
    assert list(views) == ["N", "S", "E", "W"]
    assert all(view.direction == name for name, view in views.items())
    assert all(_red_mask(view.rgb).any() for view in views.values())


def test_crop_direction_views_shows_nwes(tmp_path: Path) -> None:
    import cv2

    for name in ("N", "W", "E", "S"):
        gray = np.zeros((48, 48), dtype=np.uint8)
        gray[20:28, 20:28] = 200
        assert cv2.imwrite(str(tmp_path / f"{name}.bmp"), gray)
    row = {
        "id": 1,
        "source_tile": "N.bmp",
        "x_global": 24.0,
        "y_global": 24.0,
        "size": 8.0,
        "confidence": 0.9,
    }
    views = crop_direction_views(row, _config(tmp_path))
    assert list(views) == ["N", "W", "E", "S"]
    assert all(view.rgb.size > 0 for view in views.values())


def test_brightest_direction_crop_picks_the_bright_angle(tmp_path: Path) -> None:
    import cv2

    from src.verification.crops import brightest_direction_crop

    angles = tmp_path / "Inputs" / "Groundup v4" / "v3"
    stage = tmp_path / "Inputs" / "Groundup v4 ML off"
    angles.mkdir(parents=True)
    stage.mkdir()
    for name, value in (("N", 20), ("S", 40), ("E", 220), ("W", 30)):
        gray = np.full((64, 64), 5, dtype=np.uint8)
        if name == "E":
            gray[28:36, 28:36] = value
        assert cv2.imwrite(str(angles / f"{name}.bmp"), gray)
    mask = np.zeros((64, 64), dtype=np.uint8)
    assert cv2.imwrite(str(stage / "v3_2of4.bmp"), mask)
    row = {
        "id": 1,
        "source_tile": "v3_2of4.bmp",
        "x_global": 32.0,
        "y_global": 32.0,
        "size": 8.0,
        "confidence": 0.9,
    }
    crop = brightest_direction_crop(row, _config(stage))
    assert crop is not None
    assert crop.direction == "E"
    assert crop.tile_path.name == "E.bmp"
    assert _red_mask(crop.rgb).any()


def test_brightest_direction_crop_uses_groundup_v5_exposure_folder(tmp_path: Path) -> None:
    import cv2

    from src.verification.crops import brightest_direction_crop

    angles = tmp_path / "Inputs" / "Groundup v5" / "v3 (10^5)"
    stage = tmp_path / "Inputs" / "Groundup v5 combi"
    angles.mkdir(parents=True)
    stage.mkdir()
    for name, value in (("N", 20), ("S", 40), ("E", 220), ("W", 30)):
        gray = np.full((64, 64), 5, dtype=np.uint8)
        if name == "E":
            gray[28:36, 28:36] = value
        assert cv2.imwrite(str(angles / f"{name}.png"), gray)
    mask = np.zeros((64, 64), dtype=np.uint8)
    assert cv2.imwrite(str(stage / "v3_2of4.png"), mask)
    row = {
        "id": 1,
        "source_tile": "v3_2of4.png",
        "x_global": 32.0,
        "y_global": 32.0,
        "size": 8.0,
        "confidence": 0.9,
    }
    crop = brightest_direction_crop(row, _config(stage))
    assert crop is not None
    assert crop.direction == "E"
    assert crop.tile_path.parent == angles


def test_direction_crop_prefers_contrast_over_raw_brightness(tmp_path: Path) -> None:
    import cv2

    from src.verification.crops import brightest_direction_crop

    angles = tmp_path / "Inputs" / "Groundup v4" / "v3"
    stage = tmp_path / "Inputs" / "Groundup v4 ML off"
    angles.mkdir(parents=True)
    stage.mkdir()
    yy, xx = np.ogrid[:64, :64]
    disk = (xx - 32) ** 2 + (yy - 32) ** 2 <= 4 ** 2
    # N is brighter overall, but the speck barely clears its background.
    north = np.full((64, 64), 180, dtype=np.uint8)
    north[disk] = 200
    # W is darker, and the speck stands further above the ring around it.
    west = np.full((64, 64), 20, dtype=np.uint8)
    west[disk] = 90
    flat = np.full((64, 64), 30, dtype=np.uint8)
    for name, image in (("N", north), ("S", flat), ("E", flat), ("W", west)):
        assert cv2.imwrite(str(angles / f"{name}.bmp"), image)
    assert cv2.imwrite(str(stage / "v3_2of4.bmp"), np.zeros((64, 64), dtype=np.uint8))
    row = {
        "id": 1,
        "source_tile": "v3_2of4.bmp",
        "x_global": 32.0,
        "y_global": 32.0,
        "size": 8.0,
        "confidence": 0.9,
    }
    crop = brightest_direction_crop(row, _config(stage))
    assert crop is not None
    assert crop.direction == "W"


def test_uniform_hit_is_not_moved_onto_a_quad() -> None:
    from src.verification.crops import brightest_prominence_direction_row

    detections = pd.DataFrame(
        [
            {
                "source_tile": "v4/wafer_shot_0_quad0_exp200000.png",
                "x_global": 1000.0,
                "y_global": 2000.0,
                "size": 20_000.0,
                "confidence": 0.40,
            },
            {
                "source_tile": "v4/wafer_shot_1_quad1_exp200000.png",
                "x_global": 1100.0,
                "y_global": 1900.0,
                "size": 20_000.0,
                "confidence": 0.91,
            },
            {
                "source_tile": "v4/wafer_shot_2_quad2_exp200000.png",
                "x_global": 900.0,
                "y_global": 2100.0,
                "size": 20_000.0,
                "confidence": 0.55,
            },
            {
                "source_tile": "v4/wafer_shot_4_all_exp200000.png",
                "x_global": 1000.0,
                "y_global": 2000.0,
                "size": 20_000.0,
                "confidence": 0.99,
            },
            {
                "source_tile": "v5/wafer_shot_1_quad1_exp200000.png",
                "x_global": 1000.0,
                "y_global": 2000.0,
                "size": 20_000.0,
                "confidence": 0.98,
            },
        ]
    )
    record = {
        "source_tile": "v4/wafer_shot_4_all_exp200000.png",
        "x_global": 1000.0,
        "y_global": 2000.0,
        "size": 20_000.0,
        "confidence": 0.99,
    }
    chosen = brightest_prominence_direction_row(
        record,
        detections,
        merge_radius_nm=5_000.0,
        size_match_fraction=0.4,
    )
    assert chosen is None


def test_groundup_review_picks_combi_or_uniform_by_prominence() -> None:
    from src.verification.crops import brightest_combi_or_uniform_row

    detections = pd.DataFrame(
        [
            {
                "source_tile": "v6/wafer_shot_0_quad0_exp200000.png",
                "x_global": 1000.0,
                "y_global": 2000.0,
                "size": 20_000.0,
                "confidence": 0.99,
            },
            {
                "source_tile": "v6/wafer_shot_1_quad1_exp200000.png",
                "x_global": 1100.0,
                "y_global": 1900.0,
                "size": 20_000.0,
                "confidence": 0.97,
            },
            {
                "source_tile": "Groundup/Groundup v7 (06-10)/v6/combi_2of4.png",
                "x_global": 1000.0,
                "y_global": 2000.0,
                "size": 20_000.0,
                "confidence": 0.80,
            },
            {
                "source_tile": "v6/wafer_shot_4_all_exp200000.png",
                "x_global": 1050.0,
                "y_global": 2000.0,
                "size": 20_000.0,
                "confidence": 0.91,
            },
            {
                "source_tile": "v5/wafer_shot_4_all_exp200000.png",
                "x_global": 1000.0,
                "y_global": 2000.0,
                "size": 20_000.0,
                "confidence": 0.99,
            },
        ]
    )
    record = detections.iloc[0].to_dict()
    chosen = brightest_combi_or_uniform_row(
        record,
        detections,
        merge_radius_nm=5_000.0,
        size_match_fraction=0.4,
    )
    assert chosen is not None
    assert chosen["source_tile"] == "v6/wafer_shot_4_all_exp200000.png"
    assert chosen["confidence"] == 0.91

    detections.loc[detections["source_tile"].astype(str).str.contains("combi"), "confidence"] = 0.95
    chosen = brightest_combi_or_uniform_row(
        record,
        detections,
        merge_radius_nm=5_000.0,
        size_match_fraction=0.4,
    )
    assert chosen is not None
    assert "combi_2of4.png" in str(chosen["source_tile"])
    assert chosen["confidence"] == 0.95


def test_quad_without_review_detection_uses_the_stronger_image(tmp_path: Path) -> None:
    import cv2

    from src.verification.crops import combi_or_uniform_crop

    folder = tmp_path / "v6"
    folder.mkdir()
    yy, xx = np.ogrid[:64, :64]
    disk = (xx - 32) ** 2 + (yy - 32) ** 2 <= 4 ** 2
    quad = np.full((64, 64), 5, dtype=np.uint8)
    uniform = np.full((64, 64), 30, dtype=np.uint8)
    combi = np.full((64, 64), 20, dtype=np.uint8)
    combi[disk] = 200
    for name, image in (
        ("wafer_shot_0_quad0_exp200000.png", quad),
        ("wafer_shot_4_all_exp200000.png", uniform),
        ("combi_2of4.png", combi),
    ):
        assert cv2.imwrite(str(folder / name), image)
    row = {
        "id": 1,
        "source_tile": "v6/wafer_shot_0_quad0_exp200000.png",
        "x_global": 32.0,
        "y_global": 32.0,
        "size": 8.0,
        "confidence": 0.99,
    }
    picked = combi_or_uniform_crop(
        row,
        _config(folder),
        candidates=pd.DataFrame([row]),
        merge_radius_nm=24.0,
        size_match_fraction=0.4,
    )
    assert picked is not None
    crop, prominence = picked
    assert crop.direction == "Combi"
    assert crop.tile_path.name == "combi_2of4.png"
    assert prominence == 0.99
    assert _red_mask(crop.rgb).any()


def test_store_round_trip_and_undo(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[:, :] = CIRCLE_COLOR
    record = {
        "id": 7,
        "source_tile": "R3_1_1_5X.tif",
        "x_global": 111.4,
        "y_global": 222.6,
        "size": 20_000.0,
        "confidence": 0.99,
        "source_csv": "particles.csv",
    }
    key = detection_key(record)
    dest = store.apply_label(record, "particle", rgb)
    assert dest.is_file()
    assert key in store.labeled_keys()
    loaded = store.by_label("particle")
    assert len(loaded) == 1
    assert int(loaded.iloc[0]["particle_id"]) == 7

    restored = store.undo_last()
    assert restored is not None
    assert str(restored["key"]) == key
    assert key not in store.labeled_keys()
    assert not dest.is_file()
    assert store.by_label("particle").empty


def test_store_apply_labels_writes_particle_and_not_particle(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    particle = {
        "id": 1,
        "source_tile": "R3_1_1_5X.tif",
        "x_global": 10.0,
        "y_global": 20.0,
        "size": 20_000.0,
        "confidence": 0.9,
    }
    other = {
        "id": 2,
        "source_tile": "R3_1_1_5X.tif",
        "x_global": 80.0,
        "y_global": 90.0,
        "size": 25_000.0,
        "confidence": 0.4,
    }
    dests = store.apply_labels(
        [(particle, "particle", rgb), (other, "not_particle", rgb)]
    )
    assert len(dests) == 2
    assert all(path.is_file() for path in dests)
    counts = store.counts()
    assert counts["particle"] == 1
    assert counts["not_particle"] == 1
    assert detection_key(particle) in store.labeled_keys()
    assert detection_key(other) in store.labeled_keys()


def test_empty_label_queue_hides_this_file_until_restore_or_rewrite(tmp_path: Path) -> None:
    from src.verification.queue import (
        clear_label_queue,
        label_queue_is_cleared,
        restore_label_queue,
    )

    csv_path = tmp_path / "run" / "particles.csv"
    csv_path.parent.mkdir()
    csv_path.write_text("id\n", encoding="utf-8")
    pointer_dir = tmp_path / "Outputs"
    assert not label_queue_is_cleared(csv_path, pointer_dir)
    assert not label_queue_is_cleared(None, pointer_dir)

    clear_label_queue(csv_path, pointer_dir)
    assert label_queue_is_cleared(csv_path, pointer_dir)

    other = tmp_path / "other" / "particles.csv"
    other.parent.mkdir()
    other.write_text("id\n", encoding="utf-8")
    assert not label_queue_is_cleared(other, pointer_dir)

    csv_path.write_text("id\n1\n", encoding="utf-8")
    assert not label_queue_is_cleared(csv_path, pointer_dir)

    clear_label_queue(csv_path, pointer_dir)
    restore_label_queue(pointer_dir)
    assert not label_queue_is_cleared(csv_path, pointer_dir)


def test_label_queue_waits_until_load_last_run(tmp_path: Path) -> None:
    csv_path = tmp_path / "particles.csv"
    csv_path.write_text("id\n", encoding="utf-8")
    assert not use_last_pipeline_queue(None, csv_path)
    assert not use_last_pipeline_queue("", csv_path)
    assert use_last_pipeline_queue("last_pipeline", csv_path)
    assert not use_last_pipeline_queue("sidebar", csv_path)
    assert not use_last_pipeline_queue(None, tmp_path / "missing.csv")


def test_collection_is_newest_first_and_names_the_folder(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    older = {
        "id": 1,
        "source_tile": "v3_2of4.png",
        "x_global": 1.0,
        "y_global": 2.0,
        "size": 20_000.0,
        "confidence": 0.5,
        "source_csv": "/Outputs/Groundup v5 v3 v4/particles.csv",
    }
    newer = {
        "id": 2,
        "source_tile": "v4/N.png",
        "x_global": 3.0,
        "y_global": 4.0,
        "size": 30_000.0,
        "confidence": 0.6,
        "source_csv": "tile_inspect",
    }
    store.apply_label(older, "particle", rgb)
    store.apply_label(newer, "particle", rgb)
    loaded = store.by_label("particle")
    assert list(loaded["particle_id"].astype(int)) == [2, 1]
    assert crop_source_folder(older["source_csv"], older["source_tile"]) == "Groundup v5 v3 v4"
    assert crop_source_folder(newer["source_csv"], newer["source_tile"]) == "v4"
    assert crop_source_folder("allemaal 08/09/particles.csv") == "allemaal 08/09"
    plain = pd.DataFrame({"labeled_at": ["2026-09-01", "2026-09-24"], "id": [1, 2]})
    assert list(newest_first(plain)["id"]) == [2, 1]


def test_last_run_csv_queue_floor_is_10_um() -> None:
    from src.verification.queue import last_run_csv, load_particles_csv

    path = last_run_csv(Path(__file__).resolve().parents[1])
    assert path.is_file()
    table = load_particles_csv(path)
    kept = filter_min_size(table)
    assert len(kept) > 0
    assert float(kept["size"].min()) >= MIN_SIZE_NM


def test_last_run_csv_prefers_custom_recall_table(tmp_path: Path) -> None:
    from src.verification.queue import last_run_csv, load_particles_csv

    folder = tmp_path / "my_recall"
    folder.mkdir()
    table = pd.DataFrame(
        {
            "id": [1],
            "x_global": [100.0],
            "y_global": [10.0],
            "size": [20_000.0],
            "confidence": [1.0],
            "source_tile": ["R3_1_1_5X.tif"],
        }
    )
    csv_path = folder / "particles.csv"
    table.to_csv(csv_path, index=False)
    found = last_run_csv(
        tmp_path, {"labeling": {"source_csv": "my_recall/particles.csv"}}
    )
    assert found == csv_path
    loaded = load_particles_csv(found)
    assert len(loaded) == 1
    assert "key" in loaded.columns


def test_load_last_pipeline_uses_pointer_not_recall_csv(tmp_path: Path) -> None:
    from src.verification.queue import load_last_pipeline, write_last_pipeline

    csv_path = tmp_path / "out" / "particles.csv"
    csv_path.parent.mkdir()
    pd.DataFrame(
        {
            "id": [1],
            "x_global": [100.0],
            "y_global": [10.0],
            "size": [20_000.0],
            "confidence": [1.0],
            "source_tile": ["R3_1_1_5X.tif"],
        }
    ).to_csv(csv_path, index=False)
    write_last_pipeline(
        {"input_dir": "My tiles", "output_dir": str(tmp_path / "out"), "pixel_size_nm": 960.0},
        csv_path,
        pointer_dir=tmp_path,
    )
    found, run_config = load_last_pipeline({"input_dir": "other"}, pointer_dir=tmp_path)
    assert found == csv_path.resolve()
    assert run_config["input_dir"] == "My tiles"
    saved = json.loads((tmp_path / "out" / "last_pipeline.json").read_text(encoding="utf-8"))
    assert saved["input_dir"] == "My tiles"
    assert saved["pixel_size_nm"] == 960.0
    assert saved["csv"] == str(csv_path.resolve())
    assert not (tmp_path / "last_pipeline.json").exists()


def test_set_scoped_nsew_label_stays_on_one_field() -> None:
    north = {
        "source_tile": "v1/N.bmp",
        "x_global": 100.0,
        "y_global": 200.0,
        "size": 20_000.0,
        "confidence": 0.0,
    }
    assert detection_key(north) == "v1_NSEW_100_200"
    assert detection_key({**north, "source_tile": "v1/S.bmp"}) == "v1_NSEW_100_200"
    assert detection_key({**north, "source_tile": "v2/N.bmp"}) == "v2_NSEW_100_200"
    assert detection_key({**north, "source_tile": "N.bmp"}) == "NSEW_100_200"
    aliases = nsew_key_aliases("v1_NSEW_100_200")
    assert "v1_2of4_100_200" in aliases
    assert "NSEW_100_200" not in aliases
    assert "v2_2of4_100_200" not in aliases
    detections = pd.DataFrame(
        [
            {
                "id": 1,
                "source_tile": "v1_2of4.bmp",
                "x_global": 100.0,
                "y_global": 200.0,
                "size": 20_000.0,
                "confidence": 0.8,
            },
            {
                "id": 2,
                "source_tile": "v2_2of4.bmp",
                "x_global": 100.0,
                "y_global": 200.0,
                "size": 20_000.0,
                "confidence": 0.8,
            },
        ]
    )
    queue = unlabeled_queue([tag_table(detections)], ["v1_NSEW_100_200"])
    assert list(queue["key"].astype(str)) == ["v2_2of4_100_200"]


def test_exposure_pair_is_one_particle_only_in_that_folder() -> None:
    pixel = 3500.0
    radius = 24.0 * pixel
    rows = pd.DataFrame(
        [
            {
                "id": 1,
                "source_tile": "v3_2of4.png",
                "x_global": 1000.0,
                "y_global": 2000.0,
                "size": 40_000.0,
                "confidence": 0.8,
            },
            {
                "id": 2,
                "source_tile": "v4_2of4.png",
                "x_global": 1000.0 + 3 * pixel,
                "y_global": 2000.0,
                "size": 42_000.0,
                "confidence": 0.7,
            },
            {
                "id": 3,
                "source_tile": "v4_2of4.png",
                "x_global": 1000.0 + 80 * pixel,
                "y_global": 2000.0,
                "size": 30_000.0,
                "confidence": 0.6,
            },
        ]
    )
    linked = share_exposure_keys(tag_table(rows), "Groundup v5 combi", radius, 0.4)
    queue = unlabeled_queue([linked], [])
    assert list(queue["source_tile"]) == ["v3_2of4.png", "v4_2of4.png"]
    assert list(queue["id"]) == [1, 3]
    assert queue.iloc[0]["key"] == linked.loc[linked["id"] == 2, "key"].iloc[0]
    assert str(queue.iloc[0]["exposure_with"]) == "v4"

    separate = share_exposure_keys(tag_table(rows), "Groundup v4 ML off", radius, 0.4)
    other = unlabeled_queue([separate], [])
    assert len(other) == 3


def _submap_shot(set_id: str, role: str, x: float, y: float, particle_id: int) -> dict:
    if role == "all":
        filename = "wafer_shot_4_all_exp200000.png"
    else:
        index = role.removeprefix("quad")
        filename = f"wafer_shot_{index}_quad{index}_exp200000.png"
    return {
        "id": particle_id,
        "source_tile": f"{set_id}/{filename}",
        "x_global": x,
        "y_global": y,
        "size": 20_000.0,
        "confidence": 0.8,
        "source_csv": "particles.csv",
    }


def test_combi_and_uniform_at_the_same_place_are_one_label(tmp_path: Path) -> None:
    folder = tmp_path / "v5"
    folder.mkdir()
    for name in (
        "wafer_shot_0_quad0_exp200000.png",
        "wafer_shot_4_all_exp200000.png",
        "combi_2of4.png",
    ):
        (folder / name).write_bytes(b"png")
    combi = str(folder / "combi_2of4.png")
    uniform = str(folder / "wafer_shot_4_all_exp200000.png")
    rows = pd.DataFrame(
        [
            {
                "id": 1,
                "source_tile": combi,
                "x_global": 100_000.0,
                "y_global": 200_000.0,
                "size": 30_000.0,
                "confidence": 0.9,
            },
            {
                "id": 2,
                "source_tile": uniform,
                "x_global": 110_000.0,
                "y_global": 205_000.0,
                "size": 28_000.0,
                "confidence": 0.7,
            },
            {
                "id": 3,
                "source_tile": combi,
                "x_global": 2_000_000.0,
                "y_global": 200_000.0,
                "size": 30_000.0,
                "confidence": 0.8,
            },
        ]
    )
    queue = unlabeled_queue(
        [tag_table(rows)],
        [],
        nsew_merge_radius_nm=80_000.0,
        nsew_size_match_fraction=0.4,
    )
    assert len(queue) == 2
    assert set(queue["id"].astype(int)) == {1, 3}


def test_submap_label_covers_quads_and_uniform_shot_only(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    marked = _submap_shot("v7", "quad0", 100.4, 200.4, 1)
    store.apply_label(marked, "particle", rgb)
    assert detection_key(marked) == "v7_LIGHT_100_200"
    assert detection_key(_submap_shot("v7", "all", 100.4, 200.4, 2)) == "v7_LIGHT_100_200"
    assert detection_key(_submap_shot("v8", "quad0", 100.4, 200.4, 3)) == "v8_LIGHT_100_200"
    assert "v8_LIGHT_100_200" not in nsew_key_aliases("v7_LIGHT_100_200")
    assert "v7_all_100_200" in nsew_key_aliases("v7_LIGHT_100_200")

    detections = pd.DataFrame(
        [
            _submap_shot("v7", "quad0", 100.0, 200.0, 1),
            _submap_shot("v7", "quad1", 110.0, 204.0, 2),
            _submap_shot("v7", "quad2", 108.0, 198.0, 3),
            _submap_shot("v7", "quad3", 95.0, 205.0, 4),
            _submap_shot("v7", "all", 112.0, 202.0, 5),
            _submap_shot("v7", "quad0", 1_000_000.0, 200.0, 6),
            _submap_shot("v8", "all", 100.0, 200.0, 7),
        ]
    )
    queue = unlabeled_queue(
        [tag_table(detections)],
        store.labeled_keys(),
        nsew_merge_radius_nm=24.0,
        nsew_size_match_fraction=0.4,
        labels=store.load(),
    )
    assert sorted(queue["key"].astype(str)) == sorted(
        [detection_key(_submap_shot("v7", "quad0", 1_000_000.0, 200.0, 6)), "v8_LIGHT_100_200"]
    )
    on_all = detections_on_tile(tag_table(detections), "v7/wafer_shot_4_all_exp200000.png")
    assert "v7_LIGHT_100_200" in set(on_all["key"].astype(str))
    assert "v8_LIGHT_100_200" not in set(on_all["key"].astype(str))
    on_other = detections_on_tile(tag_table(detections), "v8/wafer_shot_4_all_exp200000.png")
    assert set(on_other["key"].astype(str)) == {"v8_LIGHT_100_200"}


def test_submap_folders_load_lighting_shots_only(tmp_path: Path) -> None:
    from src.overall.io.tile_loader import list_tile_paths

    root = tmp_path / "Groundup v7"
    for set_id in ("v7", "v8"):
        folder = root / set_id
        folder.mkdir(parents=True)
        for name in (
            "wafer_shot_0_quad0_exp200000.png",
            "wafer_shot_4_all_exp200000.png",
        ):
            (folder / name).write_bytes(b"png")
    (root / "v7" / "combi_2of4.png").write_bytes(b"png")
    (root / "v7" / "notes.txt").write_text("skip", encoding="utf-8")
    classic = root / "v9"
    classic.mkdir()
    (classic / "N.bmp").write_bytes(b"bmp")
    names = {path.relative_to(root).as_posix() for path in list_tile_paths(root)}
    assert names == {
        "v7/wafer_shot_0_quad0_exp200000.png",
        "v7/wafer_shot_4_all_exp200000.png",
        "v7/combi_2of4.png",
        "v8/wafer_shot_0_quad0_exp200000.png",
        "v8/wafer_shot_4_all_exp200000.png",
    }


def test_tile_folder_follows_the_latest_capture_until_the_user_picks_one() -> None:
    from src.overall.io.tile_loader import follow_latest_capture

    chosen, suggestion, seeded = follow_latest_capture(
        "Glovebox/Not trained on 14-09",
        None,
        "Groundup/v2 2026-10-20",
        seeded=False,
    )
    assert (chosen, suggestion, seeded) == ("Groundup/v2 2026-10-20", "Groundup/v2 2026-10-20", True)

    stayed, suggestion, seeded = follow_latest_capture(
        "Glovebox/Not trained on 14-09",
        "Groundup/v2 2026-10-20",
        "Groundup/v3 2026-10-20",
        seeded=True,
    )
    assert stayed == "Glovebox/Not trained on 14-09"
    assert suggestion == "Groundup/v2 2026-10-20"

    followed, suggestion, seeded = follow_latest_capture(
        "Groundup/v2 2026-10-20",
        "Groundup/v2 2026-10-20",
        "Groundup/v3 2026-10-20",
        seeded=True,
    )
    assert (followed, suggestion, seeded) == ("Groundup/v3 2026-10-20", "Groundup/v3 2026-10-20", True)

    empty, suggestion, seeded = follow_latest_capture("", suggestion, None, seeded=True)
    assert empty == ""
    assert seeded is True


def test_input_folder_labels_suggest_image_folders_and_submaps(tmp_path: Path) -> None:
    from src.overall.io.tile_loader import input_folder_labels

    root = tmp_path / "Inputs"
    campaign = root / "Groundup" / "Groundup v7"
    (campaign / "v7").mkdir(parents=True)
    (campaign / "v7" / "wafer_shot_0_quad0_exp200000.png").write_bytes(b"png")
    (campaign / "v7" / "wafer_shot_4_all_exp200000.png").write_bytes(b"png")
    (campaign / "notes").mkdir()
    (campaign / "notes" / "readme.txt").write_text("skip", encoding="utf-8")
    classic = root / "R3 04-08"
    classic.mkdir(parents=True)
    (classic / "R3_1_1_5X.tif").write_bytes(b"tif")
    labels = input_folder_labels(root)
    assert labels == sorted(labels)
    assert "Groundup/Groundup v7" in labels
    assert "Groundup/Groundup v7/v7" in labels
    assert "R3 04-08" in labels
    assert "Groundup/Groundup v7/notes" not in labels


def test_pipeline_merges_one_submap_and_keeps_the_next() -> None:
    from src.main_pipeline.measurement.measurer import Particle
    from src.main_pipeline.runner import _combine_particle_tables

    def particle(set_id: str, role: str, x: float) -> Particle:
        shot = _submap_shot(set_id, role, x, 200.0, 1)
        return Particle(
            id=0,
            x_global=float(shot["x_global"]),
            y_global=float(shot["y_global"]),
            size=float(shot["size"]),
            confidence=float(shot["confidence"]),
            source_tile=str(shot["source_tile"]),
        )

    table = _combine_particle_tables(
        [],
        {
            "v7/wafer_shot_0_quad0_exp200000.png": [particle("v7", "quad0", 100.0)],
            "v7/wafer_shot_4_all_exp200000.png": [particle("v7", "all", 110.0)],
            "v8/wafer_shot_0_quad0_exp200000.png": [particle("v8", "quad0", 100.0)],
        },
        {
            "pixel_size_nm": 1.0,
            "measurement": {"merge_radius_px": 8.0, "nsew_merge_radius_px": 24.0, "nsew_size_match_fraction": 0.4},
        },
    )
    assert sorted(table["source_tile"].astype(str).map(lambda name: name.split("/", 1)[0])) == [
        "v7",
        "v8",
    ]
    assert len(table) == 2


def test_pipeline_gives_combi_and_uniform_one_id_drawn_on_both() -> None:
    from src.main_pipeline.measurement.measurer import Particle
    from src.main_pipeline.runner import _combine_particle_tables
    from src.verification.inspect import detections_on_review_picture
    from src.verification.queue import collapse_submap_review_detections

    def particle(name: str, x: float, confidence: float, particle_id: int) -> Particle:
        return Particle(
            id=particle_id,
            x_global=x,
            y_global=200.0,
            size=20_000.0,
            confidence=confidence,
            source_tile=name,
        )

    combi = "v6/combi_2of4.png"
    uniform = "v6/wafer_shot_4_all_exp200000.png"
    config = {
        "pixel_size_nm": 1.0,
        "measurement": {
            "merge_radius_px": 8.0,
            "nsew_merge_radius_px": 24.0,
            "nsew_size_match_fraction": 0.4,
        },
    }
    table = _combine_particle_tables(
        [],
        {
            combi: [
                particle(combi, 100.0, 0.9, 4),
                particle(combi, 100_000.0, 0.8, 5),
            ],
            uniform: [
                particle(uniform, 110.0, 0.4, 9),
                particle(uniform, 200_000.0, 0.7, 11),
            ],
        },
        config,
    )
    assert len(table) == 3
    shared = table.loc[table["x_global"].astype(float) < 1_000]
    assert len(shared) == 1
    assert set(str(shared.iloc[0]["nsew_dirs"]).split(",")) == {"ALL", "COMBI"}
    shared_id = int(shared.iloc[0]["id"])
    combi_only = table.loc[table["x_global"].astype(float).between(50_000, 150_000)]
    uniform_only = table.loc[table["x_global"].astype(float) > 150_000]
    assert set(str(combi_only.iloc[0]["nsew_dirs"]).split(",")) == {"COMBI"}
    assert set(str(uniform_only.iloc[0]["nsew_dirs"]).split(",")) == {"ALL"}

    def shown_ids(tile: str) -> set[int]:
        hits = detections_on_review_picture(table, tile)
        return set(hits["id"].astype(int))

    assert shown_ids(combi) == {shared_id, int(combi_only.iloc[0]["id"])}
    assert shown_ids(uniform) == {shared_id, int(uniform_only.iloc[0]["id"])}

    separate = pd.DataFrame(
        [
            {
                "id": 21,
                "source_tile": uniform,
                "x_global": 100.0,
                "y_global": 200.0,
                "size": 20_000.0,
                "confidence": 0.4,
                "nsew_dirs": "ALL",
            },
            {
                "id": 7,
                "source_tile": combi,
                "x_global": 110.0,
                "y_global": 200.0,
                "size": 22_000.0,
                "confidence": 0.9,
                "nsew_dirs": "COMBI",
            },
        ]
    )
    collapsed = collapse_submap_review_detections(separate, 24.0, 0.4)
    assert len(collapsed) == 1
    assert int(collapsed.iloc[0]["id"]) == 7
    assert set(str(collapsed.iloc[0]["nsew_dirs"]).split(",")) == {"ALL", "COMBI"}
    assert set(detections_on_review_picture(collapsed, combi)["id"].astype(int)) == {7}
    assert set(detections_on_review_picture(collapsed, uniform)["id"].astype(int)) == {7}
    already = collapsed.copy()
    again = collapse_submap_review_detections(already, 24.0, 0.4)
    assert set(str(again.iloc[0]["nsew_dirs"]).split(",")) == {"ALL", "COMBI"}
    assert set(detections_on_review_picture(again, combi)["id"].astype(int)) == {7}
    assert set(detections_on_review_picture(again, uniform)["id"].astype(int)) == {7}


def test_shared_hit_keeps_the_label_from_either_picture() -> None:
    from src.verification.inspect import detections_on_review_picture
    from src.verification.queue import collapse_submap_review_detections, label_map_with_aliases

    combi = "v6/combi_2of4.png"
    uniform = "v6/wafer_shot_4_all_exp200000.png"
    quad = "v6/wafer_shot_2_quad2_exp200000.png"
    labeled_key = "v6_LIGHT_500200_200100"
    detections = pd.DataFrame(
        [
            {
                "id": 4,
                "source_tile": combi,
                "x_global": 500_000.0,
                "y_global": 200_000.0,
                "size": 30_000.0,
                "confidence": 0.95,
                "nsew_dirs": "COMBI",
                "key": "v6_LIGHT_500000_200000",
            },
            {
                "id": 21,
                "source_tile": uniform,
                "x_global": 500_200.0,
                "y_global": 200_100.0,
                "size": 28_000.0,
                "confidence": 0.4,
                "nsew_dirs": "ALL",
                "key": labeled_key,
            },
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "key": labeled_key,
                "label": "particle",
                "source_tile": quad,
                "x_global": 500_200.0,
                "y_global": 200_100.0,
                "size": 28_000.0,
            }
        ]
    )
    collapsed = collapse_submap_review_detections(
        detections, 24_000.0, 0.4, labeled_keys=[labeled_key]
    )
    assert len(collapsed) == 1
    assert int(collapsed.iloc[0]["id"]) == 4
    assert collapsed.iloc[0]["key"] == labeled_key
    colors = label_map_with_aliases(labels)
    for tile in (combi, uniform):
        shown = detections_on_review_picture(collapsed, tile)
        assert list(shown["key"]) == [labeled_key]
        assert colors[labeled_key] == "particle"


def test_label_campaign_splits_glovebox_r_tiles_and_keeps_groundup_v1() -> None:
    assert label_campaign("R3_1_23_5X.tif") == "glovebox"
    assert label_campaign("n2_4_16.tif") == "glovebox"
    assert label_campaign("n2_v2_6_18.tif") == "glovebox"
    assert label_campaign("n3_1ms_1_1_bf.tif") == "glovebox"
    assert label_campaign("bf_-1.2ms.tif") == "glovebox"
    assert label_campaign("R0_1_1.tiff") == "groundup"
    assert label_campaign("v7/wafer_shot_0_quad0_exp200000.png") == "groundup"
    assert label_campaign("N.png") == "groundup"

    frame = pd.DataFrame(
        {
            "source_tile": ["R3_1_1_5X.tif", "wafer_shot_0.png", "n2_1_8.tif"],
            "label": ["particle", "not_particle", "particle"],
        }
    )
    glovebox = rows_for_campaign(frame, "glovebox")
    groundup = rows_for_campaign(frame, "groundup")
    assert glovebox["source_tile"].tolist() == ["R3_1_1_5X.tif", "n2_1_8.tif"]
    assert groundup["source_tile"].tolist() == ["wafer_shot_0.png"]
