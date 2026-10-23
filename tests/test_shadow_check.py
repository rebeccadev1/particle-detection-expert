"""Shadow measurement on a synthetic bright-field particle."""

from __future__ import annotations

import numpy as np

from pathlib import Path

import pandas as pd

from src.shadow.shadow import (
    _parse_keys,
    detection_borders,
    direction_image,
    image_choice_labels,
    resolve_named_image,
    display_preview,
    find_label_image,
    light_distance_mm,
    mask_from_canvas,
    measure_shadow,
    paint_kept_mask,
    parse_light_from,
    run_shadow_check,
    scale_from_borders,
    shadow_direction,
    zoom_preview,
)


def _scene(shadow_below: bool) -> np.ndarray:
    rng = np.random.default_rng(0)
    image = 0.5 + rng.normal(0.0, 0.01, (200, 200))
    yy, xx = np.indices(image.shape)
    image[np.hypot(xx - 100, yy - 100) <= 10] = 0.9
    if shadow_below:
        image[111:141, 92:109] = 0.2
    else:
        image[59:89, 92:109] = 0.2
    return image


def test_shadow_length_and_area_light_from_north() -> None:
    result, _ = measure_shadow(_scene(True), 100.0, 100.0, 20.0, 0.0)
    assert result.found
    assert result.particle_type == "bright"
    assert abs(result.length_px - 30.0) <= 3.0
    assert abs(result.area_px - 30 * 17) <= 0.1 * 30 * 17


def test_shadow_on_light_side_is_ignored() -> None:
    result, _ = measure_shadow(_scene(False), 100.0, 100.0, 20.0, 0.0)
    assert not result.found


def _dark_hair_scene() -> np.ndarray:
    """Horizontal dark fiber with a shadow below (light from north)."""
    rng = np.random.default_rng(1)
    image = 0.55 + rng.normal(0.0, 0.008, (200, 200))
    image[95:105, 20:180] = 0.22
    image[105:135, 20:180] = 0.18
    return image


def test_drawn_mask_shadow_on_dark_hair() -> None:
    image = _dark_hair_scene()
    mask = np.zeros(image.shape, dtype=bool)
    mask[95:105, 20:180] = True
    result, debug = measure_shadow(
        image, 100.0, 100.0, 10.0, 0.0, particle_mask=mask
    )
    assert result.found
    assert result.particle_type == "dark"
    assert int(debug["particle"].any(axis=0).sum()) >= 80
    assert result.length_px >= 20.0
    assert not np.any(debug["shadow"] & debug["particle"])


def test_mask_from_canvas_fills_outline() -> None:
    canvas = np.zeros((100, 100, 4), dtype=np.uint8)
    for sl in ((slice(20, 80), slice(20, 25)), (slice(20, 80), slice(75, 80)),
               (slice(20, 25), slice(20, 80)), (slice(75, 80), slice(20, 80))):
        canvas[sl] = (40, 180, 70, 255)
    mask = mask_from_canvas(canvas, (200, 200), close_px=3)
    assert mask.shape == (200, 200)
    assert mask[100, 100]
    assert int(mask.sum()) > 2000


def test_mask_from_canvas_ignores_full_frame_background() -> None:
    canvas = np.zeros((80, 80, 4), dtype=np.uint8)
    canvas[:] = (90, 90, 90, 255)
    canvas[30:50, 20:60] = (40, 180, 70, 255)
    mask = mask_from_canvas(canvas, (80, 80), close_px=3)
    assert mask[40, 40]
    assert not mask[5, 5]
    assert float(mask.mean()) < 0.5


def test_mask_from_canvas_accepts_float_alpha() -> None:
    canvas = np.zeros((40, 40, 4), dtype=np.float32)
    canvas[10:30, 10:30] = (40 / 255, 180 / 255, 70 / 255, 1.0)
    mask = mask_from_canvas(canvas, (40, 40), close_px=3)
    assert mask[20, 20]
    assert int(mask.sum()) >= 200


def test_mask_from_canvas_respects_zoom_crop() -> None:
    canvas = np.zeros((50, 50, 4), dtype=np.uint8)
    canvas[:] = (40, 180, 70, 255)
    mask = mask_from_canvas(canvas, (200, 200), crop=(40, 60, 90, 110), close_px=3)
    assert mask[50, 70]
    assert not mask[10, 10]
    assert not mask[150, 150]
    assert int(mask.sum()) == 50 * 50


def test_zoom_preview_crops_and_enlarges() -> None:
    image = np.zeros((400, 400), dtype=np.float32)
    image[180:220, 180:220] = 1.0
    preview, crop = zoom_preview(image, 200.0, 200.0, zoom=4.0, max_side=200)
    y0, x0, y1, x1 = crop
    assert (y1 - y0) == 100 and (x1 - x0) == 100
    assert preview.shape[0] == 200 and preview.shape[1] == 200


def test_display_preview_downscales() -> None:
    rgb, scale = display_preview(np.zeros((1000, 1000), dtype=np.float32), max_side=200)
    assert scale == 0.2
    assert rgb.shape[0] == 200 and rgb.shape[1] == 200


def test_light_from_compass_and_stem() -> None:
    assert parse_light_from("e") == 90.0
    assert parse_light_from(None, "S.bmp") == 180.0
    dx, dy = shadow_direction(0.0)
    assert abs(dx) < 1e-9 and abs(dy - 1.0) < 1e-9


def test_light_distance_edge_centre_far_edge() -> None:
    assert abs(light_distance_mm(1023.5, 0.0, 2048, 2048, 0.0) - 6.6) < 0.01
    assert abs(light_distance_mm(1023.5, 1023.5, 2048, 2048, 0.0) - 10.0) < 1e-9
    assert abs(light_distance_mm(1023.5, 2047.0, 2048, 2048, 0.0) - 13.4) < 0.01
    assert abs(light_distance_mm(2047.0, 1023.5, 2048, 2048, 90.0) - 6.6) < 0.01


def test_find_label_image_uses_set_folder_and_run_name() -> None:
    path, _ties = find_label_image("v5/N.png", "tile_inspect")
    if path is not None:
        assert path.name == "N.png"
        assert path.parent.name.startswith("v5")
    path, _ties = find_label_image("E.bmp", "/x/Outputs/Groundup v2/particles.csv")
    if path is not None:
        assert path.parent.name == "Groundup v2"


def test_image_choice_labels_are_paths_under_inputs() -> None:
    labels = image_choice_labels()
    assert isinstance(labels, tuple)
    if labels:
        assert labels[0].startswith("Inputs/")
        assert labels == tuple(sorted(labels))


def test_resolve_named_image_uses_an_existing_path(tmp_path: Path) -> None:
    image = tmp_path / "field.png"
    image.write_bytes(b"")
    found, hits = resolve_named_image(f'  "{image}"  ')
    assert found == image
    assert hits == ()
    missing, hits = resolve_named_image("no-such-shadow-image-xyz.png")
    assert missing is None


def test_direction_image_switches_to_sibling(tmp_path: Path) -> None:
    north = tmp_path / "N.png"
    east = tmp_path / "E.png"
    north.write_bytes(b"")
    east.write_bytes(b"")
    assert direction_image(north, "E") == east
    assert direction_image(north, None) == north
    assert direction_image(north, "S") == north


def test_paint_kept_mask_tints_only_saved_pixels() -> None:
    preview = np.zeros((4, 6, 3), dtype=np.uint8)
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:4] = True
    painted = paint_kept_mask(preview, mask, (0, 0, 4, 6))
    assert painted[1, 2, 1] > painted[0, 0, 1]
    assert np.all(painted[0, 0] == 0)


def test_scale_from_borders_recovers_groundup_pixel_size() -> None:
    hits = [
        (20 * 3500.0, 80 * 3500.0, 20.0),
        (150 * 3500.0, 40 * 3500.0, 40.0),
    ]
    scale = scale_from_borders(hits, 200, 200)
    assert scale is not None
    assert abs(scale - 3500.0) < 1e-6


def test_crop_key_uses_detection_pixel_size_when_sidebar_scale_misses(tmp_path: Path) -> None:
    image = np.full((200, 200), 0.5, dtype=np.float32)
    cv2 = __import__("cv2")
    tile = tmp_path / "angles_v1_exp200000_3of6.png"
    cv2.imwrite(str(tile), (image * 255).astype(np.uint8))
    hits = [
        (20 * 3500.0, 80 * 3500.0, 20.0),
        (150 * 3500.0, 40 * 3500.0, 40.0),
    ]
    particles = tmp_path / "particles.csv"
    pd.DataFrame(
        {
            "x_global": [hit[0] for hit in hits],
            "y_global": [hit[1] for hit in hits],
            "tile_border_dist_px": [hit[2] for hit in hits],
            "source_tile": [tile.name, tile.name],
        }
    ).to_csv(particles, index=False)
    x_nm, y_nm = hits[0][0], hits[0][1]
    labels = pd.DataFrame(
        [{
            "key": "angles_v1_exp200000_3of6_70000_280000",
            "particle_id": 1,
            "source_tile": tile.name,
            "x_global": x_nm,
            "y_global": y_nm,
            "size": 20 * 3500.0,
            "source_csv": str(particles),
        }]
    )
    result = run_shadow_check(
        labels,
        {"pixel_size_nm": 960.0},
        keys=["angles_v1_exp200000_3of6_70000_280000"],
        input_dir=str(tmp_path),
        light_from="N",
        out=None,
    )
    assert result.messages == []
    assert abs(float(result.table.iloc[0]["x_px"]) - 20.0) < 0.2
    assert abs(float(result.table.iloc[0]["y_px"]) - 80.0) < 0.2
    assert len(detection_borders(particles, tile.name, {})) == 2


def test_parse_keys_splits_on_commas_and_spaces() -> None:
    keys = _parse_keys("R3_1_17_5X_42943680_1240320, v1_NSEW_1_2 R3_4_12_5X_1386512_1834272")
    assert keys == [
        "R3_1_17_5X_42943680_1240320",
        "v1_NSEW_1_2",
        "R3_4_12_5X_1386512_1834272",
    ]
