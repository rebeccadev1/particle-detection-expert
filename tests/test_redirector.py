from __future__ import annotations

import math

import pytest

from src.redirector.redirector import fresnel_unpolarized, make_shape, refract, trace_ray


def test_arc_drop_and_thickness() -> None:
    shape = make_shape(67.0, 20.0, 4.0, 3.0)
    assert abs((shape.top[-1][1] - shape.top[0][1]) - 4.0) < 1e-6
    span = math.hypot(
        shape.bottom[len(shape.top) // 2][0] - shape.top[len(shape.top) // 2][0],
        shape.bottom[len(shape.top) // 2][1] - shape.top[len(shape.top) // 2][1],
    )
    assert span == pytest.approx(3.0, abs=0.05)
    assert shape.slope_deg(33.5) > shape.slope_deg(10.0)


def test_end_angles_are_inputs() -> None:
    shape = make_shape(67.0, 20.0, 4.0, 3.0, inner_angle_deg=8.0, outer_angle_deg=20.0)
    assert shape.slope_deg(10.0) == pytest.approx(8.0, abs=0.05)
    assert shape.slope_deg(33.5) == pytest.approx(20.0, abs=0.05)


def test_snell_thirty_degrees() -> None:
    incidence = math.radians(30.0)
    direction = (math.sin(incidence), -math.cos(incidence))
    transmitted = refract(direction, (0.0, 1.0), 1.0, 1.49)
    assert transmitted is not None
    angle = math.acos(max(-1.0, min(1.0, -transmitted[1])))
    expected = math.asin(math.sin(incidence) / 1.49)
    assert abs(angle - expected) < 1e-9


def test_total_internal_reflection() -> None:
    incidence = math.radians(60.0)
    direction = (math.sin(incidence), -math.cos(incidence))
    assert refract(direction, (0.0, 1.0), 1.49, 1.0) is None


def test_fresnel_normal_incidence_averages_to_the_power_split() -> None:
    index = 1.49
    reflectance, transmission = fresnel_unpolarized(1.0, index, 1.0)
    expected = ((index - 1.0) / (index + 1.0)) ** 2
    assert reflectance == pytest.approx(expected)
    assert transmission == pytest.approx(1.0 - expected)
    assert reflectance + transmission == pytest.approx(1.0)


def test_fresnel_unpolarized_is_the_mean_of_s_and_p() -> None:
    n_from, n_to = 1.0, 1.49
    cos_i = math.cos(math.radians(45.0))
    sin_t2 = (n_from / n_to) ** 2 * (1.0 - cos_i * cos_i)
    cos_t = math.sqrt(1.0 - sin_t2)
    rs = (n_from * cos_i - n_to * cos_t) / (n_from * cos_i + n_to * cos_t)
    rp = (n_to * cos_i - n_from * cos_t) / (n_to * cos_i + n_from * cos_t)
    scale = (n_to * cos_t) / (n_from * cos_i)
    ts = (2.0 * n_from * cos_i) / (n_from * cos_i + n_to * cos_t)
    tp = (2.0 * n_from * cos_i) / (n_to * cos_i + n_from * cos_t)
    reflectance, transmission = fresnel_unpolarized(n_from, n_to, cos_i)
    assert reflectance == pytest.approx(0.5 * (rs * rs + rp * rp))
    assert transmission == pytest.approx(0.5 * scale * (ts * ts + tp * tp))
    assert reflectance + transmission == pytest.approx(1.0)


def test_total_internal_reflection_keeps_all_of_the_intensity() -> None:
    cos_i = math.cos(math.radians(60.0))
    reflectance, transmission = fresnel_unpolarized(1.49, 1.0, cos_i)
    assert reflectance == pytest.approx(1.0)
    assert transmission == pytest.approx(0.0)


def test_exit_intensity_counts_each_refraction() -> None:
    shape = make_shape(67.0, 20.0, 4.0, 3.0)
    trace = trace_ray(shape, 2.5)
    assert trace.status == "exited"
    assert trace.reflections == 1
    single = fresnel_unpolarized(1.0, 1.49, 1.0)[1]
    assert trace.intensity == pytest.approx(single * single, abs=0.01)
    assert 0.0 < trace.intensity < single


def test_default_ray_exits_downward() -> None:
    shape = make_shape(67.0, 20.0, 4.0, 3.0)
    trace = trace_ray(shape, 2.5)
    assert trace.status == "exited"
    assert trace.angle_down_deg is not None
    assert trace.toward_center


def test_fov_distance_moves_the_fov_plane() -> None:
    from src.redirector.redirector import fov_plane_z_mm

    near = make_shape(54.0, 10.0, 8.4, 2.0, 11.0, 23.0)
    far = make_shape(54.0, 10.0, 8.4, 2.0, 11.0, 23.0, fov_depth_mm=12.0)
    assert fov_plane_z_mm(near) - fov_plane_z_mm(far) == pytest.approx(7.0)
    with pytest.raises(ValueError):
        make_shape(54.0, 10.0, 8.4, 2.0, fov_depth_mm=-1.0)


def test_rejects_a_drop_too_steep_for_a_circular_arc() -> None:
    with pytest.raises(ValueError):
        make_shape(67.0, 20.0, 40.0, 3.0)


def test_fan_enters_only_through_the_thickness() -> None:
    from src.redirector.redirector import trace_fan, trace_ray

    shape = make_shape(24.0, 10.0, 0.4, 1.0)
    traces = trace_fan(shape, 11, 1.883)
    assert traces
    assert all(trace.path.startswith("outer edge right") for trace in traces)

    plain = trace_ray(shape, -0.279, 1.883)
    bounced = trace_ray(shape, -0.279, 1.883, reflect_far_side=True)
    assert "inner edge left" in bounced.path
    assert "outer edge left bounce" in bounced.path
    assert "inner edge left bounce" not in bounced.path
    assert bounced.reflections >= plain.reflections
    assert bounced.intensity < plain.intensity
