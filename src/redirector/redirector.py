from __future__ import annotations

import math
from dataclasses import dataclass

PMMA_N = 1.49
FOV_SIZE_MM = 6.86
FOV_DEPTH_MM = 5.0
_EPS = 1e-5
_MAX_STEPS = 32


_PROFILE_STEPS = 64


@dataclass(frozen=True)
class RedirectorShape:
    

    outer_diameter_mm: float
    inner_diameter_mm: float
    drop_mm: float
    thickness_mm: float
    inner_angle_deg: float
    outer_angle_deg: float
    top: tuple[tuple[float, float], ...]
    bottom: tuple[tuple[float, float], ...]
    top_normals: tuple[tuple[float, float], ...]
    bottom_normals: tuple[tuple[float, float], ...]
    inner_normal: tuple[float, float]
    outer_normal: tuple[float, float]
    fov_depth_mm: float = FOV_DEPTH_MM

    @property
    def outer_radius_mm(self) -> float:
        return self.outer_diameter_mm / 2.0

    @property
    def inner_radius_mm(self) -> float:
        return self.inner_diameter_mm / 2.0

    def slope_deg(self, radius_mm: float) -> float:
        
        span = self.outer_radius_mm - self.inner_radius_mm
        if span <= 0:
            return self.inner_angle_deg
        u = (float(radius_mm) - self.inner_radius_mm) / span
        u = min(1.0, max(0.0, u))
        return math.degrees(math.atan(_profile_slope(self, u)))


def spherical_end_angles(
    outer_diameter_mm: float,
    inner_diameter_mm: float,
    drop_mm: float,
) -> tuple[float, float]:
    
    outer_r = float(outer_diameter_mm) / 2.0
    inner_r = float(inner_diameter_mm) / 2.0
    drop = float(drop_mm)
    if inner_r <= 0 or outer_r <= inner_r:
        raise ValueError("Outer diameter must be larger than the inner diameter.")
    if drop <= 0:
        raise ValueError("Height drop must be positive.")
    gap = outer_r * outer_r - inner_r * inner_r
    if drop * drop >= gap:
        raise ValueError(
            "Height drop is too large for a circular arc between these diameters."
        )
    radial = 0.5 * (drop + gap / drop)
    sphere_r = math.sqrt(radial * radial + inner_r * inner_r)
    return (
        math.degrees(math.asin(inner_r / sphere_r)),
        math.degrees(math.asin(outer_r / sphere_r)),
    )


def _profile_slope(shape: RedirectorShape, u: float) -> float:
    
    span = shape.outer_radius_mm - shape.inner_radius_mm
    slope_in = math.tan(math.radians(shape.inner_angle_deg))
    slope_out = math.tan(math.radians(shape.outer_angle_deg))
    m0 = slope_in * span
    m1 = slope_out * span
    uu = u * u
    dz_du = (
        (6.0 * uu - 6.0 * u) * 0.0
        + (3.0 * uu - 4.0 * u + 1.0) * m0
        + (-6.0 * uu + 6.0 * u) * shape.drop_mm
        + (3.0 * uu - 2.0 * u) * m1
    )
    return dz_du / span


def _profile_height(shape: RedirectorShape, u: float) -> float:
    span = shape.outer_radius_mm - shape.inner_radius_mm
    slope_in = math.tan(math.radians(shape.inner_angle_deg))
    slope_out = math.tan(math.radians(shape.outer_angle_deg))
    m0 = slope_in * span
    m1 = slope_out * span
    uu = u * u
    uuu = uu * u
    return (
        (uuu - 2.0 * uu + u) * m0
        + (-2.0 * uuu + 3.0 * uu) * shape.drop_mm
        + (uuu - uu) * m1
    )


def make_shape(
    outer_diameter_mm: float,
    inner_diameter_mm: float,
    drop_mm: float,
    thickness_mm: float,
    inner_angle_deg: float | None = None,
    outer_angle_deg: float | None = None,
    fov_depth_mm: float = FOV_DEPTH_MM,
) -> RedirectorShape:
    outer_r = float(outer_diameter_mm) / 2.0
    inner_r = float(inner_diameter_mm) / 2.0
    drop = float(drop_mm)
    thickness = float(thickness_mm)
    if inner_r <= 0 or outer_r <= inner_r:
        raise ValueError("Outer diameter must be larger than the inner diameter.")
    if drop <= 0:
        raise ValueError("Height drop must be positive.")
    if thickness <= 0:
        raise ValueError("Thickness must be positive.")
    if fov_depth_mm < 0:
        raise ValueError("Distance to the field of view cannot be negative.")
    if inner_angle_deg is None or outer_angle_deg is None:
        inner_angle, outer_angle = spherical_end_angles(
            outer_diameter_mm, inner_diameter_mm, drop_mm
        )
        inner_angle_deg = inner_angle if inner_angle_deg is None else inner_angle_deg
        outer_angle_deg = outer_angle if outer_angle_deg is None else outer_angle_deg
    inner_angle_deg = float(inner_angle_deg)
    outer_angle_deg = float(outer_angle_deg)
    if abs(inner_angle_deg) >= 89 or abs(outer_angle_deg) >= 89:
        raise ValueError("Edge angles must stay below 89°.")
    draft = RedirectorShape(
        outer_diameter_mm=float(outer_diameter_mm),
        inner_diameter_mm=float(inner_diameter_mm),
        drop_mm=drop,
        thickness_mm=thickness,
        inner_angle_deg=inner_angle_deg,
        outer_angle_deg=outer_angle_deg,
        top=(),
        bottom=(),
        top_normals=(),
        bottom_normals=(),
        inner_normal=(0.0, 0.0),
        outer_normal=(0.0, 0.0),
    )
    span = outer_r - inner_r
    top: list[tuple[float, float]] = []
    top_normals: list[tuple[float, float]] = []
    bottom: list[tuple[float, float]] = []
    bottom_normals: list[tuple[float, float]] = []
    for index in range(_PROFILE_STEPS + 1):
        u = index / _PROFILE_STEPS
        radius = inner_r + u * span
        height = _profile_height(draft, u)
        alpha = math.atan(_profile_slope(draft, u))
        normal = (-math.sin(alpha), math.cos(alpha))
        top.append((radius, height))
        top_normals.append(normal)
        bottom.append((radius - thickness * normal[0], height - thickness * normal[1]))
        bottom_normals.append((-normal[0], -normal[1]))
    inner_alpha = math.atan(_profile_slope(draft, 0.0))
    outer_alpha = math.atan(_profile_slope(draft, 1.0))
    return RedirectorShape(
        outer_diameter_mm=draft.outer_diameter_mm,
        inner_diameter_mm=draft.inner_diameter_mm,
        drop_mm=drop,
        thickness_mm=thickness,
        inner_angle_deg=inner_angle_deg,
        outer_angle_deg=outer_angle_deg,
        top=tuple(top),
        bottom=tuple(bottom),
        top_normals=tuple(top_normals),
        bottom_normals=tuple(bottom_normals),
        inner_normal=(-math.cos(inner_alpha), -math.sin(inner_alpha)),
        outer_normal=(math.cos(outer_alpha), math.sin(outer_alpha)),
        fov_depth_mm=float(fov_depth_mm),
    )


def _unit(vx: float, vy: float) -> tuple[float, float]:
    length = math.hypot(vx, vy)
    if length == 0:
        raise ValueError("Zero-length vector.")
    return vx / length, vy / length


def reflect(direction: tuple[float, float], normal: tuple[float, float]) -> tuple[float, float]:
    
    dot = direction[0] * normal[0] + direction[1] * normal[1]
    return _unit(
        direction[0] - 2.0 * dot * normal[0],
        direction[1] - 2.0 * dot * normal[1],
    )


def refract(
    direction: tuple[float, float],
    normal: tuple[float, float],
    n_from: float,
    n_to: float,
) -> tuple[float, float] | None:
    
    nx, ny = normal
    cos_i = -(direction[0] * nx + direction[1] * ny)
    if cos_i < 0:
        nx, ny = -nx, -ny
        cos_i = -cos_i
    cos_i = min(1.0, cos_i)
    sin_t2 = (n_from / n_to) ** 2 * (1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        return None
    cos_t = math.sqrt(1.0 - sin_t2)
    scale = n_from / n_to
    return _unit(
        scale * direction[0] + (scale * cos_i - cos_t) * nx,
        scale * direction[1] + (scale * cos_i - cos_t) * ny,
    )


def angle_down_deg(direction: tuple[float, float]) -> float:
    
    return math.degrees(math.atan2(-direction[1], abs(direction[0])))


def _incidence_cosine(direction: tuple[float, float], normal: tuple[float, float]) -> float:
    
    cos_i = -(direction[0] * normal[0] + direction[1] * normal[1])
    if cos_i < 0.0:
        cos_i = -cos_i
    return min(1.0, cos_i)


def fresnel_unpolarized(n_from: float, n_to: float, cos_i: float) -> tuple[float, float]:
    
    if n_from <= 0.0 or n_to <= 0.0:
        raise ValueError("Refractive index must be positive.")
    cos_i = min(1.0, max(0.0, float(cos_i)))
    sin_t2 = (n_from / n_to) ** 2 * (1.0 - cos_i * cos_i)
    if sin_t2 >= 1.0 or cos_i == 0.0:
        return 1.0, 0.0
    cos_t = math.sqrt(1.0 - sin_t2)
    denom_s = n_from * cos_i + n_to * cos_t
    denom_p = n_to * cos_i + n_from * cos_t
    rs = (n_from * cos_i - n_to * cos_t) / denom_s
    rp = (n_to * cos_i - n_from * cos_t) / denom_p
    reflectance = 0.5 * (rs * rs + rp * rp)
    scale = (n_to * cos_t) / (n_from * cos_i)
    ts = (2.0 * n_from * cos_i) / denom_s
    tp = (2.0 * n_from * cos_i) / denom_p
    transmission = 0.5 * scale * (ts * ts + tp * tp)
    reflectance = min(1.0, max(0.0, reflectance))
    transmission = min(1.0, max(0.0, transmission))
    return reflectance, transmission


def _interface_factor(
    direction: tuple[float, float],
    normal: tuple[float, float],
    n_from: float,
    n_to: float,
    *,
    transmitted: bool,
) -> float:
    
    reflectance, transmission = fresnel_unpolarized(
        n_from, n_to, _incidence_cosine(direction, normal)
    )
    return transmission if transmitted else reflectance


@dataclass(frozen=True)
class RayTrace:
    height_mm: float
    status: str
    angle_down_deg: float | None
    toward_center: bool | None
    exit_face: str | None
    exit_radius_mm: float | None
    reflections: int
    intensity: float
    path: str
    points: tuple[tuple[float, float], ...]


def _segment_time(
    origin: tuple[float, float],
    direction: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float | None:
    edge = (end[0] - start[0], end[1] - start[1])
    denom = direction[0] * edge[1] - direction[1] * edge[0]
    if abs(denom) < 1e-12:
        return None
    offset = (start[0] - origin[0], start[1] - origin[1])
    distance = (offset[0] * edge[1] - offset[1] * edge[0]) / denom
    along = (offset[0] * direction[1] - offset[1] * direction[0]) / denom
    if distance > _EPS and -1e-6 <= along <= 1.0 + 1e-6:
        return distance
    return None


def _mirrored(point: tuple[float, float], side: float) -> tuple[float, float]:
    return (side * point[0], point[1])


def _rim(shape: RedirectorShape, which: str, side: float) -> tuple[tuple[float, float], tuple[float, float]]:
    if which == "outer":
        start, end = shape.top[-1], shape.bottom[-1]
    else:
        start, end = shape.top[0], shape.bottom[0]
    return _mirrored(start, side), _mirrored(end, side)


def _ray_colour(in_fov: bool, downward: bool | None, selected: bool = False) -> str:
    if selected:
        return "#e83e8c"
    if downward is None or not in_fov:
        return "#9aa1ab"
    if downward:
        return "#c2410c"
    return "#1d4ed8"


def _candidates(
    shape: RedirectorShape,
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> list[tuple[float, str, tuple[float, float], tuple[float, float]]]:
    hits: list[tuple[float, str, tuple[float, float], tuple[float, float]]] = []
    chains = (
        (shape.top, shape.top_normals, "top"),
        (shape.bottom, shape.bottom_normals, "bottom"),
    )
    for side, label in ((1.0, "right"), (-1.0, "left")):
        for points, normals, name in chains:
            for index in range(len(points) - 1):
                start = _mirrored(points[index], side)
                end = _mirrored(points[index + 1], side)
                distance = _segment_time(origin, direction, start, end)
                if distance is None:
                    continue
                point = (
                    origin[0] + distance * direction[0],
                    origin[1] + distance * direction[1],
                )
                normal = (side * normals[index][0], normals[index][1])
                hits.append((distance, f"{name} {label}", point, normal))
        for which, edge_name, normal in (
            ("outer", "outer edge", shape.outer_normal),
            ("inner", "inner edge", shape.inner_normal),
        ):
            start, end = _rim(shape, which, side)
            distance = _segment_time(origin, direction, start, end)
            if distance is None:
                continue
            point = (
                origin[0] + distance * direction[0],
                origin[1] + distance * direction[1],
            )
            hits.append((distance, f"{edge_name} {label}", point, (side * normal[0], normal[1])))
    hits.sort(key=lambda item: item[0])
    return hits


def _extend(point: tuple[float, float], direction: tuple[float, float], length: float) -> tuple[float, float]:
    return (point[0] + length * direction[0], point[1] + length * direction[1])


def trace_ray(
    shape: RedirectorShape,
    height_mm: float,
    refractive_index: float = PMMA_N,
    reflect_far_side: bool = False,
) -> RayTrace:
    
    if refractive_index <= 1.0:
        raise ValueError("Refractive index must be greater than 1.")
    origin = _start_point(shape, height_mm)
    direction = (-1.0, 0.0)
    medium = 1.0
    points = [origin]
    reflections = 0
    intensity = 1.0
    faces: list[str] = []
    for _ in range(_MAX_STEPS):
        hits = _candidates(shape, origin, direction)
        if not hits:
            break
        _distance, face, point, air_normal = hits[0]
        points.append(point)
        if reflect_far_side and medium != 1.0 and face.startswith("outer edge"):
            faces.append(f"{face} bounce")
            inward = (-air_normal[0], -air_normal[1])
            intensity *= _interface_factor(
                direction, inward, refractive_index, 1.0, transmitted=False
            )
            direction = reflect(direction, inward)
            reflections += 1
            origin = point
            continue
        faces.append(face)
        if medium == 1.0:
            normal = air_normal
            leaving = refract(direction, normal, 1.0, refractive_index)
            intensity *= _interface_factor(
                direction, normal, 1.0, refractive_index, transmitted=leaving is not None
            )
            if leaving is None:
                direction = reflect(direction, normal)
                reflections += 1
            else:
                direction = leaving
                medium = refractive_index
        else:
            normal = (-air_normal[0], -air_normal[1])
            leaving = refract(direction, normal, refractive_index, 1.0)
            intensity *= _interface_factor(
                direction,
                normal,
                refractive_index,
                1.0,
                transmitted=leaving is not None,
            )
            if leaving is None:
                direction = reflect(direction, normal)
                reflections += 1
            else:
                direction = leaving
                medium = 1.0
        origin = point
    escaped = medium == 1.0 and len(faces) > 0 and not _candidates(shape, origin, direction)
    if not faces:
        return RayTrace(
            height_mm=float(height_mm),
            status="missed",
            angle_down_deg=None,
            toward_center=None,
            exit_face=None,
            exit_radius_mm=None,
            reflections=0,
            intensity=intensity,
            path="misses the part",
            points=tuple(points + [_extend(origin, direction, 20.0)]),
        )
    if not escaped:
        return RayTrace(
            height_mm=float(height_mm),
            status="trapped",
            angle_down_deg=None,
            toward_center=None,
            exit_face=faces[-1],
            exit_radius_mm=abs(points[-1][0]),
            reflections=reflections,
            intensity=intensity,
            path=" → ".join(faces),
            points=tuple(points),
        )
    exit_point = points[-1]
    return RayTrace(
        height_mm=float(height_mm),
        status="exited",
        angle_down_deg=angle_down_deg(direction),
        toward_center=exit_point[0] * direction[0] < 0,
        exit_face=faces[-1],
        exit_radius_mm=abs(exit_point[0]),
        reflections=reflections,
        intensity=intensity,
        path=" → ".join(faces),
        points=tuple(points + [_extend(exit_point, direction, 18.0)]),
    )


def _start_point(shape: RedirectorShape, height_mm: float) -> tuple[float, float]:
    outer_r = max(shape.top[-1][0], shape.bottom[-1][0])
    return (outer_r + 8.0, float(height_mm))


def enters_through_thickness(shape: RedirectorShape, height_mm: float) -> bool:
    
    origin = _start_point(shape, height_mm)
    hits = _candidates(shape, origin, (-1.0, 0.0))
    return bool(hits) and hits[0][1] == "outer edge right"


def thickness_span(shape: RedirectorShape) -> tuple[float, float]:
    
    top, bottom = _rim(shape, "outer", 1.0)
    low, high = sorted((top[1], bottom[1]))
    if high - low < 1e-9:
        return low, high
    probes = [low + (high - low) * (index + 0.5) / 60.0 for index in range(60)]
    open_face = [height for height in probes if enters_through_thickness(shape, height)]
    if len(open_face) >= 2:
        return open_face[0], open_face[-1]
    return low, high


def ray_heights(shape: RedirectorShape, count: int) -> list[float]:
    
    if count < 1:
        raise ValueError("Ray count must be at least 1.")
    low, high = thickness_span(shape)
    if high - low < 1e-9 or count == 1:
        return [0.5 * (low + high)]
    margin = 0.04 * (high - low)
    low += margin
    high -= margin
    step = (high - low) / (count - 1)
    return [low + index * step for index in range(count)]


def trace_fan(
    shape: RedirectorShape,
    count: int = 15,
    refractive_index: float = PMMA_N,
    reflect_far_side: bool = False,
) -> list[RayTrace]:
    return [
        trace_ray(shape, height, refractive_index, reflect_far_side)
        for height in ray_heights(shape, count)
    ]


def fov_plane_z_mm(shape: RedirectorShape) -> float:
    return shape.bottom[0][1] - shape.fov_depth_mm


def fov_crossing_mm(shape: RedirectorShape, trace: RayTrace) -> float | None:
    
    if trace.status != "exited" or len(trace.points) < 2:
        return None
    exit_point = trace.points[-2]
    far = trace.points[-1]
    dx = far[0] - exit_point[0]
    dy = far[1] - exit_point[1]
    length = math.hypot(dx, dy)
    if length == 0.0 or dy >= -1e-9:
        return None
    dx /= length
    dy /= length
    z_fov = fov_plane_z_mm(shape)
    if exit_point[1] <= z_fov:
        return None
    distance = (z_fov - exit_point[1]) / dy
    if distance < 0.0:
        return None
    return exit_point[0] + distance * dx


def hits_fov(shape: RedirectorShape, trace: RayTrace) -> bool:
    
    crossing = fov_crossing_mm(shape, trace)
    return crossing is not None and abs(crossing) <= FOV_SIZE_MM / 2.0


def fov_beam(
    shape: RedirectorShape,
    refractive_index: float = PMMA_N,
    reflect_far_side: bool = False,
    samples: int = 401,
) -> tuple[float, float | None, float | None]:
    
    low, high = thickness_span(shape)
    if high - low < 1e-9:
        trace = trace_ray(shape, 0.5 * (low + high), refractive_index, reflect_far_side)
        if hits_fov(shape, trace) and trace.angle_down_deg is not None:
            return 1.0, trace.angle_down_deg, trace.angle_down_deg
        return 0.0, None, None
    hits = 0
    angles: list[float] = []
    for index in range(samples):
        height = low + (high - low) * (index + 0.5) / samples
        trace = trace_ray(shape, height, refractive_index, reflect_far_side)
        if hits_fov(shape, trace) and trace.angle_down_deg is not None:
            hits += 1
            angles.append(trace.angle_down_deg)
    low_angle = min(angles) if angles else None
    high_angle = max(angles) if angles else None
    return hits / samples, low_angle, high_angle


def _surface_points(shape: RedirectorShape, which: str, side: float) -> list[tuple[float, float]]:
    points = shape.top if which == "top" else shape.bottom
    return [_mirrored(point, side) for point in points]


def section_image(
    shape: RedirectorShape,
    traces: list[RayTrace],
    selected: int | None = None,
) -> tuple[object, list[list[tuple[float, float]]]]:
    
    from PIL import Image, ImageDraw

    top_right = _surface_points(shape, "top", 1.0)
    bottom_right = _surface_points(shape, "bottom", 1.0)
    z_fov = fov_plane_z_mm(shape)
    half = FOV_SIZE_MM / 2.0
    drawn: list[list[tuple[float, float]]] = []
    for trace in traces:
        points = list(trace.points)
        crossing = fov_crossing_mm(shape, trace)
        if crossing is not None:
            points[-1] = (crossing, z_fov)
        drawn.append(points)
    part_x = [abs(point[0]) for point in top_right + bottom_right]
    part_y = [point[1] for point in top_right + bottom_right]
    half_w = max(part_x) * 1.18
    y_high = max(part_y)
    part_h = max(y_high - min(part_y), 0.05)
    width, height = 980, 520
    px_x = (0.84 * width) / (2.0 * half_w)
    target_part_px = min(0.42 * height, max(170.0, px_x * part_h * 4.0))
    px_z = target_part_px / part_h
    span_x = width / px_x
    span_y = max(height / px_z, (y_high - z_fov) / 0.8)
    min_x = -span_x / 2.0
    max_y = y_high + 0.14 * span_y
    min_y = max_y - span_y

    def xy(point: tuple[float, float]) -> tuple[float, float]:
        return (
            (point[0] - min_x) / span_x * width,
            (max_y - point[1]) / span_y * height,
        )

    def ring(side: float) -> list[tuple[float, float]]:
        top = _surface_points(shape, "top", side)
        bottom = _surface_points(shape, "bottom", side)
        return [xy(point) for point in top + list(reversed(bottom))]

    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    draw.line([xy((0.0, max_y)), xy((0.0, min_y))], fill="#9aa1ab", width=1)
    for side in (1.0, -1.0):
        draw.polygon(ring(side), fill="#d6e6f5", outline="#1f3a5f")
    polylines: list[list[tuple[float, float]]] = []
    order = [index for index in range(len(traces)) if index != selected]
    if selected is not None and 0 <= selected < len(traces):
        order.append(selected)
    for index in order:
        pixels = [xy(point) for point in drawn[index]]
        polylines.append(pixels)
        in_fov = hits_fov(shape, traces[index])
        downward = None if traces[index].angle_down_deg is None else traces[index].angle_down_deg >= 0.0
        colour = _ray_colour(in_fov, downward, selected=index == selected)
        width_px = 5 if index == selected else 2
        if len(pixels) >= 2:
            draw.line(pixels, fill=colour, width=width_px)
    polylines = [[xy(point) for point in points] for points in drawn]
    fov_a, fov_b = xy((-half, z_fov)), xy((half, z_fov))
    draw.line([fov_a, fov_b], fill="#15803d", width=5)
    draw.text((16, 8), "downward", fill="#c2410c")
    draw.text((110, 8), "upward", fill="#1d4ed8")
    draw.text((200, 8), "selected", fill="#e83e8c")
    draw.text((fov_a[0], min(height - 18, fov_b[1] + 6)), "FOV", fill="#166534")
    centre = xy((0.0, y_high))
    draw.text((centre[0] - 22, max(centre[1] - 16, 28)), "centre", fill="#3c4450")
    return image, polylines


def _nearest_ray(
    polylines: list[list[tuple[float, float]]],
    x: float,
    y: float,
    limit: float = 14.0,
) -> int | None:
    best: int | None = None
    best_distance = limit
    for index, pixels in enumerate(polylines):
        for start, end in zip(pixels, pixels[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length_sq = dx * dx + dy * dy
            if length_sq == 0.0:
                distance = math.hypot(x - start[0], y - start[1])
            else:
                along = max(0.0, min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / length_sq))
                distance = math.hypot(x - (start[0] + along * dx), y - (start[1] + along * dy))
            if distance < best_distance:
                best_distance = distance
                best = index
    return best


def _apply_redirector(
    outer_mm: float,
    inner_mm: float,
    drop_mm: float,
    thickness_mm: float,
    refractive_index: float,
    inner_angle_deg: float | None = None,
    outer_angle_deg: float | None = None,
    bounce: bool | None = None,
) -> None:
    import streamlit as st

    if inner_angle_deg is None or outer_angle_deg is None:
        inner_angle_deg, outer_angle_deg = spherical_end_angles(outer_mm, inner_mm, drop_mm)
    st.session_state["redir_outer_v3"] = outer_mm
    st.session_state["redir_inner_v3"] = inner_mm
    st.session_state["redir_drop_v3"] = drop_mm
    st.session_state["redir_thick_v3"] = thickness_mm
    st.session_state["redir_index_v3"] = refractive_index
    st.session_state["redir_inner_angle"] = round(float(inner_angle_deg), 1)
    st.session_state["redir_outer_angle"] = round(float(outer_angle_deg), 1)
    if bounce is not None:
        st.session_state["redir_bounce"] = bounce


def render_redirector_tab() -> None:
    
    import pandas as pd
    import streamlit as st
    from streamlit_image_coordinates import streamlit_image_coordinates
    
    with st.container(border=True, key="block-redirector-shape"):
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        outer = col_a.number_input(
            "Outer diameter (mm)", min_value=1.0, value=54.0, step=0.5, key="redir_outer_v3"
        )
        inner = col_b.number_input(
            "Inner diameter (mm)", min_value=1.0, value=10.0, step=0.5, key="redir_inner_v3"
        )
        drop = col_c.number_input(
            "Height drop (mm)", min_value=0.1, value=8.4, step=0.1, key="redir_drop_v3",
            help="How far the inner edge sits below the outer edge.",
        )
        thickness = col_d.number_input(
            "Thickness (mm)", min_value=0.1, value=2.0, step=0.1, key="redir_thick_v3",
            help="Material thickness, perpendicular to the curved surface.",
        )
        index = col_e.number_input(
            "Refractive index", min_value=1.01, value=PMMA_N, step=0.001, format="%.3f",
            key="redir_index_v3",
            help="Plexiglass (PMMA) is about 1.49. Schott N-LASF31A is 1.883.",
        )
        angle_row = st.columns(3)
        inner_angle = angle_row[0].number_input(
            "Inner angle (°)", min_value=-75.0, max_value=75.0, value=11.0, step=0.1,
            key="redir_inner_angle",
            help="Slope of the surface above horizontal at the inner diameter.",
        )
        outer_angle = angle_row[1].number_input(
            "Outer angle (°)", min_value=-75.0, max_value=75.0, value=23.0, step=0.1,
            key="redir_outer_angle",
            help="Slope of the surface above horizontal at the outer diameter.",
        )
        fov_depth = angle_row[2].number_input(
            "Distance to FOV (mm)", min_value=0.0, value=FOV_DEPTH_MM, step=0.1,
            key="redir_fov_depth",
            help="How far the 6.86 mm field of view sits below the inner bottom edge. "
            "The optimal values were found for 5 mm.",
        )
        button_row = st.columns([1, 1, 2])
        button_row[0].button(
            "Default values",
            on_click=_apply_redirector,
            args=(67.0, 20.0, 4.0, 3.0, PMMA_N),
            help="67 mm outer, 20 mm inner, 4 mm drop, 3 mm thick, plexiglass.",
        )
        button_row[1].button(
            "Optimal values",
            on_click=_apply_redirector,
            args=(54.0, 10.0, 8.4, 2.0, PMMA_N),
            kwargs={"inner_angle_deg": 11.0, "outer_angle_deg": 23.0, "bounce": False},
            help="54 mm outer, 10 mm inner, 8.4 mm drop, 2 mm thick, plexiglass. End angles 11° and 23°.",
        )
        reflect_far_side = button_row[2].checkbox(
            "Bounce off the outer diameter",
            key="redir_bounce",
            help="Rays pass through the plexiglass, including the opposite side. The outer face reflects them all back. The inner opening does not.",
        )
        count = st.slider("Number of rays", min_value=5, max_value=41, value=21, step=2, key="redir_rays_v2")

    with st.container(border=True, key="block-redirector-rays"):
        try:
            shape = make_shape(outer, inner, drop, thickness, inner_angle, outer_angle, fov_depth)
        except ValueError as exc:
            st.error(str(exc))
            return

        traces = [
            trace
            for trace in trace_fan(shape, int(count), float(index), reflect_far_side)
            if trace.path.startswith("outer edge right")
        ]
        fraction, angle_low, angle_high = fov_beam(shape, float(index), reflect_far_side)
        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric(
            "Edge angles",
            f"{shape.inner_angle_deg:.1f}° / {shape.outer_angle_deg:.1f}°",
            help="Surface angle above horizontal at the inner rim, then the outer rim.",
        )
        if angle_low is not None and angle_high is not None:
            metric_b.metric(
                "Output angle",
                f"{angle_low:.1f}° to {angle_high:.1f}°",
                help="Degrees below horizontal, only for the part of the beam that crosses the field of view.",
            )
        else:
            metric_b.metric("Output angle", "—")
        metric_c.metric(
            "Rays in FOV",
            f"{100.0 * fraction:.1f}%",
            help="Share of a uniform beam across the thickness that crosses the 6.86 mm square. Independent of how many rays are drawn.",
        )
        exited = [trace for trace in traces if trace.status == "exited"]
        if exited:
            low = min(trace.intensity for trace in exited)
            high = max(trace.intensity for trace in exited)
            intensity_text = (
                f"{100.0 * low:.1f}%"
                if abs(high - low) < 0.005
                else f"{100.0 * low:.1f}% to {100.0 * high:.1f}%"
            )
            metric_d.metric(
                "Exit intensity",
                intensity_text,
                help=(
                    "Share of the source still in each exiting ray. "
                    "The rest is lost to Fresnel reflection. "
                    "A total internal reflection keeps the whole ray; a partial reflection keeps only R."
                ),
            )
        else:
            metric_d.metric("Exit intensity", "—")

        selected = st.session_state.get("redir_selected_ray")
        if not isinstance(selected, int) or not 0 <= selected < len(traces):
            selected = None
        image, polylines = section_image(shape, traces, selected)
        click = streamlit_image_coordinates(
            image,
            key="redir_pick",
            cursor="pointer",
            use_column_width="always",
        )
        if isinstance(click, dict) and "x" in click:
            stamp = click.get("unix_time")
            if stamp != st.session_state.get("redir_click_stamp"):
                st.session_state["redir_click_stamp"] = stamp
                shown_w = float(click.get("width") or image.size[0])
                shown_h = float(click.get("height") or image.size[1])
                x = float(click["x"]) * image.size[0] / max(shown_w, 1.0)
                y = float(click["y"]) * image.size[1] / max(shown_h, 1.0)
                st.session_state["redir_selected_ray"] = _nearest_ray(polylines, x, y)
                st.rerun()
        st.caption("Click a ray to highlight it.")

        rows = []
        for trace in traces:
            in_fov = hits_fov(shape, trace)
            rows.append(
                {
                    "Ray height (mm)": round(trace.height_mm, 3),
                    "Output angle (° down)": None if not in_fov or trace.angle_down_deg is None else round(trace.angle_down_deg, 2),
                    "Hits FOV": "yes" if in_fov else "no",
                    "Exit face": trace.exit_face or "",
                    "Exit radius (mm)": None if trace.exit_radius_mm is None else round(trace.exit_radius_mm, 2),
                    "Reflections": trace.reflections,
                    "Exit intensity (%)": (
                        None if trace.status != "exited" else round(100.0 * trace.intensity, 1)
                    ),
                    "Intensity lost (%)": (
                        None
                        if trace.status != "exited"
                        else round(100.0 - round(100.0 * trace.intensity, 1), 1)
                    ),
                    "Path": trace.path,
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")