"""Streamlit controls for the ring light and Basler capture script."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import streamlit as st

from src.capture.focus import autofocus_status, start_autofocus, stop_autofocus
from src.capture.preview import preview_snapshot, start_preview, stop_preview
from src.capture.scan_capture import scan_capture_status, start_scan_capture, stop_scan_capture
from src.capture.stage import (
    DEFAULT_COLUMNS,
    DEFAULT_FEED,
    DEFAULT_ROWS,
    DEFAULT_STEP_MM,
    build_scan_gcode,
    connect_printer,
    disconnect_printer,
    move_axis,
    pause_plan,
    printer_status,
    scan_summary,
    start_gcode,
    stop_scan,
)

_CAPTURE_PATH = Path(__file__).resolve().parent / "camera_automation_v3.py"

def _load_arduino_link() -> dict:
    """Keep the open Arduino across a script reload so the port is not opened twice."""
    shared = getattr(builtins, "_asml_arduino_link", None)
    if isinstance(shared, dict) and "ser" in shared:
        return shared
    shared = {"ser": None, "port": None, "ack": ""}
    builtins._asml_arduino_link = shared
    return shared


_LINK = _load_arduino_link()

CAPTURE_WIDGET_DEFAULTS: dict = {
    "capture_mode": "Uniform",
    "capture_nsew_style": "Automatic",
    "capture_total_ms": 100.0,
}

STAGE_WIDGET_DEFAULTS: dict = {
    "stage_step_mm": DEFAULT_STEP_MM,
    "stage_columns": DEFAULT_COLUMNS,
    "stage_rows": DEFAULT_ROWS,
    "stage_feed": DEFAULT_FEED,
}

FOCUS_WIDGET_DEFAULTS: dict = {
    "focus_max_travel_mm": 1.0,
    "focus_coarse_mm": 0.05,
    "focus_fine_mm": 0.01,
    "focus_feed": 300,
}


def keep_capture_inputs() -> None:
    """Call on every run. Streamlit drops widget values while this tab is hidden."""
    for key in CAPTURE_WIDGET_DEFAULTS:
        held = f"_held_{key}"
        if key in st.session_state:
            st.session_state[held] = st.session_state[key]
        elif held in st.session_state:
            st.session_state[key] = st.session_state[held]


def keep_stage_inputs() -> None:
    """Call on every run so stage factors survive leaving the Capture tab."""
    for key in (*STAGE_WIDGET_DEFAULTS, *FOCUS_WIDGET_DEFAULTS):
        held = f"_held_{key}"
        if key in st.session_state:
            st.session_state[held] = st.session_state[key]
        elif held in st.session_state:
            st.session_state[key] = st.session_state[held]


def _capture():
    name = "camera_automation_v3"
    spec = importlib.util.spec_from_file_location(name, _CAPTURE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_CAPTURE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_module():
    global _CAP
    if _CAP is None:
        _CAP = _capture()
    return _CAP


_CAP = None


def _serial():
    ser = _LINK.get("ser")
    if ser is None:
        return None
    if not getattr(ser, "is_open", False):
        _LINK["ser"] = None
        _LINK["port"] = None
        return None
    return ser


def _forget_port() -> None:
    ser = _LINK.get("ser")
    if ser is not None:
        try:
            if getattr(ser, "is_open", False):
                ser.close()
        except Exception:
            pass
    _LINK["ser"] = None
    _LINK["port"] = None


def _run_serial(action):
    """Run a command on the open port. A dead port is dropped so Connect works again."""
    ser = _serial()
    if ser is None:
        st.error("Connect the Arduino first.")
        return None
    try:
        reply = action(ser)
    except Exception as exc:
        _forget_port()
        st.error(f"{exc} The serial port was closed.")
        return None
    _LINK["ack"] = reply
    return reply


def _selected_mode() -> str:
    mode = st.session_state.get("capture_mode", "Uniform")
    return mode if mode in ("Uniform", "NSEW") else "Uniform"


def _apply_mode_light(cap, mode: str):
    if mode == "Uniform":
        return _run_serial(cap.light_all)
    return _run_serial(cap.lights_off)


def render_capture_tab() -> None:
    """Uniform lighting, or one side at a time. One total-exposure control."""
    cap = _capture_module()

    st.caption(
        "Capture and scan use the light already on. Connect the Arduino only to change the ring. "
        "NSEW still needs it, because each frame switches sides. "
        "Close the Arduino Serial Monitor before connecting."
    )

    mode = st.radio(
        "Lighting",
        ("Uniform", "NSEW"),
        horizontal=True,
        key="capture_mode",
    )
    nsew_style = "Automatic"
    if mode == "NSEW":
        nsew_style = st.radio(
            "NSEW capture",
            ("Automatic", "Manual"),
            horizontal=True,
            key="capture_nsew_style",
        )

    try:
        ports = cap.list_serial_ports()
    except ImportError:
        st.error("pyserial is not installed. From the project folder: pip install pyserial")
        return

    devices = [device for device, _label in ports]
    labels = {device: label for device, label in ports}
    ser = _serial()

    if devices:
        if st.session_state.get("arduino_port") not in devices:
            st.session_state.pop("arduino_port", None)
        st.selectbox(
            "Arduino port",
            devices,
            index=None,
            placeholder="Select a port",
            key="arduino_port",
            format_func=lambda device: labels.get(device, device),
            disabled=ser is not None,
        )
    else:
        st.info("No serial ports found. Plug in the Uno and reopen this tab.")

    connect_col, disconnect_col, status_col = st.columns([1, 1, 2])
    arduino_port = st.session_state.get("arduino_port")
    if connect_col.button(
        "Connect to Arduino",
        type="primary",
        disabled=not arduino_port or ser is not None,
    ):
        port = arduino_port
        printer = printer_status()
        if printer["connected"] and port == printer["port"]:
            st.error("That port is already the printer. Choose the other port for the Arduino.")
        else:
            opened = None
            try:
                opened = cap.open_arduino(port)
                if _selected_mode() == "Uniform":
                    cap.light_all(opened)
                    ack = "OK ALL"
                else:
                    cap.lights_off(opened)
                    ack = "OK OFF"
            except Exception as exc:
                if opened is not None:
                    try:
                        opened.close()
                    except Exception:
                        pass
                st.error(str(exc))
            else:
                _LINK["ser"] = opened
                _LINK["port"] = port
                _LINK["ack"] = ack
                st.session_state["capture_lit_mode"] = _selected_mode()
                st.rerun()

    if disconnect_col.button("Disconnect Camera", disabled=ser is None):
        live = _serial()
        if live is not None:
            try:
                cap.lights_off(live)
            except Exception as exc:
                st.warning(f"LEDs may still be on: {exc}")
        _forget_port()
        _LINK["ack"] = ""
        st.session_state.pop("capture_lit_mode", None)
        st.rerun()

    if ser is not None:
        status_col.success(f"Connected to {_LINK['port']}")
        if _LINK["ack"]:
            status_col.caption(f"Last reply: {_LINK['ack']}")
    else:
        status_col.info("Not connected")

    if ser is not None and st.session_state.get("capture_lit_mode") != mode:
        if _apply_mode_light(cap, mode):
            st.session_state["capture_lit_mode"] = mode

    total_ms = st.number_input(
        "Total exposure (ms)",
        min_value=4.0,
        max_value=4000.0,
        value=float(CAPTURE_WIDGET_DEFAULTS["capture_total_ms"]),
        step=1.0,
        key="capture_total_ms",
        help="Dose from a single side. The camera time is calculated from this and the lighting mode.",
    )
    sides_on = cap.RING_SIDES if mode == "Uniform" else 1
    shutter = cap.shutter_ms(float(total_ms), sides_on)
    _render_live(cap, shutter)

    focusing = autofocus_status()["running"] or scan_capture_status()["running"]
    if mode == "Uniform":
        on_col, off_col = st.columns(2)
        if on_col.button("All sides on", disabled=ser is None, width="stretch"):
            _run_serial(cap.light_all)
        if off_col.button("All off", disabled=ser is None, width="stretch"):
            _run_serial(cap.lights_off)
        if st.button("Capture", type="primary", disabled=focusing):
            _capture_run(cap, mode, shutter)
    elif nsew_style == "Automatic":
        st.caption(
            "One click lights N, then E, then S, then W, and saves each frame. "
            "It waits 0.5 seconds between sides."
        )
        if st.button("Capture N, E, S, W", type="primary", disabled=ser is None or focusing):
            _capture_run(cap, mode, shutter)
    else:
        st.caption("Each button lights that side and takes its picture.")
        side_cols = st.columns(len(cap.DIRECTIONS))
        for index, direction in enumerate(cap.DIRECTIONS):
            if side_cols[index].button(
                direction,
                key=f"capture_side_{direction}",
                disabled=ser is None or focusing,
                width="stretch",
            ):
                _capture_one_side(cap, index, shutter)
        if st.button("New set", disabled=ser is None):
            st.session_state.pop("capture_manual_folder", None)
            st.session_state.pop("capture_manual_row", None)
            st.session_state.pop("capture_manual_col", None)
            st.rerun()

    st.divider()
    _render_stage(cap)
    st.divider()
    _show_last_run()


def _stage_columns() -> int:
    return int(st.session_state.get("stage_columns", STAGE_WIDGET_DEFAULTS["stage_columns"]))


def _manual_target(cap) -> tuple[Path, int, int]:
    current = st.session_state.get("capture_manual_folder")
    row = st.session_state.get("capture_manual_row")
    col = st.session_state.get("capture_manual_col")
    if current and Path(current).is_dir() and row and col:
        return Path(current), int(row), int(col)
    run = cap.allocate_run_folder(cap.CAPTURES_DIR)
    st.session_state["capture_manual_folder"] = str(run)
    st.session_state["capture_manual_row"] = 1
    st.session_state["capture_manual_col"] = 1
    return run, 1, 1


def _capture_one_side(cap, index: int, shutter_ms: float) -> None:
    ser = _serial()
    if ser is None:
        st.error("Connect the Arduino first.")
        return
    exposure_us = shutter_ms * 1000.0
    camera = None
    try:
        direction = cap.DIRECTIONS[index]
        with st.spinner(f"Capturing {direction}…"):
            stop_preview()
            camera, converter = cap.open_camera(exposure_us)
            folder, row, col = _manual_target(cap)
            filename = cap.capture_side(
                camera,
                converter,
                ser,
                folder,
                index,
                cap.grab_timeout_ms(exposure_us),
                row,
                col,
            )
            cap.write_manifest(
                folder,
                {
                    "mode": "nsew",
                    "row": row,
                    "col": col,
                    "total_exposure_ms": float(st.session_state["capture_total_ms"]),
                    "shutter_ms": shutter_ms,
                    "exposure_us": exposure_us,
                    "sides_on": 1,
                    "port": _LINK.get("port"),
                    "files": [filename],
                },
            )
    except Exception as exc:
        st.error(str(exc))
        return
    finally:
        cap.shutdown(camera, None)
        start_preview(cap, exposure_us)
    st.session_state["capture_last_run"] = {
        "folder": str(folder),
        "files": [cap.image_filename(row, col, name) for name in cap.DIRECTIONS],
        "mode": "NSEW manual",
    }
    _LINK["ack"] = f"OK {index}"
    st.success(f"Saved {filename} to {folder}")


def _capture_run(cap, mode: str, shutter_ms: float) -> None:
    ser = _serial()
    if mode != "Uniform" and ser is None:
        st.error("NSEW capture needs the Arduino to switch sides.")
        return
    exposure_us = shutter_ms * 1000.0
    timeout_ms = cap.grab_timeout_ms(exposure_us)
    camera = None
    folder = None
    saved: list[str] = []
    failed = False
    try:
        with st.spinner("Capturing…" if mode == "Uniform" else "Capturing N, E, S, W…"):
            stop_preview()
            camera, converter = cap.open_camera(exposure_us)
            folder = cap.allocate_run_folder(cap.CAPTURES_DIR)
            row, col = 1, 1
            if mode == "Uniform":
                saved = [cap.capture_uniform(camera, converter, ser, folder, timeout_ms, row, col)]
            else:
                saved = cap.capture_nsew(camera, converter, ser, folder, timeout_ms, row, col)
            cap.write_manifest(
                folder,
                {
                    "mode": "uniform" if mode == "Uniform" else "nsew",
                    "row": row,
                    "col": col,
                    "total_exposure_ms": float(st.session_state["capture_total_ms"]),
                    "shutter_ms": shutter_ms,
                    "exposure_us": exposure_us,
                    "sides_on": cap.RING_SIDES if mode == "Uniform" else 1,
                    "port": _LINK.get("port"),
                    "files": saved,
                },
            )
    except Exception as exc:
        failed = True
        st.error(str(exc))
    finally:
        cap.shutdown(camera, None)
        start_preview(cap, exposure_us)
    if folder is not None and saved:
        st.session_state["capture_last_run"] = {
            "folder": str(folder),
            "files": saved,
            "mode": mode,
        }
        _LINK["ack"] = "OK ALL" if mode == "Uniform" else f"OK {len(saved) - 1}"
    if not failed and saved:
        st.success(f"Saved {', '.join(saved)} to {folder}")


def _show_last_run() -> None:
    last = st.session_state.get("capture_last_run")
    if not last:
        return
    folder = Path(last["folder"])
    files = last.get("files") or ([last["file"]] if last.get("file") else [])
    st.subheader("Last capture")
    label = last.get("mode") or "Capture"
    st.write(f"{label} — `{folder}`")
    cols = st.columns(max(len(files), 1))
    for column, name in zip(cols, files):
        path = folder / name
        if path.is_file():
            column.image(str(path), caption=path.stem, width="stretch")
        else:
            column.warning(f"Missing {name}")


def suspend_live_view() -> None:
    """Close the camera when the Capture tab is not on screen."""
    stop_preview()


@st.fragment(run_every=0.25)
def _live_feed() -> None:
    snap = preview_snapshot()
    frame = snap["frame"]
    if frame is None:
        if snap["error"]:
            st.error(snap["error"])
        else:
            st.info("Opening the camera…")
        return
    st.image(frame, width=480)
    if snap["error"]:
        st.caption(snap["error"])


def _render_live(cap, shutter_ms: float) -> None:
    st.subheader("Live view")
    if scan_capture_status()["running"]:
        st.caption("The scan has the camera. The live view returns when it finishes.")
        return
    st.caption(f"Same shutter as the capture, {shutter_ms:.1f} ms. The feed pauses while a picture is saved.")
    start_preview(cap, shutter_ms * 1000.0)
    _live_feed()


def _stage_pause() -> dict:
    mode = st.session_state.get("capture_mode", "Uniform")
    total_ms = float(st.session_state.get("capture_total_ms", 100.0))
    return pause_plan(total_ms, mode)


def _stage_program() -> str:
    return build_scan_gcode(
        step_mm=float(st.session_state["stage_step_mm"]),
        columns=int(st.session_state["stage_columns"]),
        rows=int(st.session_state["stage_rows"]),
        feed=float(st.session_state["stage_feed"]),
        dwell_ms=_stage_pause()["dwell_ms"],
    )


def _focus_limits() -> str | None:
    travel = float(st.session_state["focus_max_travel_mm"])
    coarse = float(st.session_state["focus_coarse_mm"])
    fine = float(st.session_state["focus_fine_mm"])
    if coarse > travel or fine > travel:
        return "Coarse and fine steps have to fit inside the max travel."
    return None


def _render_focus(locked: bool) -> None:
    """Hill-climb Z from the current height using the live image."""
    st.subheader("Focus")
    st.caption(
        "Steps printer Z from the current height and keeps the direction that makes "
        "the live image sharper. A soft reading is scored again, then one more step "
        "is taken before turning around. The first step is +Z."
    )
    travel_col, coarse_col, fine_col, feed_col = st.columns(4)
    travel_col.number_input(
        "Max travel (mm)",
        min_value=0.01,
        max_value=20.0,
        value=float(FOCUS_WIDGET_DEFAULTS["focus_max_travel_mm"]),
        step=0.01,
        format="%.3f",
        key="focus_max_travel_mm",
        disabled=locked,
        help="How far Z may move up or down from the height where autofocus starts.",
    )
    coarse_col.number_input(
        "Coarse step (mm)",
        min_value=0.001,
        max_value=5.0,
        value=float(FOCUS_WIDGET_DEFAULTS["focus_coarse_mm"]),
        step=0.001,
        format="%.3f",
        key="focus_coarse_mm",
        disabled=locked,
        help="First pass. A soft frame is scored again, and one more step is taken before reversing.",
    )
    fine_col.number_input(
        "Fine step (mm)",
        min_value=0.001,
        max_value=5.0,
        value=float(FOCUS_WIDGET_DEFAULTS["focus_fine_mm"]),
        step=0.001,
        format="%.3f",
        key="focus_fine_mm",
        disabled=locked,
        help="Second pass around the sharpest height from the coarse pass.",
    )
    feed_col.number_input(
        "Z speed (mm/min)",
        min_value=1,
        max_value=6000,
        value=int(FOCUS_WIDGET_DEFAULTS["focus_feed"]),
        step=50,
        key="focus_feed",
        disabled=locked,
    )
    limit_error = _focus_limits()
    if limit_error:
        st.error(limit_error)
    _focus_progress()


def _format_score(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _format_focus_step(index: int, step: dict) -> str:
    place = f"{step['offset_mm']:+.3f} mm"
    score = _format_score(float(step["sharpness"]))
    decision = step["decision"]
    if decision == "start":
        return f"{index:>2}. Start      {place}   {score}"
    if decision == "back":
        return f"{index:>2}. Back to    {place}   {score}"
    phase = {"coarse": "Coarse", "fine": "Fine"}.get(step["phase"], str(step["phase"]))
    verdict = {"sharper": "sharper", "again": "again"}.get(decision, "not sharper")
    return f"{index:>2}. {phase:<10} {place}   {score}   {verdict}"


def _focus_steps(info: dict) -> None:
    """Expandable log of each height autofocus has scored."""
    steps = info["steps"]
    run_id = info["run_id"]
    if run_id and st.session_state.get("_focus_steps_seen") != run_id:
        st.session_state["focus_steps_open"] = True
        st.session_state["_focus_steps_seen"] = run_id
    with st.expander("Focus steps", expanded=False, key="focus_steps_open", on_change="rerun"):
        if not steps:
            st.caption("Each height that autofocus tries shows up here.")
            return
        st.code("\n".join(_format_focus_step(index, step) for index, step in enumerate(steps, start=1)))


@st.fragment(run_every=0.5)
def _focus_progress() -> None:
    info = autofocus_status()
    printer = printer_status()
    snap = preview_snapshot()
    if info["running"]:
        st.session_state["_focus_was_running"] = True
        place = f"{info['offset_mm']:+.3f} mm from the start"
        if info["stopping"]:
            st.caption(f"Stopping. Camera is at {place}.")
        else:
            st.caption(
                f"Z {place}. Sharpness {_format_score(info['sharpness'])}. "
                f"Best {info['best_offset_mm']:+.3f} mm "
                f"({_format_score(info['best_sharpness'])})."
            )
    elif st.session_state.pop("_focus_was_running", False):
        st.rerun(scope="app")
    elif info["phase"] == "error" and info["error"]:
        st.error(info["error"])
    elif info["phase"] == "done":
        height = f"{info['best_offset_mm']:+.3f} mm from the start"
        if info["stopped"]:
            st.warning(f"Stopped. Camera is at the sharpest height tried, {height}.")
        else:
            st.success(
                f"Focused at {height}. Sharpness {_format_score(info['best_sharpness'])}."
            )
    _focus_steps(info)
    if info["running"]:
        if st.button("Stop focus", type="primary"):
            stop_autofocus()
        return
    ready = (
        printer["online"]
        and not printer["printing"]
        and snap["frame"] is not None
        and _focus_limits() is None
    )
    if st.button("Autofocus", type="primary", disabled=not ready):
        try:
            start_autofocus(
                max_travel_mm=float(st.session_state["focus_max_travel_mm"]),
                coarse_mm=float(st.session_state["focus_coarse_mm"]),
                fine_mm=float(st.session_state["focus_fine_mm"]),
                feed=float(st.session_state["focus_feed"]),
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.rerun(scope="app")


@st.fragment(run_every=1)
def _stage_progress() -> None:
    info = printer_status()
    focusing = autofocus_status()["running"]
    capture = scan_capture_status()
    if capture["error"] and not capture["running"]:
        st.error(capture["error"])
    if capture["running"]:
        st.session_state["_scan_capture_was_running"] = True
        total = max(capture["total"], 1)
        label = f"({capture['row']},{capture['col']})" if capture["row"] else "starting"
        if capture["stopping"]:
            st.caption(f"Stopping after this stop. Last picture is {label}.")
        elif capture["focusing"]:
            st.progress(
                min(capture["index"], total) / total,
                text=(
                    f"Autofocus after step {capture['steps']}"
                    + (
                        f", attempt {capture['focus_attempt']}"
                        if capture.get("focus_attempt", 1) > 1
                        else ""
                    )
                    + f". Sharpness {_format_score(capture['sharpness'])}."
                ),
            )
        else:
            st.progress(
                min(capture["index"], total) / total,
                text=f"Stop {capture['index']} of {capture['total']} — {label}",
            )
        if st.button("Stop scan and capture", type="primary"):
            stop_scan_capture()
        return
    if st.session_state.pop("_scan_capture_was_running", False):
        st.rerun(scope="app")
    if capture["saved"] and not capture["error"]:
        where = f" to {capture['folder']}" if capture.get("folder") else ""
        st.success(f"Saved {capture['saved']} stops{where}.")
    if not info["connected"]:
        st.info("Printer not connected")
    elif focusing:
        st.caption("Autofocus is using the printer.")
    elif info["printing"]:
        total = max(info["total"], 1)
        index = min(info["index"], total)
        st.progress(index / total, text=f"Moving — command {index} of {info['total']}")
        if st.button("Stop scan", type="primary"):
            stop_scan()
            st.rerun(scope="app")
        return
    elif info["online"]:
        st.success(f"Printer ready on {info['port']}")
    else:
        st.warning(f"Port {info['port']} is open. Waiting for the printer to answer.")
    if info["error"]:
        st.caption(info["error"])
    st.caption(
        "Scan and capture uses the lighting mode above and leaves the ring as it is "
        "unless the Arduino is connected. NSEW takes N, E, S, and W before the stage moves again. "
        "After every 7 stage steps it refocuses with the Focus settings. "
        "If sharpness is not above 80% of the score autofocus started with, it autofocuses again."
    )
    move_col, both_col = st.columns(2)
    if move_col.button("Start scan", type="primary", disabled=not info["online"] or focusing):
        try:
            start_gcode(_stage_program())
        except Exception as exc:
            st.error(str(exc))
        else:
            st.rerun(scope="app")
    if both_col.button(
        "Scan and capture",
        disabled=(
            not info["online"]
            or focusing
            or _focus_limits() is not None
            or (_selected_mode() == "NSEW" and _serial() is None)
        ),
    ):
        try:
            start_scan_capture(
                ser=_serial(),
                mode=_selected_mode(),
                total_exposure_ms=float(st.session_state["capture_total_ms"]),
                step_mm=float(st.session_state["stage_step_mm"]),
                columns=int(st.session_state["stage_columns"]),
                rows=int(st.session_state["stage_rows"]),
                feed=float(st.session_state["stage_feed"]),
                focus_max_travel_mm=float(st.session_state.get("focus_max_travel_mm", 1.0)),
                focus_coarse_mm=float(st.session_state.get("focus_coarse_mm", 0.05)),
                focus_fine_mm=float(st.session_state.get("focus_fine_mm", 0.01)),
                focus_feed=float(st.session_state.get("focus_feed", 300)),
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.rerun(scope="app")


def _format_jog_mm(delta_mm: float) -> str:
    return f"{delta_mm:+.3f}".rstrip("0").rstrip(".")


def _jog_stage(axis: str, delta_mm: float) -> None:
    feed = float(st.session_state.get("stage_feed", DEFAULT_FEED))
    try:
        move_axis(axis, delta_mm, feed)
    except Exception as exc:
        st.error(str(exc))
    else:
        st.success(f"Moved {axis} {_format_jog_mm(delta_mm)} mm.")


def _render_stage_jog(locked: bool) -> None:
    """Short steps so a connected printer can be checked without starting a scan."""
    st.caption(
        "Jog X and Y by 1 mm, or Z by 0.1 mm or 1 mm, from the current position. "
        "Uses the moving speed below. Positive Z is up."
    )
    info = printer_status()
    ready = info["online"] and not locked
    x_minus, x_plus, y_minus, y_plus = st.columns(4)
    if x_minus.button("X −1 mm", disabled=not ready, width="stretch"):
        _jog_stage("X", -1)
    if x_plus.button("X +1 mm", disabled=not ready, width="stretch"):
        _jog_stage("X", 1)
    if y_minus.button("Y −1 mm", disabled=not ready, width="stretch"):
        _jog_stage("Y", -1)
    if y_plus.button("Y +1 mm", disabled=not ready, width="stretch"):
        _jog_stage("Y", 1)
    z_minus, z_down, z_up, z_plus = st.columns(4)
    if z_minus.button("Z −1 mm", disabled=not ready, width="stretch"):
        _jog_stage("Z", -1)
    if z_down.button("Z −0.1 mm", disabled=not ready, width="stretch"):
        _jog_stage("Z", -0.1)
    if z_up.button("Z +0.1 mm", disabled=not ready, width="stretch"):
        _jog_stage("Z", 0.1)
    if z_plus.button("Z +1 mm", disabled=not ready, width="stretch"):
        _jog_stage("Z", 1)


def _render_stage(cap) -> None:
    """Move the 3D-printer stage with the same serpentine as scan_40x60.gcode."""
    st.subheader("Stage")

    try:
        ports = cap.list_serial_ports()
    except ImportError:
        st.error("pyserial is not installed. From the project folder: pip install pyserial")
        return

    devices = [device for device, _label in ports]
    labels = {device: label for device, label in ports}
    info = printer_status()
    connected = info["connected"]

    if devices:
        if st.session_state.get("printer_port") not in devices:
            st.session_state.pop("printer_port", None)
        st.selectbox(
            "Printer port",
            devices,
            index=None,
            placeholder="Select a port",
            key="printer_port",
            format_func=lambda device: labels.get(device, device),
            disabled=connected,
        )
    else:
        st.info("No serial ports found. Plug in the printer and reopen this tab.")

    connect_col, disconnect_col = st.columns(2)
    printer_port = st.session_state.get("printer_port")
    if connect_col.button("Connect Printer", disabled=not printer_port or connected):
        if _LINK.get("port") and printer_port == _LINK.get("port") and _serial() is not None:
            st.error("That port is already the Arduino. Choose the other port for the printer.")
        else:
            try:
                connect_printer(printer_port)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.rerun()
    if disconnect_col.button("Disconnect Printer", disabled=not connected):
        disconnect_printer()
        st.rerun()

    scanning = (
        printer_status()["printing"]
        or autofocus_status()["running"]
        or scan_capture_status()["running"]
    )
    _render_stage_jog(scanning)
    step_col, x_col, row_col = st.columns(3)
    step_col.number_input(
        "Step (mm)",
        min_value=0.1,
        max_value=50.0,
        value=float(STAGE_WIDGET_DEFAULTS["stage_step_mm"]),
        step=0.001,
        format="%.3f",
        key="stage_step_mm",
        disabled=scanning,
        help="Distance between picture centers. 6.174 mm is a 6.86 mm field of view with 10 percent overlap.",
    )
    x_col.number_input(
        "Columns",
        min_value=1,
        max_value=80,
        value=int(STAGE_WIDGET_DEFAULTS["stage_columns"]),
        step=1,
        key="stage_columns",
        disabled=scanning,
        help="Stops along X. The stage steps one fewer time than this.",
    )
    row_col.number_input(
        "Rows",
        min_value=1,
        max_value=80,
        value=int(STAGE_WIDGET_DEFAULTS["stage_rows"]),
        step=1,
        key="stage_rows",
        disabled=scanning,
        help="Stops along Y. The stage steps one fewer time than this.",
    )
    st.number_input(
        "Moving speed (mm/min)",
        min_value=1,
        max_value=6000,
        value=int(STAGE_WIDGET_DEFAULTS["stage_feed"]),
        step=100,
        key="stage_feed",
        disabled=scanning,
    )

    summary = scan_summary(
        float(st.session_state["stage_step_mm"]),
        int(st.session_state["stage_columns"]),
        int(st.session_state["stage_rows"]),
    )
    pause = _stage_pause()
    frame_word = "frame" if pause["frames"] == 1 else "frames"
    st.caption(
        f"{summary['columns']} × {summary['rows']} stops, "
        f"travel {summary['x_mm']:.2f} mm in X and {summary['y_mm']:.2f} mm in Y. "
        f"Pause {pause['dwell_ms']} ms at each stop: {pause['frames']} {frame_word} "
        f"at {pause['shutter_ms']:.1f} ms shutter, {pause['motion_ms']:.0f} ms after each move, "
        f"{pause['settle_ms']:.0f} ms light settle, "
        f"and {pause['margin_ms']} ms margin on each frame"
        + (
            f", plus {pause['gap_ms']:.0f} ms between sides."
            if pause["gaps"]
            else "."
        )
    )

    try:
        program = _stage_program()
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.expander("G-code based on current settings"):
        st.code(program, language="text")
        st.download_button(
            "Download gcode",
            data=program,
            file_name="scanwafer.gcode",
            mime="text/plain",
            disabled=scanning,
            key="stage_download_gcode",
        )

    st.divider()
    _render_focus(scanning)
    _stage_progress()