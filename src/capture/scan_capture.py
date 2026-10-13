"""Move the serpentine and take a picture at every stop.

Uniform saves one all-sides frame per stop. NSEW keeps the stage still until
N, E, S, and W have each been saved. Start scan still only moves the printer.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from src.capture.camera_automation_v2 import (
    CAPTURES_DIR,
    MOTION_PAUSE_S,
    RING_SIDES,
    allocate_run_folder,
    capture_nsew,
    capture_uniform,
    grab_timeout_ms,
    open_camera,
    shutter_ms,
    write_manifest,
)
from src.capture.stage import move_axis, printer_status, scan_stops

_JOB_LOCK = threading.Lock()
_JOB: dict = {
    "thread": None,
    "stop": None,
    "running": False,
    "stopping": False,
    "index": 0,
    "total": 0,
    "row": 0,
    "col": 0,
    "saved": 0,
    "error": "",
    "mode": "",
    "folder": "",
}


def scan_capture_status() -> dict:
    with _JOB_LOCK:
        thread = _JOB.get("thread")
        if _JOB["running"] and thread is not None and not thread.is_alive():
            _JOB["running"] = False
            _JOB["stopping"] = False
            if not _JOB["error"]:
                _JOB["error"] = "The scan stopped before the first picture."
        return {
            "running": bool(_JOB["running"]),
            "stopping": bool(_JOB["stopping"]),
            "index": int(_JOB["index"]),
            "total": int(_JOB["total"]),
            "row": int(_JOB["row"]),
            "col": int(_JOB["col"]),
            "saved": int(_JOB["saved"]),
            "error": _JOB["error"],
            "mode": _JOB["mode"],
            "folder": _JOB["folder"],
        }


def stop_scan_capture() -> None:
    """Finish the stop in progress, then do not move to the next one."""
    with _JOB_LOCK:
        stop = _JOB.get("stop")
        if stop is not None:
            stop.set()
            _JOB["stopping"] = True


def start_scan_capture(
    *,
    ser,
    mode: str,
    total_exposure_ms: float,
    step_mm: float,
    columns: int,
    rows: int,
    feed: float,
    captures_dir: Path = CAPTURES_DIR,
) -> None:
    """Start the combined scan on a background thread."""
    if mode not in ("Uniform", "NSEW"):
        raise ValueError("mode must be Uniform or NSEW")
    if ser is None or not getattr(ser, "is_open", False):
        raise RuntimeError("Connect the Arduino first.")
    status = printer_status()
    if not status["connected"]:
        raise RuntimeError("Connect the printer first.")
    if not status["online"]:
        raise RuntimeError("The printer has not answered yet. Wait a moment and try again.")
    if status["printing"]:
        raise RuntimeError("A scan is already running.")
    stops = scan_stops(columns, rows, step_mm)
    with _JOB_LOCK:
        thread = _JOB.get("thread")
        if thread is not None and thread.is_alive():
            raise RuntimeError("A capture scan is already running.")
        stop = threading.Event()
        _JOB["stop"] = stop
        _JOB["running"] = True
        _JOB["stopping"] = False
        _JOB["index"] = 0
        _JOB["total"] = len(stops)
        _JOB["row"] = 0
        _JOB["col"] = 0
        _JOB["saved"] = 0
        _JOB["error"] = ""
        _JOB["mode"] = mode
        _JOB["folder"] = ""
    worker = threading.Thread(
        target=_scan_thread,
        kwargs={
            "ser": ser,
            "mode": mode,
            "total_exposure_ms": float(total_exposure_ms),
            "feed": float(feed),
            "stops": stops,
            "captures_dir": Path(captures_dir),
            "stop": stop,
        },
        name="scan-capture",
        daemon=True,
    )
    with _JOB_LOCK:
        _JOB["thread"] = worker
    worker.start()


def _scan_thread(
    *,
    ser,
    mode: str,
    total_exposure_ms: float,
    feed: float,
    stops: list[tuple[int, int, str | None, float]],
    captures_dir: Path,
    stop: threading.Event,
) -> None:
    from src.capture.camera_automation_v2 import shutdown
    from src.capture.preview import stop_preview

    camera = None
    saved = 0
    try:
        sides_on = RING_SIDES if mode == "Uniform" else 1
        shutter = shutter_ms(total_exposure_ms, sides_on)
        exposure_us = shutter * 1000.0
        timeout_ms = grab_timeout_ms(exposure_us)
        stop_preview()
        camera, converter = open_camera(exposure_us)
        run_dir = allocate_run_folder(captures_dir)
        with _JOB_LOCK:
            _JOB["folder"] = str(run_dir)
        for index, (row, col, axis, delta) in enumerate(stops, start=1):
            if stop.is_set():
                break
            with _JOB_LOCK:
                _JOB["index"] = index
                _JOB["row"] = row
                _JOB["col"] = col
            if axis is not None:
                move_axis(axis, delta, feed)
                time.sleep(MOTION_PAUSE_S)
            folder = run_dir / f"({row},{col})"
            folder.mkdir()
            if mode == "Uniform":
                files = [capture_uniform(camera, converter, ser, folder, timeout_ms)]
            else:
                files = capture_nsew(camera, converter, ser, folder, timeout_ms)
            write_manifest(
                folder,
                {
                    "mode": "uniform" if mode == "Uniform" else "nsew",
                    "row": row,
                    "col": col,
                    "total_exposure_ms": total_exposure_ms,
                    "shutter_ms": shutter,
                    "exposure_us": exposure_us,
                    "sides_on": sides_on,
                    "files": files,
                },
            )
            saved += 1
            with _JOB_LOCK:
                _JOB["saved"] = saved
    except Exception as exc:
        with _JOB_LOCK:
            _JOB["error"] = str(exc)
    finally:
        try:
            shutdown(camera, None)
        except Exception as exc:
            with _JOB_LOCK:
                if not _JOB["error"]:
                    _JOB["error"] = str(exc)
        with _JOB_LOCK:
            _JOB["running"] = False
            _JOB["stopping"] = False
            _JOB["saved"] = saved
