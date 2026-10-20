"""Capture with the transistor-switched four-side ring.

Upload ``Inital arduino code/arduino_camera_v3`` once with the Arduino IDE.
After that, this script is the only program that should open the USB port.

That sketch does not answer ``?``. On boot it prints ``READY``. This script
waits for that line. A board that is already running does not print it again,
so the script then sends ``x`` and expects ``OK OFF``. The next capture turns
the quadrants back on.

Quadrant 0 is N, then E, S, W, on pins 2, 3, 4, and 5. Lighting commands
after connect are ``f`` (all on), ``0``..``3`` (one side), and ``x`` (all off).

    python camera_automation_v3.py --mode uniform --total-exposure-ms 100
    python camera_automation_v3.py --mode nsew --total-exposure-ms 100
"""

from __future__ import annotations

import sys
import time

try:
    from src.capture import camera_automation_v2 as v2
except ImportError:
    import camera_automation_v2 as v2

# Same capture behavior as v2. Only the serial handshake differs.
ACK_TIMEOUT_S = v2.ACK_TIMEOUT_S
BAUD = v2.BAUD
CAPTURES_DIR = v2.CAPTURES_DIR
DIRECTIONS = v2.DIRECTIONS
GRAB_TIMEOUT_MS = v2.GRAB_TIMEOUT_MS
MOTION_PAUSE_S = v2.MOTION_PAUSE_S
NSEW_GAP_S = v2.NSEW_GAP_S
READY_TIMEOUT_S = v2.READY_TIMEOUT_S
RING_SIDES = v2.RING_SIDES
SETTLE_S = v2.SETTLE_S
TOTAL_EXPOSURE_MS = v2.TOTAL_EXPOSURE_MS

allocate_capture_folder = v2.allocate_capture_folder
allocate_run_folder = v2.allocate_run_folder
capture_nsew = v2.capture_nsew
capture_side = v2.capture_side
capture_uniform = v2.capture_uniform
find_arduino_port = v2.find_arduino_port
folder_location = v2.folder_location
grab_frame = v2.grab_frame
grab_timeout_ms = v2.grab_timeout_ms
image_filename = v2.image_filename
light_all = v2.light_all
lights_off = v2.lights_off
list_serial_ports = v2.list_serial_ports
next_capture_location = v2.next_capture_location
next_run_folder = v2.next_run_folder
open_camera = v2.open_camera
open_live_camera = v2.open_live_camera
orient_frame = v2.orient_frame
parse_args = v2.parse_args
read_live_frame = v2.read_live_frame
save_frame = v2.save_frame
set_exposure = v2.set_exposure
set_segment = v2.set_segment
shutter_ms = v2.shutter_ms
shutdown = v2.shutdown
write_manifest = v2.write_manifest

# A resetting Uno prints READY after the bootloader, usually within a second.
# Listen that long before probing, so the probe is not still in the buffer
# when setup() finishes.
BOOT_LISTEN_S = 2.0


def _handshake(ser) -> str:
    """Return ``READY`` or ``OK OFF``.

    ``READY`` comes from ``setup()`` on the transistor-ring sketch. A CH340
    that is already running does not reset when the port opens, so that line
    is gone. ``x`` is answered from ``loop()`` and switches every quadrant off.
    """
    deadline = time.monotonic() + BOOT_LISTEN_S
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            line = v2._read_line(ser, min(0.5, remaining))
        except TimeoutError:
            continue
        except Exception as exc:
            if "multiple access" in str(exc) or "returned no data" in str(exc):
                raise RuntimeError(
                    f"{ser.port} is already open. Disconnect the other device "
                    "and choose a different port."
                ) from exc
            raise
        if line.startswith("READY"):
            return line
        if line.startswith("ERR"):
            raise RuntimeError(f"Arduino replied {line!r} during connect")

    reply = v2.request_line(ser, "x", timeout_s=READY_TIMEOUT_S)
    if reply != "OK OFF":
        raise RuntimeError(f"expected 'OK OFF', got {reply!r}")
    return reply


def open_arduino(port: str):
    """Open the transistor-ring sketch. It always has four sides, N E S W."""
    import serial

    ser = serial.Serial(port, BAUD, timeout=0.2)
    try:
        _handshake(ser)
    except Exception:
        ser.close()
        raise
    return ser


def main(argv: list[str] | None = None) -> int:
    previous = v2.open_arduino
    v2.open_arduino = open_arduino
    try:
        return v2.main(argv)
    finally:
        v2.open_arduino = previous


if __name__ == "__main__":
    sys.exit(main())
