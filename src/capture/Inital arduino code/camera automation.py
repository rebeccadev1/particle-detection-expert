from pypylon import pylon
import cv2
import time
import serial
import os
import re

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
ARDUINO_PORT = "COM7"       # <-- change to your port (Windows: "COM3" etc.,
                             #     Linux/Mac: "/dev/ttyACM0" or "/dev/ttyUSB0")
ARDUINO_BAUD = 9600

# Must match `amountOfQuadrants` in the Arduino sketch — the two
# programs don't share this value automatically, so update both.
amountOfQuadrants = 4          # e.g. 4, 6, 8, ...

num_images = amountOfQuadrants  # one shot per segment; change if you want more/less
exposure = 100000.0           # 50 ms
settle_delay = 0.3           # seconds to let the LEDs/strip settle after switching
pause_between_shots = 2.0    # seconds between shots (was 5.0 in your version)

BASE_DIR = "."               # where the v1, v2, v3, ... folders get created


def make_next_run_folder(base_dir=BASE_DIR, prefix="v"):
    """Find the next free v<N> folder (v1, v2, v3, ...) and create it."""
    pattern = re.compile(rf"^{prefix}(\d+)$")
    existing = []
    for name in os.listdir(base_dir):
        match = pattern.match(name)
        if match and os.path.isdir(os.path.join(base_dir, name)):
            existing.append(int(match.group(1)))

    next_n = max(existing, default=0) + 1
    folder = os.path.join(base_dir, f"{prefix}{next_n}")
    os.makedirs(folder)
    return folder


run_folder = make_next_run_folder()
print(f"Foto's worden opgeslagen in: {run_folder}")

# ---------------------------------------------------------------
# 1. Connect to the Arduino (LED ring controller)
# ---------------------------------------------------------------
print("Verbinden met Arduino...")
arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=2)
time.sleep(2.0)  # Arduino resets on serial connect; wait for it to boot

# Drain the "READY" boot message
while arduino.in_waiting:
    print("Arduino:", arduino.readline().decode(errors="ignore").strip())


def set_quadrant(q):
    """Tell the Arduino to light segment q (0..amountOfQuadrants-1) and wait for its ack."""
    arduino.reset_input_buffer()
    arduino.write(f"{q}\n".encode())
    ack = arduino.readline().decode(errors="ignore").strip()
    print(f"Arduino ack: {ack}")


# ---------------------------------------------------------------
# 2. Connect to the camera
# ---------------------------------------------------------------
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

if camera.TriggerMode.IsWritable():
    camera.TriggerMode.Value = "On"
    camera.TriggerSource.Value = "Software"

camera.ExposureTime.Value = exposure

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

print("Start met foto's maken...")

try:
    for i in range(num_images):
        quadrant = i % amountOfQuadrants

        # Switch LEDs to the right quadrant for this shot
        set_quadrant(quadrant)
        time.sleep(settle_delay)

        # Trigger and grab exactly one frame
        camera.ExecuteSoftwareTrigger()
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            image = converter.Convert(grab_result)
            img = image.GetArray()

            filename = f"wafer_shot_{i}_quad{quadrant}_exp{int(exposure)}.png"
            filepath = os.path.join(run_folder, filename)
            cv2.imwrite(filepath, img)
            print(f"Foto {i} (kwadrant {quadrant}) gemaakt en opgeslagen: {filepath}")

        grab_result.Release()

        if i < num_images - 1:
            print(f"Pauze na foto {i}")
            time.sleep(pause_between_shots)

finally:
    # 3. Stop and close everything cleanly, even if something went wrong
    camera.StopGrabbing()

    # Leave the camera in free-run mode so tools like pylon Viewer
    # show a live image again afterward (otherwise it stays waiting
    # for a software trigger that only this script sends).
    if camera.TriggerMode.IsWritable():
        camera.TriggerMode.Value = "Off"

    camera.Close()

    arduino.write(b"x")  # turn LEDs off
    time.sleep(0.1)
    arduino.close()

print("Klaar!")
