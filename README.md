# Particle detection software

This app finds contaminant particles on optical-microscope images of a patterned wafer, then lets you review, label, and optionally retrain a filter on those hits. Image analysis runs on a folder of tiles. Camera capture, the ring light, and the motorized stage are optional and only needed when you are taking new pictures.

Algorithm detail lives in [HOW_IT_WORKS.md](HOW_IT_WORKS.md). This file is the setup and operating guide.

## What you need

**To analyze images you already have**

- macOS (this setup is what the capture code expects; the analysis itself is ordinary Python)
- Python 3.11 or newer (this project has been run on 3.13)
- A folder of tile images: TIFF, BMP, PNG, or JPEG

**To capture new images** (Capture tab only)

- A Basler camera that pypylon can open, with the Basler pylon runtime installed
- An Arduino Uno (or a CH340 clone) running the ring-light sketch
- A 3D-printer-style stage that speaks G-code, plus [Printrun](https://github.com/kliment/Printrun) (`printcore`)
- The Arduino and the printer must be on **two different USB serial ports**

You can ignore the camera, Arduino, and printer until you use the Capture tab.

## Folder layout

The app lives in `particle_detection/`. Inputs and outputs sit next to that folder, not inside it:

```
ASML SE/
  Inputs/                 tile folders the pipeline reads
    Groundup/             new captures are written here
      v1 2026-10-19/      one run: (1,1).png, (1,1)_N.png, capture.json, …
  Outputs/                pipeline results (particles.csv, mosaic, PDF, …)
  particle_detection/
    app.py                Streamlit entry point
    config.yaml           standard detection settings
    nsew_config.yaml      settings used when the sidebar NSEW toggle is on
    requirements.txt
    models/               trained classifiers (*.joblib)
    labels/               labels.csv written by the Verification tab
    src/
    tests/
```

A bare folder name in the sidebar, such as `Groundup/v13 2026-10-16`, is resolved under `ASML SE/Inputs`. A bare output name, such as `v13 2026-10-16 v1`, is created under `ASML SE/Outputs`. An absolute path is used as-is.

Tile names the pipeline places on the mosaic match this pattern (see `filename_pattern` in `config.yaml`):

```text
(row,col).tif
(9,7).png
(1,1)_N.png
```

Row and column are integers. Extension can be `tif`, `tiff`, `bmp`, `png`, `jpg`, or `jpeg`. Capture writes exactly this style. Files that do not match are still detected, but they are not placed on the stitched mosaic.

## 1. Install Python dependencies

From a terminal:

```bash
cd "/Users/rebeccajekel/Desktop/ASML SE/particle_detection"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs the analysis stack (NumPy, OpenCV, scikit-image, pandas, scikit-learn), Streamlit, PyYAML, pyserial, and pypylon.

Check that the app can import:

```bash
python -c "import streamlit, cv2, yaml, serial; print('ok')"
```

### Basler pylon (only if you will use the camera)

`pypylon` needs the Basler **pylon Camera Software Suite** installed on the machine before the camera will open. Install the macOS pylon runtime from Basler, then confirm the Python binding:

```bash
python -c "from pypylon import pylon; print(len(pylon.TlFactory.GetInstance().EnumerateDevices()), 'camera(s)')"
```

A count of `0` means the runtime is present but no camera is connected or powered. An import error means pylon or pypylon is not installed correctly.

### Stage control (only if you will move the printer)

Printrun is **not** in `requirements.txt`. Install it into the same virtual environment:

```bash
source .venv/bin/activate
pip install Printrun
```

The stage code also looks for a pipx Pronterface install under `~/Library/Application Support/pipx/venvs/printrun/` if the package is missing from the venv. Prefer installing it in `.venv`.

## 2. Start the app

Always run Streamlit from `particle_detection/`, with the virtual environment active:

```bash
cd "/Users/rebeccajekel/Desktop/ASML SE/particle_detection"
source .venv/bin/activate
streamlit run app.py
```

The browser opens on **Particle Detection Software**. Leave that terminal running. Stop the app with Ctrl+C in that terminal.

The window has six tabs: **Main Pipeline**, **Verification**, **Training**, **Shadow**, **Redirector**, and **Capture**. Settings for a detection run live in the left sidebar and come from `config.yaml`.

## 3. Run detection on an existing folder

1. Put the tiles in a folder under `ASML SE/Inputs`, or note the absolute path if they live somewhere else.
2. Open **Main Pipeline**.
3. In the sidebar **Loading** section, set the input folder. A name relative to `Inputs/` is enough.
4. Set the output folder name. Leave it blank to let the app pick the next `{input folder name} vN` under `Outputs/`.
5. Check **Pixel size**. The sidebar shows micrometres per pixel. `config.yaml` ships with `pixel_size_nm: 3350` (3.35 µm/px). Sizes in the results table are only meaningful if this matches the objective you used.
6. Leave the other sidebar values alone for a first run. **Reset to standard values** copies `config.yaml` back into the sidebar and turns the NSEW toggle off.
7. Click **Run pipeline**.

The job runs in the background. Status stays on the Main Pipeline tab. When it finishes, the tab shows the particle table, a size histogram in micrometres, and a downsampled mosaic with a marker on each hit.

**NSEW toggle.** Turn this on only for a folder shot under one-side-at-a-time lighting (filenames like `(1,1)_N.png`). It overlays `nsew_config.yaml` on top of the standard settings. Uniform (all four sides on) runs should leave the toggle off.

### Files written for each run

Under the chosen folder in `Outputs/`:

| File | What it is |
|------|------------|
| `particles.csv` | One row per merged particle: position, size, confidence, source tile |
| `particles.xlsx` | The same table in Excel |
| `mosaic_overlay.png` or `.jpg` | Stitched overview with detections marked |
| `last_run.pdf` | Report of that run |
| `last_pipeline.json` | What the app reloads for **Last run** |
| `parameters.json` | Settings used for that run |

**Last run** on the Main Pipeline tab reopens the most recent finished table and mosaic without starting another job.

## 4. Review and label

Open **Verification**.

**Tinder** walks detections one at a time. Mark each as a particle or not a particle. Labels are appended to `particle_detection/labels/labels.csv`. Keys that are already labeled, including N/S/E/W aliases of the same hit, are skipped.

**Pointer** opens one tile at a time so you can zoom and mark a speck the detector missed.

**Database** shows saved crops in two campaigns, **Groundup** and **Glovebox**, split into particles and not-particles. **Area line** outlines the measured area on particle crops.

Label circled JPEGs are for people. Training does not read those circled images. It reads `labels.csv` and the original unmarked tiles.

## 5. Train a new filter (optional)

The detector is classical image processing. The model only scores blobs that detector already found. It does not replace detection.

1. Label a useful set of hits in Verification first.
2. Open **Training**.
3. Leave **Mode** on `patch` unless you specifically want the older residual model (that mode needs a `particles.csv`).
4. **Labels CSV** defaults to `labels/labels.csv`.
5. **Tile folder** should be the same images you labeled. Blank uses the folder in `config.yaml` / the sidebar.
6. Leave **Output model** blank. Training writes the next `models/particle_clf_vN.joblib` and does not overwrite an existing version.
7. Click **Run training**.

Point the sidebar ML model path at the new file and set the threshold, or edit `ml.model_path` and `ml.threshold` in `config.yaml`. If `ml.enabled` is true and that file is missing, the pipeline stops. It will not silently skip the filter.

The same training entry point from a terminal, still inside `particle_detection/` with the venv active:

```bash
python -m src.training
python -m src.training --mode residual --detections "../Outputs/recall/particles.csv"
```

## 6. Set up the capture hardware

Skip this section if you only analyze folders that already exist.

Only one program should hold each USB port. Quit the Arduino Serial Monitor and Pronterface before connecting from this app. The Capture tab is the program that opens both ports.

### 6.1 Camera

1. Install the Basler pylon runtime (section 1).
2. Connect and power the camera.
3. Confirm `EnumerateDevices()` reports at least one device (section 1).

The Capture tab opens the first camera pylon enumerates. Exposure is computed from the **total exposure** control: that number is the dose from a single ring side, in milliseconds. Uniform lighting has all four sides on, so the shutter is that value divided by 4. NSEW lights one side, so the shutter equals the total.

### 6.2 Ring light and Arduino

The Capture tab loads `src/capture/camera_automation_v3.py`. Upload this sketch once with the Arduino IDE:

`src/capture/Inital arduino code/arduino_camera_v3`

That sketch drives four transistor-switched sides. Quadrant 0 is N, then E, S, W, on Arduino pins **2, 3, 4, and 5**. Serial is **9600 baud**. On boot it prints `READY`. Commands are one ASCII line ending in a newline: `f` (all on), `0`–`3` (one side), `x` (all off).

Per side, for an NPN low-side switch (as commented in the sketch):

- Arduino pin → about 1 kΩ → transistor base
- Emitter → supply ground, and that ground tied to Arduino ground
- Collector → LED strip negative
- LED strip positive → the LED supply (5 V or 12 V, matching the strip)

Do not power a full ring from the Arduino USB port. Use an external supply and share ground with the Arduino.

There is a second sketch, `src/capture/arduino_camera_v2.ino`, for an addressable FastLED ring on pin 6. The app does **not** use that sketch. Upload the v3 file above.

### 6.3 Stage

The stage is a printer-style controller. Motion is relative G-code (`G91`). Nothing is homed. The scan starts from wherever the stage is when you press start.

Defaults in `src/capture/stage.py`:

- Field of view 6.86 mm
- 10% overlap, so the step is 6.174 mm
- A 60 mm by 40 mm area
- Feed 1000
- Baud **115200**

Odd rows run left to right. Even rows run back toward column 1. Between rows the stage steps in −Y.

Jog buttons move 1 mm in X or Y, and 1 mm or 0.1 mm in Z, so you can put the wafer in frame and focus before a scan.

## 7. Capture a run

1. Start the app (section 2) and open **Capture**.
2. Choose **Uniform** (all four sides on) or **NSEW** (one side at a time). NSEW can be **Automatic** (N, E, S, W in one click) or **Manual** (one side per click).
3. Plug in the Arduino. Select its port. Click **Connect Arduino**. If Uniform is selected, the ring turns on. If NSEW is selected, the ring turns off until a capture.
4. Plug in the printer on the **other** port. Select that port. Click **Connect Printer**. The app refuses to open the same port for both.
5. Jog X, Y, and Z until the field is framed and in focus, or use **Autofocus**. Autofocus needs the printer connected and the camera free.
6. Set columns, rows, step, and feed if the default 60×40 mm grid is wrong. Set total exposure.
7. Start the scan. Each stop pauses, then the camera grabs. Uniform writes `(row,col).png`. NSEW writes `(row,col)_N.png`, `_E`, `_S`, and `_W`.

Each run is created only after the camera accepts setup, at:

```text
ASML SE/Inputs/Groundup/vN YYYY-MM-DD
```

`N` is the next run number for that date, starting at 1. `capture.json` in that folder records the stops.

Leaving the Capture tab closes the live view so the camera is not held open. Disconnect Arduino and Disconnect Printer when you are done, so the ports are free next time.

To analyze what you just captured, go to **Main Pipeline**, set the input folder to that `vN YYYY-MM-DD` directory, and run the pipeline. Use the NSEW sidebar toggle when the folder contains one-side images.

Command-line capture (same code, no Streamlit), from `particle_detection/` with the venv active:

```bash
python src/capture/camera_automation_v3.py --mode uniform --total-exposure-ms 100
python src/capture/camera_automation_v3.py --mode nsew --total-exposure-ms 100
```

## 8. The other tabs

**Shadow** measures the dark region on the side of a particle away from the light, either from labeled database crops or from an uploaded tile. Pixel size is the sidebar value.

**Redirector** is a design view of the plexiglass ring that aims the ring light. It does not move hardware. **Default values** and **Optimal values** fill the diameter, drop, thickness, and angle fields. Click a ray in the drawing to highlight it.

## 9. Check the install with tests

From `particle_detection/` with the venv active:

```bash
pytest
```

Tests do not need the camera, Arduino, or printer. They use synthetic tiles in `tests/fixtures/synthetic.py`.

## 10. When something fails

**`streamlit: command not found`**  
The virtual environment is not active, or `pip install -r requirements.txt` did not finish. Run `source .venv/bin/activate` inside `particle_detection/` and install again.

**Pipeline stays idle or the UI never updates**  
Run `streamlit run app.py` from `particle_detection/`, not from another folder. The app puts that folder on `sys.path`.

**No tiles found**  
The sidebar folder must contain images, and names you want on the mosaic must look like `(row,col).png`. A bare name is looked up under `ASML SE/Inputs`.

**Sizes look wrong by a constant factor**  
Pixel size in the sidebar does not match the objective. Fix it before comparing micrometres.

**`ml.enabled` error about a missing joblib**  
`config.yaml` points at `models/particle_clf_v4_residual.joblib`. That file is in `models/` in this tree. If you moved the project without `models/`, turn ML off in the sidebar or copy the `.joblib` files across.

**No serial ports**  
Plug in the board and reopen the Capture tab. On macOS the port is usually `/dev/cu.usbserial-…` or `/dev/cu.usbmodem…`.

**Arduino connects then every command fails**  
The wrong sketch is on the board, or another program still has the port. Upload `arduino_camera_v3`, quit Serial Monitor, disconnect, and connect again. The v3 sketch answers `OK OFF` to `x`.

**“That port is already the printer”**  
Arduino and printer were given the same device. Pick the other port.

**`Printrun is not available`**  
`pip install Printrun` inside the same `.venv` you use to launch Streamlit.

**Camera does not open**  
Install the Basler pylon runtime, then check `from pypylon import pylon`. The live view and a scan cannot share the camera. Wait until a scan finishes, or stop it, before opening live view. Only one camera is used: the first device pylon lists.

**Printer moves but the wafer walks off the mosaic**  
The scan is relative and does not home. Jog back to the start corner before the next scan. Overlap on the mosaic is the sidebar **overlap fraction** (default 0.1), not a measured registration. If neighbouring tiles do not meet, set that fraction to the real overlap.

**A run is slow**  
Full-resolution detection on every tile is expected to take minutes. `pipeline.workers` in `config.yaml` is the thread count (`10` in the shipped file). Lower it if the machine is unusable while a job runs.
