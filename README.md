# Particle detection software (TL;DR)

This app finds particles on darkfield images of a patterned wafer, then lets you review the hits. You need a folder of tile images (TIFF, BMP, PNG, or JPEG). The setup (camera, ring light, and stage) is only needed when you are taking new pictures.

How the detector actually works is all written down in the [HOW_IT_WORKS.md](HOW_IT_WORKS.md), if you're interested :).

## Install

You need Python 3.11 or newer. 

On Mac: From this folder in your terminal, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows: Activate with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

## Install

With the virtual environment active, from this folder:

```bash
streamlit run app.py
```

The browser opens on **Particle Detection Software**. Leave the terminal running. Stop the app with Ctrl+C.

## Starting the app

From now onwards, you can also double-click **Particle Detection.app** (Mac) or **Start Particle Detection.bat** (Windows). The first Mac launch may need Control-click → Open. A second launch only opens the browser if the app is already running. Quit the Mac app from the Dock, or close the Windows console, to stop it.

Congrats! You've read the whole README!
