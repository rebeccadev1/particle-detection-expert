"""Measure the shadow of already-marked particles, picked by ID.

Example::

    python src/shadow/shadow_check.py --key R3_1_17_5X_42943680_1240320 --light-from N

``--csv`` is ``labels/labels.csv`` (ID = ``particle_id``) or a run's
``particles.csv`` (ID = ``id``). Label IDs repeat across runs: narrow them
with ``--run`` (substring of ``source_csv``) or pass ``--key``.

Writes ``Outputs/Shadow check/shadow_check.csv`` and one PNG per particle.
Same measurement as the **Shadow** tab (``src/shadow/shadow.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.overall.config import load_config  # noqa: E402
from src.shadow.shadow import (  # noqa: E402
    DEFAULT_OUT,
    LABELS_CSV,
    LIGHT_HEIGHT_MM,
    RING_CENTRE_MM,
    RING_EDGE_MM,
    run_shadow_check,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ids", type=float, nargs="*", default=[], help="Particle IDs.")
    parser.add_argument("--key", nargs="*", default=[], help="Exact label keys.")
    parser.add_argument("--csv", type=Path, default=LABELS_CSV,
                        help="labels.csv (default) or a run's particles.csv.")
    parser.add_argument("--run", default=None, help="Substring of source_csv to narrow labels.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--input-dir", default=None,
                        help="Tile folder (default: found under Inputs from each label).")
    parser.add_argument("--tile", type=Path, default=None,
                        help="Measure on this image instead of the row's source tile "
                             "(e.g. a bright-field shot of the same field).")
    parser.add_argument("--pixel-size-nm", type=float, default=None)
    parser.add_argument("--light-from", default=None,
                        help="N/S/E/W or degrees clockwise from image top. "
                             "Default: taken from an N/S/E/W tile name.")
    parser.add_argument("--light-height", type=float, default=LIGHT_HEIGHT_MM,
                        help="Light height above the wafer (mm).")
    parser.add_argument("--ring-edge", type=float, default=RING_EDGE_MM,
                        help="Distance to the light (mm) at the image edge on the light side.")
    parser.add_argument("--ring-centre", type=float, default=RING_CENTRE_MM,
                        help="Distance to the light (mm) at the image centre.")
    parser.add_argument("--k", type=float, default=3.0, help="Shadow threshold in noise σ.")
    parser.add_argument("--max-length-factor", type=float, default=3.0,
                        help="Search window: shadow up to this many diameters long.")
    parser.add_argument("--preprocess", choices=("light", "config"), default="light",
                        help="light = float + 1 px blur; config = full config corrections.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Folder under Outputs/.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.ids and not args.key:
        print("Give --ids and/or --key.")
        return 2
    config = load_config(args.config)
    if args.pixel_size_nm:
        config["pixel_size_nm"] = float(args.pixel_size_nm)
    result = run_shadow_check(
        pd.read_csv(args.csv),
        config,
        ids=args.ids,
        keys=args.key,
        run=args.run,
        input_dir=args.input_dir,
        tile=args.tile,
        light_from=args.light_from,
        light_height_mm=args.light_height,
        ring_edge_mm=args.ring_edge,
        ring_centre_mm=args.ring_centre,
        k_mad=args.k,
        max_length_factor=args.max_length_factor,
        preprocess=args.preprocess,
        out=args.out,
    )
    for message in result.messages:
        print(message)
    if result.table.empty:
        return 1
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(result.table.to_string(index=False))
    print(f"\nWrote {result.out_dir / 'shadow_check.csv'} and {len(result.images)} PNG(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
