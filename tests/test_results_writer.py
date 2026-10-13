from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from src.overall.io.results_writer import (
    PARAMETERS_SHEET,
    PARTICLES_SHEET,
    excel_bytes,
    parameters_table,
    write_xlsx,
)


def _particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1],
            "x_global": [100.0],
            "y_global": [200.0],
            "size": [15000.0],
            "confidence": [0.9],
            "source_tile": ["R3_1_1_5X.tif"],
        }
    )


def test_parameters_table_flattens_nested_run_settings() -> None:
    table = parameters_table(
        {
            "input_dir": "R3 04-08",
            "pixel_size_nm": 960.0,
            "detection": {
                "min_size_nm": 20000.0,
                "recall_mode": False,
                "recall": {"min_confidence": 0.4},
            },
            "preprocessing": {"contrast_percentiles": [1.0, 99.0]},
        }
    )
    values = dict(zip(table["parameter"], table["value"]))
    assert values["input_dir"] == "R3 04-08"
    assert values["pixel_size_nm"] == 960.0
    assert values["detection.min_size_nm"] == 20000.0
    assert values["detection.recall_mode"] is False
    assert values["detection.recall.min_confidence"] == 0.4
    assert values["preprocessing.contrast_percentiles"] == "[1.0, 99.0]"


def test_write_xlsx_adds_parameters_sheet(tmp_path: Path) -> None:
    config = {
        "output_dir": "Single v7",
        "detection": {"min_size_nm": 20000.0, "method": "fft"},
    }
    path = write_xlsx(_particles(), tmp_path / "particles.xlsx", config)
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets) == [PARTICLES_SHEET, PARAMETERS_SHEET]
    particles = sheets[PARTICLES_SHEET]
    assert list(particles["source_tile"]) == ["R3_1_1_5X.tif"]
    params = dict(
        zip(sheets[PARAMETERS_SHEET]["parameter"], sheets[PARAMETERS_SHEET]["value"])
    )
    assert params["output_dir"] == "Single v7"
    assert params["detection.min_size_nm"] == 20000.0
    assert params["detection.method"] == "fft"


def test_particles_at_least_counts_cumulative_sizes() -> None:
    from src.main_pipeline.report.report_generator import particles_at_least

    table = pd.DataFrame({"size": [9_000.0, 10_000.0, 20_000.0, 25_000.0, 49_999.0, 75_000.0]})
    counts = dict(zip(particles_at_least(table)["Size"], particles_at_least(table)["Particles"]))
    assert int(counts["≥10 µm"]) == 5
    assert int(counts["≥20 µm"]) == 4
    assert int(counts["≥25 µm"]) == 3
    assert int(counts["≥50 µm"]) == 1
    assert int(counts["≥75 µm"]) == 1
    empty = particles_at_least(pd.DataFrame())
    assert list(empty["Size"]) == ["≥10 µm", "≥20 µm", "≥25 µm", "≥50 µm", "≥75 µm"]
    assert list(empty["Particles"]) == [0, 0, 0, 0, 0]


def test_excel_bytes_includes_parameters_sheet() -> None:
    book = pd.ExcelFile(BytesIO(excel_bytes(_particles(), {"input_dir": "tiles"})))
    assert book.sheet_names == [PARTICLES_SHEET, PARAMETERS_SHEET]
    params = pd.read_excel(book, sheet_name=PARAMETERS_SHEET)
    assert list(params["parameter"]) == ["input_dir"]
    assert list(params["value"]) == ["tiles"]
