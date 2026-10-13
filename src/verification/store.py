"""On-disk label table and crop JPEGs."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.verification.queue import detection_key, nsew_key_aliases
from src.main_pipeline.measurement.measurer import (
    is_nsew_family_tile,
    is_submap_lighting_tile,
)
from src.main_pipeline.report.report_generator import encode_overlay_jpeg

LABEL_VALUES = ("particle", "not_sure", "not_particle")
CAMPAIGNS = ("groundup", "glovebox")

# R1, R2, … are glovebox tiles. R0_1_1 lives in Groundup v1.
# n2/n3/bf shots are the R4–R7 glovebox folders; their names do not start with R.
_GLOVEBOX_TILE = re.compile(r"^(?:R[1-9]|n2_|n3_|bf_)", re.IGNORECASE)

COLUMNS = (
    "key",
    "particle_id",
    "source_tile",
    "x_global",
    "y_global",
    "size",
    "confidence",
    "label",
    "crop_path",
    "source_csv",
    "labeled_at",
)


class LabelStore:
    """CSV + JPEG collections under ``root`` (default ``labels/``)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.csv_path = self.root / "labels.csv"
        self.crops_dir = self.root / "crops"
        for label in LABEL_VALUES:
            (self.crops_dir / label).mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return _empty_table()
        df = pd.read_csv(self.csv_path)
        for column in COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        return df.loc[:, list(COLUMNS)]

    def labeled_keys(self) -> set[str]:
        df = self.load()
        if df.empty:
            return set()
        return {str(key) for key in df["key"].tolist()}

    def counts(self) -> dict[str, int]:
        df = self.load()
        counts = {label: 0 for label in LABEL_VALUES}
        if df.empty:
            return counts
        for label, group in df.groupby("label"):
            name = str(label)
            if name in counts:
                counts[name] = int(len(group))
        return counts

    def by_label(self, label: str) -> pd.DataFrame:
        if label not in LABEL_VALUES:
            raise ValueError(f"Unknown label {label!r}")
        df = self.load()
        if df.empty:
            return df
        return newest_first(df.loc[df["label"] == label])

    def apply_label(
        self,
        record: Mapping[str, Any],
        label: str,
        crop: np.ndarray,
    ) -> Path:
        """Write the crop JPEG and append/replace the CSV row immediately."""
        return self.apply_labels([(record, label, crop)])[0]

    def apply_labels(
        self,
        items: list[tuple[Mapping[str, Any], str, np.ndarray]],
    ) -> list[Path]:
        """Write many crops and replace their CSV rows in one save."""
        if not items:
            return []
        prepared: dict[str, tuple[Mapping[str, Any], str, np.ndarray]] = {}
        order: list[str] = []
        for record, label, crop in items:
            if label not in LABEL_VALUES:
                raise ValueError(f"Unknown label {label!r}")
            key = _label_key(record)
            if key not in prepared:
                order.append(key)
            prepared[key] = (record, label, crop)

        drop_keys: set[str] = set()
        for key in order:
            drop_keys.update(nsew_key_aliases(key))

        df = self.load()
        if not df.empty:
            mask = df["key"].astype(str).isin(drop_keys)
            for crop_path in df.loc[mask, "crop_path"].tolist():
                if crop_path in (None, ""):
                    continue
                path = Path(str(crop_path))
                if path.is_file():
                    path.unlink()
            df = df.loc[~mask].reset_index(drop=True)

        stamped = _now()
        rows: list[dict[str, Any]] = []
        dests: list[Path] = []
        for key in order:
            record, label, crop = prepared[key]
            dest = self.crops_dir / label / f"{key}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(encode_overlay_jpeg(np.asarray(crop)))
            rows.append(_label_row(record, key, label, dest, stamped))
            dests.append(dest)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        self._write(df)
        return dests

    def update_sizes(self, sizes_nm: Mapping[str, float]) -> int:
        """Write photo diameters onto existing rows. Leaves the label and crop."""
        if not sizes_nm:
            return 0
        df = self.load()
        if df.empty or "key" not in df.columns:
            return 0
        updated = 0
        keys = df["key"].astype(str)
        for key, size_nm in sizes_nm.items():
            mask = keys == str(key)
            if not mask.any():
                continue
            idx = df.index[mask][-1]
            df.loc[idx, "size"] = float(size_nm)
            updated += 1
        if updated:
            self._write(df)
        return updated

    def unlabel(self, key: str) -> dict[str, Any] | None:
        """Remove a labeled record and its crop. Returns the dropped row if any."""
        df = self.load()
        if df.empty:
            return None
        mask = df["key"].astype(str).isin(nsew_key_aliases(key))
        if not mask.any():
            return None
        dropped = df.loc[mask].iloc[-1].to_dict()
        for crop_path in df.loc[mask, "crop_path"].tolist():
            if crop_path in (None, ""):
                continue
            path = Path(str(crop_path))
            if path.is_file():
                path.unlink()
        self._write(df.loc[~mask].reset_index(drop=True))
        return {str(k): v for k, v in dropped.items()}

    def undo_last(self) -> dict[str, Any] | None:
        """Remove the most recently labeled row and return it to the queue."""
        df = self.load()
        if df.empty:
            return None
        times = df["labeled_at"].astype(str)
        idx = times.idxmax()
        key = str(df.loc[idx, "key"])
        return self.unlabel(key)

    def _write(self, df: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if df.empty:
            _empty_table().to_csv(self.csv_path, index=False)
            return
        df.loc[:, list(COLUMNS)].to_csv(self.csv_path, index=False)


def _label_key(record: Mapping[str, Any]) -> str:
    key = str(record.get("key") or detection_key(record))
    tile = str(record.get("source_tile", "") or "")
    if is_nsew_family_tile(tile) or is_submap_lighting_tile(tile):
        return detection_key(record)
    return key


def _label_row(
    record: Mapping[str, Any],
    key: str,
    label: str,
    dest: Path,
    stamped: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "particle_id": record.get("id", record.get("particle_id", "")),
        "source_tile": record.get("source_tile", ""),
        "x_global": record.get("x_global", ""),
        "y_global": record.get("y_global", ""),
        "size": record.get("size", ""),
        "confidence": record.get("confidence", ""),
        "label": label,
        "crop_path": str(dest),
        "source_csv": record.get("source_csv", ""),
        "labeled_at": stamped,
    }


def label_campaign(source_tile: object) -> str:
    """``glovebox`` or ``groundup`` from the photo file name."""
    name = Path(str(source_tile or "")).name
    if _GLOVEBOX_TILE.match(name):
        return "glovebox"
    return "groundup"


def rows_for_campaign(df: pd.DataFrame, campaign: str) -> pd.DataFrame:
    """Keep rows whose source photo belongs to ``groundup`` or ``glovebox``."""
    if campaign not in CAMPAIGNS:
        raise ValueError(f"Unknown campaign {campaign!r}")
    if df is None or df.empty or "source_tile" not in df.columns:
        return df
    mask = df["source_tile"].map(label_campaign) == campaign
    return df.loc[mask].reset_index(drop=True)


def newest_first(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with the latest ``labeled_at`` first."""
    if df is None or df.empty:
        return df
    if "labeled_at" not in df.columns:
        return df.iloc[::-1].reset_index(drop=True)
    times = pd.to_datetime(df["labeled_at"], utc=True, errors="coerce")
    ordered = df.assign(_labeled_at=times).sort_values(
        "_labeled_at", ascending=False, na_position="last", kind="mergesort"
    )
    return ordered.drop(columns="_labeled_at").reset_index(drop=True)


def crop_source_folder(source_csv: object, source_tile: object = "") -> str:
    """Run folder for a labeled crop.

    Detector rows use the output folder that wrote ``particles.csv``.
    Pointer marks (``tile_inspect``) use the image set in ``source_tile``,
    such as ``v3/N.png`` → ``v3``.
    """
    text = str(source_csv or "").replace("\\", "/").strip()
    if text.lower() in ("", "nan", "none", "tile_inspect"):
        tile = str(source_tile or "").replace("\\", "/")
        parent = tile.rsplit("/", 1)[0] if "/" in tile else ""
        if parent and parent.lower() not in ("nan", "."):
            return parent
        return "tile_inspect" if text == "tile_inspect" else ""
    path = Path(text)
    if path.suffix.lower() == ".csv":
        parent = path.parent
        if parent.is_absolute():
            return parent.name
        folder = str(parent)
        return "" if folder in ("", ".") else folder
    return path.name


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in COLUMNS})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
