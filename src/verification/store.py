from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.verification.queue import detection_key, nsew_key_aliases
from src.overall.io.naming import (
    is_nsew_family_tile,
    is_submap_lighting_tile,
)
from src.analyse.report.report_generator import encode_overlay_jpeg

LABEL_VALUES = ("particle", "not_sure", "not_particle")
ALL_LABELS_NAME = "all_labels.csv"
_SPLIT_LABEL_FILES = (
    "labels_uniform.csv",
    "labels_nsew.csv",
    "labels_glovebox.csv",
    "labels_groundup.csv",
)

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
    

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.csv_path = self.root / ALL_LABELS_NAME
        self.crops_dir = self.root / "crops"
        for label in LABEL_VALUES:
            (self.crops_dir / label).mkdir(parents=True, exist_ok=True)
        self._frame: pd.DataFrame | None = None
        self._frame_stamp: tuple[str, int, int] | None = None

    def load(self) -> pd.DataFrame:
        path = self.csv_path
        if not path.exists():
            return _empty_table()
        try:
            stat = path.stat()
            stamp = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            stamp = None
        if stamp is not None and stamp == self._frame_stamp and self._frame is not None:
            return self._frame.copy()
        df = pd.read_csv(path)
        for column in COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        frame = df.loc[:, list(COLUMNS)]
        self._frame = frame
        self._frame_stamp = stamp
        return frame.copy()

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
        
        return self.apply_labels([(record, label, crop)])[0]

    def apply_labels(
        self,
        items: list[tuple[Mapping[str, Any], str, np.ndarray]],
    ) -> list[Path]:
        
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
                path = resolve_crop_path(self.root, crop_path)
                if path is not None and path.is_file():
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

    def unlabel(self, key: str) -> dict[str, Any] | None:
        
        df = self.load()
        if df.empty:
            return None
        mask = df["key"].astype(str).isin(nsew_key_aliases(key))
        if not mask.any():
            return None
        dropped = df.loc[mask].iloc[-1].to_dict()
        for crop_path in df.loc[mask, "crop_path"].tolist():
            path = resolve_crop_path(self.root, crop_path)
            if path is not None and path.is_file():
                path.unlink()
        self._write(df.loc[~mask].reset_index(drop=True))
        return {str(k): v for k, v in dropped.items()}

    def undo_last(self) -> dict[str, Any] | None:
        
        df = self.load()
        if df.empty:
            return None
        times = df["labeled_at"].astype(str)
        idx = times.idxmax()
        key = str(df.loc[idx, "key"])
        return self.unlabel(key)

    def _write(self, df: pd.DataFrame) -> None:
        
        self.root.mkdir(parents=True, exist_ok=True)
        table = _empty_table() if df.empty else df.loc[:, list(COLUMNS)]
        table.to_csv(self.csv_path, index=False)
        self._frame = table
        try:
            stat = self.csv_path.stat()
            self._frame_stamp = (
                str(self.csv_path.resolve()),
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        except OSError:
            self._frame_stamp = None
        _drop_split_label_files(self.root)


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


def _drop_split_label_files(root: Path) -> None:
    for name in _SPLIT_LABEL_FILES:
        path = root / name
        if path.is_file():
            path.unlink()


def newest_first(df: pd.DataFrame) -> pd.DataFrame:
    
    if df is None or df.empty:
        return df
    if "labeled_at" not in df.columns:
        return df.iloc[::-1].reset_index(drop=True)
    times = pd.to_datetime(df["labeled_at"], utc=True, errors="coerce")
    ordered = df.assign(_labeled_at=times).sort_values(
        "_labeled_at", ascending=False, na_position="last", kind="mergesort"
    )
    return ordered.drop(columns="_labeled_at").reset_index(drop=True)


def resolve_crop_path(
    labels_root: str | Path,
    crop_path: object,
    key: object = "",
    label: object = "",
) -> Path | None:
    root = Path(labels_root)
    crops = root / "crops"
    text = _path_text(crop_path)
    stored = Path(text) if text else None
    if stored is not None and stored.is_file():
        return stored

    names: list[str] = []
    if stored is not None and stored.name not in ("", ".", ".."):
        names.append(stored.name)
    key_text = _path_text(key)
    if key_text:
        filename = f"{key_text}.jpg"
        if filename not in names:
            names.append(filename)

    folders: list[str] = []
    if stored is not None:
        parent = stored.parent.name
        if parent not in ("", ".", "..", "crops", "labels"):
            folders.append(parent)
    label_text = _path_text(label)
    if label_text and label_text not in folders:
        folders.append(label_text)
    for folder_name in LABEL_VALUES:
        if folder_name not in folders:
            folders.append(folder_name)

    for folder_name in folders:
        for name in names:
            candidate = crops / folder_name / name
            if candidate.is_file():
                return candidate
    if stored is not None and not stored.is_absolute():
        for base in (root.parent, root):
            candidate = base / stored
            if candidate.is_file():
                return candidate
    return None


def _path_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in ("", "nan", "none"):
        return ""
    return text


def crop_source_folder(source_csv: object, source_tile: object = "") -> str:
    
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
