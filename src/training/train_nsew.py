"""Fit a residual ExtraTrees on NSEW combi labels only.

Uniform Groundup crops stay in particle_clf_v4_residual. This model uses the
same detection-time columns, but the rows are labels drawn on 2-of-4 combi
images, including the false hits marked not_particle.

Train: v3_2of4 and v5_2of4.
Threshold: chosen on v6_2of4, which is not in the fit.
N/S/E/W direction tiles and the N40 uniform circles are left out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.main_pipeline.ml.features import FEATURE_COLUMNS, dataframe_feature_matrix
from src.main_pipeline.ml.infer import KIND_RESIDUAL, load_artifact
from src.training.train import (
    _load_feature_tables,
    _metrics_at_threshold,
    _oof_proba,
    _positive_proba_matrix,
    _trees,
    choose_f1_threshold,
    choose_threshold,
    keep_all_threshold,
    save_artifact,
)

LABELS = ROOT / "labels" / "all_labels.csv"
MANIFEST = ROOT.parent / "Outputs" / "nsew_v1_train_keys.csv"
DETECTIONS = (
    ROOT.parent / "Outputs" / "Groundup v5 v3 v4" / "particles.csv",
    ROOT.parent / "Outputs" / "Groundup v5 v5 v6" / "particles.csv",
)
MODEL = ROOT / "models" / "particle_clf_v8.joblib"
V4_MODEL = ROOT / "models" / "particle_clf_v4_residual.joblib"
KEYS_OUT = ROOT.parent / "Outputs" / "nsew_v8_train_keys.csv"
COMBI_SUFFIX = "2of4.png"


def combi_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    """2-of-4 combi rows from the NSEW key list. Direction tiles are dropped."""
    table = pd.read_csv(path)
    table["key"] = table["key"].astype(str)
    tile = table["source_tile"].astype(str)
    keep = tile.str.endswith(COMBI_SUFFIX)
    out = table.loc[keep].copy()
    if out.empty:
        raise ValueError(f"No {COMBI_SUFFIX} rows in {path}")
    return out.reset_index(drop=True)


def load_combi_features(
    manifest: pd.DataFrame,
    labels_path: Path = LABELS,
    detection_paths: tuple[Path, ...] = DETECTIONS,
) -> pd.DataFrame:
    """Join combi labels to residual feature columns. Uniform rows never enter."""
    labels = pd.read_csv(labels_path)
    labels["key"] = labels["key"].astype(str)
    labeled = labels["label"].astype(str).isin(("particle", "not_particle"))
    keys = set(manifest["key"].astype(str))
    keep = labels.loc[labeled & labels["key"].isin(keys)].copy()
    keep = keep.drop_duplicates(subset=["key"], keep="last")
    features = _load_feature_tables(list(detection_paths))
    merged = keep.merge(features, on="key", how="inner", suffixes=("", "_det"))
    split = manifest.loc[:, ["key", "split", "set_id", "source_tile"]].drop_duplicates("key")
    merged = merged.drop(columns=["source_tile"], errors="ignore")
    merged = merged.merge(split, on="key", how="left")
    if len(merged) != len(keep):
        missing = len(keep) - len(merged)
        raise ValueError(f"{missing} combi labels have no residual features")
    needed = ["size", "confidence", *FEATURE_COLUMNS[2:]]
    empty = [name for name in needed if merged[name].isna().any()]
    if empty:
        raise ValueError(f"Combi feature columns have missing values: {empty}")
    return merged.reset_index(drop=True)


def _holdout_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = _metrics_at_threshold(y, proba, threshold)
    pred = np.asarray(proba) >= float(threshold)
    labels = np.asarray(y) == 1
    metrics["false_hits_removed"] = float(((~labels) & ~pred).sum())
    metrics["false_hits"] = float((~labels).sum())
    metrics["true_hits_lost"] = float((labels & ~pred).sum())
    metrics["true_hits"] = float(labels.sum())
    return metrics


def train_nsew(table: pd.DataFrame) -> dict:
    """Fit on the train split. The shipped threshold is the v6 F1 cutoff."""
    train = table["split"].astype(str) == "train"
    hold = table["split"].astype(str) == "holdout_tile"
    if not train.any() or not hold.any():
        raise ValueError("Need a train split and a holdout_tile split")
    y = (table["label"].astype(str) == "particle").astype(int).to_numpy()
    x = dataframe_feature_matrix(table)
    groups = table["source_tile"].astype(str).to_numpy()
    y_train, x_train, g_train = y[train.to_numpy()], x[train.to_numpy()], groups[train.to_numpy()]
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos < 2 or n_neg < 2:
        raise ValueError(f"Need both classes on the combi train split (got {n_pos} / {n_neg})")
    model = _trees()
    oof = _oof_proba(model, x_train, y_train, g_train)
    model.fit(x_train, y_train)
    hold_y = y[hold.to_numpy()]
    hold_p = _positive_proba_matrix(model, x[hold.to_numpy()])
    threshold = choose_f1_threshold(hold_y, hold_p)
    metrics = _holdout_metrics(hold_y, hold_p, threshold)
    unhappy_t, unhappy = choose_threshold(hold_y, hold_p)
    importances = {
        name: float(value)
        for name, value in zip(FEATURE_COLUMNS, model.feature_importances_)
    }
    return {
        "kind": KIND_RESIDUAL,
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "threshold": float(threshold),
        "threshold_f1": float(threshold),
        "threshold_unhappiness": float(unhappy_t),
        "threshold_keep_all": float(keep_all_threshold(hold_y, hold_p)),
        "metrics": metrics,
        "holdout_unhappiness_metrics": unhappy,
        "train_oof_at_threshold": _holdout_metrics(y_train, oof, threshold),
        "n_samples": int(train.sum()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_holdout": int(hold.sum()),
        "feature_importances": importances,
        "train_tiles": "v3_2of4 + v5_2of4",
        "held_out": "v6_2of4",
        "dataset": (
            "Labels drawn on Groundup 2-of-4 combi images, including not_particle "
            "false hits. Uniform crops and N40 dies are not in the fit."
        ),
    }


def _print_report(artifact: dict, table: pd.DataFrame) -> None:
    metrics = artifact["metrics"]
    print(
        f"train n={artifact['n_samples']} "
        f"(particle {artifact['n_positive']} / not {artifact['n_negative']})  "
        f"holdout n={artifact['n_holdout']}",
        flush=True,
    )
    print(
        f"v6 F1 threshold={artifact['threshold']:.2f}  "
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} U={metrics['unhappiness']:.2f}",
        flush=True,
    )
    print(
        f"  v6 false hits removed {int(metrics['false_hits_removed'])}/"
        f"{int(metrics['false_hits'])}  "
        f"true hits lost {int(metrics['true_hits_lost'])}/{int(metrics['true_hits'])}",
        flush=True,
    )
    print(
        f"  unhappiness cutoff={artifact['threshold_unhappiness']:.2f}  "
        f"keep_all={artifact['threshold_keep_all']:.3f}",
        flush=True,
    )
    oof = artifact["train_oof_at_threshold"]
    print(
        f"  train OOF at that cutoff P={oof['precision']:.3f} "
        f"R={oof['recall']:.3f} F1={oof['f1']:.3f}",
        flush=True,
    )
    v4 = load_artifact(V4_MODEL)
    from src.main_pipeline.ml.infer import _positive_proba

    hold = table["split"].astype(str) == "holdout_tile"
    y = (table["label"].astype(str) == "particle").astype(int).to_numpy()
    x = dataframe_feature_matrix(table)
    v4_p = _positive_proba(v4["model"], x[hold.to_numpy()])
    v4_metrics = _holdout_metrics(y[hold.to_numpy()], v4_p, 0.23)
    print(
        f"  v4 @ 0.23 on the same v6 combi  "
        f"P={v4_metrics['precision']:.3f} R={v4_metrics['recall']:.3f} "
        f"F1={v4_metrics['f1']:.3f}  "
        f"removed {int(v4_metrics['false_hits_removed'])}/{int(v4_metrics['false_hits'])} "
        f"false, lost {int(v4_metrics['true_hits_lost'])}/{int(v4_metrics['true_hits'])} true",
        flush=True,
    )
    ranked = sorted(artifact["feature_importances"].items(), key=lambda item: item[1], reverse=True)
    print("feature importances:", flush=True)
    for name, value in ranked:
        print(f"  {name:22s} {value:.4f}", flush=True)


def main() -> int:
    v4_before = V4_MODEL.stat().st_mtime_ns
    manifest = combi_manifest()
    table = load_combi_features(manifest)
    counts = (
        table.groupby(["set_id", "split"])["label"]
        .value_counts()
        .unstack(fill_value=0)
    )
    print(counts.to_string(), flush=True)
    artifact = train_nsew(table)
    saved = save_artifact(artifact, MODEL)
    manifest.loc[manifest["key"].isin(set(table["key"].astype(str)))].to_csv(KEYS_OUT, index=False)
    print(f"Wrote {saved}", flush=True)
    print(f"Wrote {KEYS_OUT.name} ({len(table)} keys).", flush=True)
    _print_report(artifact, table)
    if V4_MODEL.stat().st_mtime_ns != v4_before:
        raise SystemExit("particle_clf_v4_residual.joblib was modified")
    print(f"Left {V4_MODEL.name} unchanged.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
