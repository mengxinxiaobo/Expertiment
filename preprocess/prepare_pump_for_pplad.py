#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def fill_split(frame: pd.DataFrame, fallback: pd.Series) -> pd.DataFrame:
    # Each split is imputed independently to avoid carrying test values into train.
    result = frame.interpolate(method="linear", axis=0, limit_direction="both")
    result = result.ffill().bfill()
    result = result.fillna(fallback)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle pump sensor CSV to PPLAD-compatible NPY files."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to sensor.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/PUMP"))
    parser.add_argument(
        "--label-mode",
        choices=["non-normal", "broken-only"],
        default="non-normal",
        help=(
            "non-normal: BROKEN and RECOVERING are anomalies; "
            "broken-only: only BROKEN is anomalous."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["initial-normal-prefix"],
        default="initial-normal-prefix",
        help="Use the initial continuous NORMAL prefix as unsupervised training data.",
    )
    parser.add_argument("--min-train-rows", type=int, default=1000)
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        raise ValueError("CSV is empty.")

    original_columns = [str(c) for c in df.columns]
    status_col = detect_column(original_columns, ("machine_status", "status"))
    time_col = detect_column(original_columns, ("timestamp", "time", "datetime", "date"))

    if status_col is None:
        raise ValueError(
            "Could not find machine_status/status column. "
            f"Columns are: {original_columns[:15]}..."
        )

    if time_col is not None:
        parsed_time = pd.to_datetime(df[time_col], errors="coerce")
        if parsed_time.notna().any():
            df = df.assign(__parsed_time=parsed_time).sort_values(
                "__parsed_time", kind="stable"
            ).drop(columns="__parsed_time").reset_index(drop=True)

    status = (
        df[status_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": "UNKNOWN", "": "UNKNOWN"})
    )

    status_counts = status.value_counts(dropna=False).to_dict()
    if args.label_mode == "non-normal":
        labels_all = (status != "NORMAL").astype(np.int64).to_numpy()
    else:
        labels_all = (status == "BROKEN").astype(np.int64).to_numpy()

    # Prefer the documented sensor_00 ... sensor_51 columns.
    sensor_cols = [
        str(c)
        for c in df.columns
        if re.fullmatch(r"sensor[_\s-]*\d+", str(c).strip(), flags=re.IGNORECASE)
    ]

    if not sensor_cols:
        excluded = {status_col}
        if time_col is not None:
            excluded.add(time_col)
        excluded.update(
            c for c in df.columns
            if str(c).strip().lower().startswith("unnamed")
            or str(c).strip().lower() in {"index", "id"}
        )
        sensor_cols = [str(c) for c in df.columns if c not in excluded]

    features = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    if features.shape[1] == 0:
        raise ValueError("No usable sensor columns were found.")

    anomaly_indices = np.flatnonzero(labels_all == 1)
    if anomaly_indices.size == 0:
        raise ValueError(
            f"No anomalies found with label mode {args.label_mode!r}; "
            "check machine_status values."
        )

    train_end = int(anomaly_indices[0])
    if train_end < args.min_train_rows:
        raise ValueError(
            f"Initial normal prefix has only {train_end} rows, below "
            f"--min-train-rows={args.min_train_rows}."
        )
    if train_end >= len(df):
        raise ValueError("Test split would be empty.")

    train_raw = features.iloc[:train_end].copy()
    test_raw = features.iloc[train_end:].copy()
    test_labels = labels_all[train_end:]

    all_nan_cols = [c for c in sensor_cols if train_raw[c].isna().all()]
    non_all_nan = [c for c in sensor_cols if c not in all_nan_cols]

    # Remove sensors that contain no variation in normal training data.
    train_nunique = train_raw[non_all_nan].nunique(dropna=True)
    constant_cols = train_nunique[train_nunique <= 1].index.tolist()
    kept_cols = [c for c in non_all_nan if c not in constant_cols]

    if not kept_cols:
        raise ValueError("All sensor columns were all-NaN or constant in train data.")

    train_raw = train_raw[kept_cols]
    test_raw = test_raw[kept_cols]

    train_median = train_raw.median(axis=0, skipna=True)
    train_filled = fill_split(train_raw, train_median)
    test_filled = fill_split(test_raw, train_median)

    train = train_filled.to_numpy(dtype=np.float32)
    test = test_filled.to_numpy(dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    if not np.isfinite(train).all():
        raise ValueError("Train data still contains NaN/Inf after imputation.")
    if not np.isfinite(test).all():
        raise ValueError("Test data still contains NaN/Inf after imputation.")
    if test.shape[0] != test_labels.shape[0]:
        raise AssertionError("Test and label lengths differ.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "PUMP_train.npy"
    test_path = args.output_dir / "PUMP_test.npy"
    label_path = args.output_dir / "PUMP_test_label.npy"
    columns_path = args.output_dir / "PUMP_columns.json"
    metadata_path = args.output_dir / "PUMP_preprocess_metadata.json"

    np.save(train_path, train)
    np.save(test_path, test)
    np.save(label_path, test_labels)
    columns_path.write_text(
        json.dumps(kept_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "source_csv": str(csv_path),
        "source_sha256": sha256(csv_path),
        "raw_shape": list(df.shape),
        "status_column": status_col,
        "time_column": time_col,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "label_mode": args.label_mode,
        "split_protocol": args.split,
        "train_end_index_exclusive": train_end,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "label_shape": list(test_labels.shape),
        "test_anomaly_points": int(test_labels.sum()),
        "test_anomaly_ratio_percent": float(test_labels.mean() * 100.0),
        "original_sensor_count": len(sensor_cols),
        "kept_sensor_count": len(kept_cols),
        "dropped_all_nan_columns": all_nan_cols,
        "dropped_constant_columns": constant_cols,
        "kept_columns": kept_cols,
        "output_sha256": {
            train_path.name: sha256(train_path),
            test_path.name: sha256(test_path),
            label_path.name: sha256(label_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("PUMP preprocessing completed.")
    print(f"raw shape            : {tuple(df.shape)}")
    print(f"status counts        : {status_counts}")
    print(f"label mode           : {args.label_mode}")
    print(f"train normal prefix  : [0, {train_end})")
    print(f"train shape          : {train.shape}")
    print(f"test shape           : {test.shape}")
    print(f"label shape          : {test_labels.shape}")
    print(f"test anomaly ratio   : {test_labels.mean() * 100.0:.4f}%")
    print(f"dropped all-NaN      : {all_nan_cols}")
    print(f"dropped constant     : {constant_cols}")
    print(f"output directory     : {args.output_dir}")


if __name__ == "__main__":
    main()
