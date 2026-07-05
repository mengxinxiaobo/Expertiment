#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_FILES = [f"hai-train{i}.csv" for i in range(1, 5)]
TEST_FILES = [f"hai-test{i}.csv" for i in range(1, 3)]
LABEL_FILES = [f"label-test{i}.csv" for i in range(1, 3)]

TIME_NAMES = {
    "time", "timestamp", "datetime", "date", "observed_time", "event_time"
}
LABEL_NAMES = {
    "attack", "label", "anomaly", "is_attack", "attack_label", "y"
}
INDEX_NAMES = {"index", "id", "row", "row_id"}


def norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_real_csv(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        prefix = f.read(200)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the real CSV. "
            "Run `git lfs install && git lfs pull` in the HAI repository, "
            "or download the actual release files."
        )
    if path.stat().st_size < 1024:
        raise RuntimeError(
            f"{path} is unexpectedly small ({path.stat().st_size} bytes). "
            "It may not be the actual dataset file."
        )


def detect_time_column(columns: list[str]) -> str:
    for c in columns:
        if norm(c) in TIME_NAMES:
            return c
    # The HAI documentation defines the first column as observed time.
    return columns[0]


def is_auxiliary_column(name: str) -> bool:
    n = norm(name)
    return (
        n in TIME_NAMES
        or n in LABEL_NAMES
        or n in INDEX_NAMES
        or n.startswith("unnamed")
        or n.startswith("attack")
    )


def feature_columns(columns: list[str], time_col: str) -> list[str]:
    return [
        c for c in columns
        if c != time_col and not is_auxiliary_column(c)
    ]


def detect_label_column(frame: pd.DataFrame, time_col: str) -> str:
    normalized = {norm(c): c for c in frame.columns}
    for candidate in ("attack", "label", "anomaly", "is_attack", "attack_label"):
        if candidate in normalized:
            return normalized[candidate]

    binary_candidates: list[str] = []
    for c in frame.columns:
        if c == time_col or norm(c).startswith("unnamed"):
            continue
        numeric = pd.to_numeric(frame[c], errors="coerce").dropna()
        values = set(numeric.unique().tolist())
        if values and values.issubset({0, 1, 0.0, 1.0}):
            binary_candidates.append(c)

    if len(binary_candidates) == 1:
        return binary_candidates[0]
    raise ValueError(
        "Could not uniquely identify the attack label column. "
        f"Columns: {list(frame.columns)}; binary candidates: {binary_candidates}"
    )


def to_float32(frame: pd.DataFrame, columns: list[str], source: Path) -> np.ndarray:
    raw = frame[columns]
    numeric = raw.apply(pd.to_numeric, errors="coerce")

    bad_total = 0
    for c in columns:
        raw_nonempty = raw[c].notna() & raw[c].astype(str).str.strip().ne("")
        bad = raw_nonempty & numeric[c].isna()
        bad_total += int(bad.sum())
    if bad_total:
        raise ValueError(
            f"{source} contains {bad_total} non-empty non-numeric feature values."
        )

    array = numeric.to_numpy(dtype=np.float32, copy=True)
    if np.isinf(array).any():
        raise ValueError(f"{source} contains +Inf/-Inf feature values.")
    return array


def read_data_file(
    path: Path,
    expected_features: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"{path} is empty.")

    columns = [str(c) for c in frame.columns]
    frame.columns = columns
    time_col = detect_time_column(columns)
    features = feature_columns(columns, time_col)

    if expected_features is not None and features != expected_features:
        missing = [c for c in expected_features if c not in features]
        extra = [c for c in features if c not in expected_features]
        raise ValueError(
            f"Feature schema mismatch in {path}. "
            f"Missing={missing}, extra={extra}, "
            f"order_matches={features == expected_features}"
        )

    times = frame[time_col].astype(str).str.strip().to_numpy()
    parsed = pd.to_datetime(frame[time_col], errors="coerce")
    monotonic = bool(parsed.is_monotonic_increasing) if parsed.notna().all() else None

    array = to_float32(frame, features, path)
    info = {
        "file": path.name,
        "rows": int(len(frame)),
        "columns": int(len(columns)),
        "time_column": time_col,
        "feature_count": int(len(features)),
        "nan_count": int(np.isnan(array).sum()),
        "time_parse_success": bool(parsed.notna().all()),
        "time_monotonic_non_decreasing": monotonic,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }
    return array, times, features, info


def read_label_file(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"{path} is empty.")

    columns = [str(c) for c in frame.columns]
    frame.columns = columns
    time_col = detect_time_column(columns)
    label_col = detect_label_column(frame, time_col)

    numeric = pd.to_numeric(frame[label_col], errors="coerce")
    if numeric.isna().any():
        raise ValueError(
            f"{path}: label column {label_col!r} has "
            f"{int(numeric.isna().sum())} non-numeric/missing values."
        )
    unique = sorted(set(numeric.unique().tolist()))
    if not set(unique).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"{path}: attack labels are not binary: {unique}")

    labels = (numeric.to_numpy() > 0).astype(np.int64)
    times = frame[time_col].astype(str).str.strip().to_numpy()
    info = {
        "file": path.name,
        "rows": int(len(frame)),
        "columns": int(len(columns)),
        "time_column": time_col,
        "label_column": label_col,
        "attack_points": int(labels.sum()),
        "attack_ratio_percent": float(labels.mean() * 100.0),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }
    return labels, times, info



def validate_timestamp_alignment(
    test_times: np.ndarray,
    label_times: np.ndarray,
    test_name: str,
    label_name: str,
) -> dict:
    """
    Validate HAI test/label timestamp alignment without changing row order.

    HAI 23.05 label files may encode timestamps with lower precision than
    the second-level test files. The validator accepts exact equality, a safe
    constant offset of at most one second, or exact minute-bucket equality
    where every label timestamp equals floor(test timestamp to minute).
    In every accepted case, official row order is retained and labels are
    neither shifted nor modified.
    """
    if len(test_times) != len(label_times):
        raise ValueError(
            f"{test_name}/{label_name}: timestamp lengths differ: "
            f"{len(test_times)} vs {len(label_times)}"
        )

    if np.array_equal(test_times, label_times):
        return {
            "status": "exact",
            "constant_offset_seconds": 0.0,
            "row_count": int(len(test_times)),
        }

    test_parsed = pd.to_datetime(
        pd.Series(test_times, dtype="string"), errors="coerce"
    )
    label_parsed = pd.to_datetime(
        pd.Series(label_times, dtype="string"), errors="coerce"
    )

    if test_parsed.isna().any() or label_parsed.isna().any():
        mismatch = np.flatnonzero(test_times != label_times)
        first = int(mismatch[0]) if mismatch.size else -1
        raise ValueError(
            f"Unparseable timestamp mismatch between {test_name} and "
            f"{label_name}; first mismatch row={first}, "
            f"test={test_times[first]!r}, label={label_times[first]!r}"
        )

    test_ns = test_parsed.astype("int64").to_numpy()
    label_ns = label_parsed.astype("int64").to_numpy()

    if np.array_equal(test_ns, label_ns):
        return {
            "status": "format-only-difference",
            "constant_offset_seconds": 0.0,
            "row_count": int(len(test_times)),
        }

    deltas = test_ns - label_ns
    unique_deltas = np.unique(deltas)

    same_progression = (
        len(test_ns) <= 1
        or np.array_equal(np.diff(test_ns), np.diff(label_ns))
    )

    if unique_deltas.size == 1 and same_progression:
        offset_seconds = float(unique_deltas[0] / 1_000_000_000.0)
        if abs(offset_seconds) <= 1.0:
            return {
                "status": "constant-rowwise-offset",
                "constant_offset_seconds": offset_seconds,
                "row_count": int(len(test_times)),
                "first_test_timestamp": str(test_times[0]),
                "first_label_timestamp": str(label_times[0]),
                "last_test_timestamp": str(test_times[-1]),
                "last_label_timestamp": str(label_times[-1]),
                "alignment_policy": (
                    "official file row order; no row shifting and no label "
                    "modification"
                ),
            }

    # HAI 23.05 label-test2.csv stores timestamps at minute resolution:
    # every second-level test row within a minute repeats that minute's label
    # timestamp. Accept this only when it is true for every row.
    test_minute_ns = (
        test_parsed.dt.floor("min").astype("int64").to_numpy()
    )
    minute_bucket_match = np.array_equal(test_minute_ns, label_ns)

    if minute_bucket_match:
        test_diffs = np.diff(test_ns)
        if len(test_diffs) and np.any(test_diffs <= 0):
            raise ValueError(
                f"{test_name}: test timestamps are not strictly increasing."
            )

        label_diffs = np.diff(label_ns)
        allowed_label_steps = np.isin(
            label_diffs,
            np.array([0, 60_000_000_000], dtype=np.int64),
        )
        if len(label_diffs) and not bool(np.all(allowed_label_steps)):
            bad = np.flatnonzero(~allowed_label_steps)
            first_bad = int(bad[0])
            raise ValueError(
                f"{label_name}: minute-resolution timestamps contain an "
                f"unexpected step at row {first_bad}: "
                f"{label_diffs[first_bad]} ns"
            )

        offsets_seconds = (test_ns - label_ns) / 1_000_000_000.0
        return {
            "status": "label-minute-resolution-rowwise",
            "constant_offset_seconds": None,
            "minimum_offset_seconds": float(offsets_seconds.min()),
            "maximum_offset_seconds": float(offsets_seconds.max()),
            "row_count": int(len(test_times)),
            "first_test_timestamp": str(test_times[0]),
            "first_label_timestamp": str(label_times[0]),
            "last_test_timestamp": str(test_times[-1]),
            "last_label_timestamp": str(label_times[-1]),
            "alignment_policy": (
                "label timestamp equals floor(test timestamp to minute) for "
                "every row; official file row order retained; no row shifting "
                "and no label modification"
            ),
        }

    mismatch = np.flatnonzero(test_ns != label_ns)
    first = int(mismatch[0]) if mismatch.size else -1
    delta_preview = np.unique(deltas[: min(len(deltas), 10000)])
    raise ValueError(
        f"Unsafe timestamp mismatch between {test_name} and {label_name}; "
        f"first mismatch row={first}, "
        f"test={test_times[first]!r}, label={label_times[first]!r}, "
        f"unique_delta_ns_preview={delta_preview[:10].tolist()}, "
        f"same_progression={same_progression}. "
        "The converter refuses to guess label alignment."
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the official HAI 23.05 CSV split to PPLAD HAI_*.npy files."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("dataset/HAI"),
        help="Directory containing hai-train1..4, hai-test1..2 and label-test1..2."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dataset/HAI")
    )
    parser.add_argument("--expected-channels", type=int, default=86)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    paths = [input_dir / f for f in TRAIN_FILES + TEST_FILES + LABEL_FILES]
    for path in paths:
        assert_real_csv(path)

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    metadata: dict = {
        "dataset": "HAI 23.05",
        "protocol": (
            "official HAI train/test file split; train1-4 concatenated in numeric "
            "order, test1-2 and label-test1-2 concatenated in matching order"
        ),
        "input_dir": str(input_dir),
        "train_files": [],
        "test_files": [],
        "label_files": [],
    }

    features: list[str] | None = None

    for filename in TRAIN_FILES:
        array, _, current_features, info = read_data_file(
            input_dir / filename, features
        )
        if features is None:
            features = current_features
        train_parts.append(array)
        metadata["train_files"].append(info)

    assert features is not None
    if len(features) != args.expected_channels:
        raise ValueError(
            f"Expected {args.expected_channels} HAI channels, found {len(features)}. "
            f"Feature columns: {features}"
        )

    for test_name, label_name in zip(TEST_FILES, LABEL_FILES):
        test_array, test_times, current_features, test_info = read_data_file(
            input_dir / test_name, features
        )
        labels, label_times, label_info = read_label_file(input_dir / label_name)

        if len(test_array) != len(labels):
            raise ValueError(
                f"{test_name} has {len(test_array)} rows but "
                f"{label_name} has {len(labels)} labels."
            )
        alignment = validate_timestamp_alignment(
            test_times=test_times,
            label_times=label_times,
            test_name=test_name,
            label_name=label_name,
        )
        test_info["label_timestamp_alignment"] = alignment
        label_info["test_timestamp_alignment"] = alignment

        if alignment["status"] != "exact":
            print(
                f"Timestamp alignment accepted for {test_name}/{label_name}: "
                f"{alignment['status']}, "
                f"offset_seconds={alignment['constant_offset_seconds']}"
            )

        test_parts.append(test_array)
        label_parts.append(labels)
        metadata["test_files"].append(test_info)
        metadata["label_files"].append(label_info)

    train = np.concatenate(train_parts, axis=0).astype(np.float32, copy=False)
    test = np.concatenate(test_parts, axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(label_parts, axis=0).astype(np.int64, copy=False)

    if train.shape[1] != args.expected_channels:
        raise AssertionError(train.shape)
    if test.shape[1] != args.expected_channels:
        raise AssertionError(test.shape)
    if test.shape[0] != labels.shape[0]:
        raise AssertionError((test.shape, labels.shape))

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "HAI_train.npy"
    test_path = output_dir / "HAI_test.npy"
    label_path = output_dir / "HAI_test_label.npy"
    columns_path = output_dir / "HAI_columns.json"
    metadata_path = output_dir / "HAI_preprocess_metadata.json"

    np.save(train_path, train)
    np.save(test_path, test)
    np.save(label_path, labels)
    columns_path.write_text(
        json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata.update({
        "feature_count": int(len(features)),
        "feature_columns": features,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "label_shape": list(labels.shape),
        "train_nan_count": int(np.isnan(train).sum()),
        "test_nan_count": int(np.isnan(test).sum()),
        "test_attack_points": int(labels.sum()),
        "test_attack_ratio_percent": float(labels.mean() * 100.0),
        "output_sha256": {
            train_path.name: sha256(train_path),
            test_path.name: sha256(test_path),
            label_path.name: sha256(label_path),
        },
    })
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("HAI 23.05 preprocessing completed.")
    print(f"features             : {len(features)}")
    print(f"train shape          : {train.shape}")
    print(f"test shape           : {test.shape}")
    print(f"label shape          : {labels.shape}")
    print(f"train NaN count      : {np.isnan(train).sum()}")
    print(f"test NaN count       : {np.isnan(test).sum()}")
    print(f"test attack points   : {int(labels.sum())}")
    print(f"test attack ratio    : {labels.mean() * 100.0:.6f}%")
    print(f"output directory     : {output_dir}")


if __name__ == "__main__":
    main()
