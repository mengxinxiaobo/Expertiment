#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def count_nonfinite(array: np.ndarray, chunk_rows: int = 50000) -> tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for start in range(0, array.shape[0], chunk_rows):
        block = np.asarray(array[start:start + chunk_rows])
        nan_count += int(np.isnan(block).sum())
        inf_count += int(np.isinf(block).sum())
    return nan_count, inf_count


def anomaly_segments(label: np.ndarray) -> np.ndarray:
    y = np.asarray(label, dtype=np.int64).reshape(-1)
    starts = np.flatnonzero((y == 1) & (np.r_[0, y[:-1]] == 0))
    ends = np.flatnonzero((y == 1) & (np.r_[y[1:], 0] == 0))
    return ends - starts + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate WADI NPY files for PPLAD's official WaDi loader."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/WADI"))
    parser.add_argument("--expected-channels", type=int, default=127)
    parser.add_argument("--win-size", type=int, default=90)
    args = parser.parse_args()

    paths = {
        "train": args.data_dir / "WaDi_train.npy",
        "test": args.data_dir / "WaDi_test.npy",
        "label": args.data_dir / "WaDi_test_label.npy",
        "metadata": args.data_dir / "WaDi_preprocess_metadata.json",
    }
    for key in ("train", "test", "label"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])

    train = np.load(paths["train"], mmap_mode="r")
    test = np.load(paths["test"], mmap_mode="r")
    label = np.load(paths["label"], mmap_mode="r").reshape(-1)

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"Train/test must be 2D: {train.shape}, {test.shape}")
    if train.shape[1] != args.expected_channels:
        raise ValueError(
            f"Train channels={train.shape[1]}, expected={args.expected_channels}"
        )
    if test.shape[1] != args.expected_channels:
        raise ValueError(
            f"Test channels={test.shape[1]}, expected={args.expected_channels}"
        )
    if test.shape[0] != label.shape[0]:
        raise ValueError(
            f"Test/label length mismatch: {test.shape[0]} vs {label.shape[0]}"
        )
    if train.shape[0] < args.win_size or test.shape[0] < args.win_size:
        raise ValueError("Train or test is shorter than win_size.")

    unique = np.unique(label)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"Labels are not binary: {unique}")

    train_nan, train_inf = count_nonfinite(train)
    test_nan, test_inf = count_nonfinite(test)
    if train_inf or test_inf:
        raise ValueError(
            f"Inf values are not allowed: train={train_inf}, test={test_inf}"
        )

    lengths = anomaly_segments(label)
    if len(lengths) == 0:
        raise ValueError("No attack segments were found in WaDi_test_label.npy.")

    metadata = {}
    if paths["metadata"].is_file():
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    print("WADI data validation passed.")
    print(f"train shape          : {train.shape}, dtype={train.dtype}")
    print(f"test shape           : {test.shape}, dtype={test.dtype}")
    print(f"label shape          : {label.shape}, dtype={label.dtype}")
    print(f"channels             : {train.shape[1]}")
    print(f"train NaN cells      : {train_nan}")
    print(f"test NaN cells       : {test_nan}")
    print("NaN handling         : PPLAD loader converts NaN to zero")
    print(f"test attack points   : {int(label.sum())}")
    print(f"test attack ratio    : {label.mean() * 100.0:.6f}%")
    print(f"attack segments      : {len(lengths)}")
    print(f"shortest segment     : {int(lengths.min())}")
    print(f"longest segment      : {int(lengths.max())}")
    print(f"mean segment length  : {float(lengths.mean()):.3f}")
    print(f"win_size             : {args.win_size}")
    if metadata:
        parsed = metadata.get("attack_interval_parser", {}).get(
            "valid_interval_count"
        )
        print(f"xlsx intervals       : {parsed}")


if __name__ == "__main__":
    main()
