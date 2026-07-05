#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PUMP NPY files for PPLAD.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/PUMP"))
    parser.add_argument("--win-size", type=int, default=60)
    args = parser.parse_args()

    train_path = args.data_dir / "PUMP_train.npy"
    test_path = args.data_dir / "PUMP_test.npy"
    label_path = args.data_dir / "PUMP_test_label.npy"

    for path in (train_path, test_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train = np.load(train_path, mmap_mode="r")
    test = np.load(test_path, mmap_mode="r")
    label = np.load(label_path, mmap_mode="r").reshape(-1)

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"train/test must be 2D: {train.shape}, {test.shape}")
    if train.shape[1] != test.shape[1]:
        raise ValueError("Train/test channel counts differ.")
    if test.shape[0] != label.shape[0]:
        raise ValueError("Test/label lengths differ.")
    if train.shape[0] < args.win_size or test.shape[0] < args.win_size:
        raise ValueError("Train or test sequence is shorter than win_size.")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Train/test contains NaN or Inf.")
    unique = np.unique(label)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"Labels are not binary: {unique}")

    print("PUMP data validation passed.")
    print(f"train shape         : {train.shape}, dtype={train.dtype}")
    print(f"test shape          : {test.shape}, dtype={test.dtype}")
    print(f"label shape         : {label.shape}, dtype={label.dtype}")
    print(f"channels            : {train.shape[1]}")
    print(f"test anomaly ratio  : {label.mean() * 100.0:.4f}%")
    print(f"win_size            : {args.win_size}")


if __name__ == "__main__":
    main()
