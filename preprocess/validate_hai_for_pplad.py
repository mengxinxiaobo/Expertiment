#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate HAI NPY files for the official PPLAD HAI setup."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/HAI"))
    parser.add_argument("--expected-channels", type=int, default=86)
    parser.add_argument("--win-size", type=int, default=90)
    args = parser.parse_args()

    paths = {
        "train": args.data_dir / "HAI_train.npy",
        "test": args.data_dir / "HAI_test.npy",
        "label": args.data_dir / "HAI_test_label.npy",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    train = np.load(paths["train"], mmap_mode="r")
    test = np.load(paths["test"], mmap_mode="r")
    label = np.load(paths["label"], mmap_mode="r").reshape(-1)

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"train/test must be 2D: {train.shape}, {test.shape}")
    if train.shape[1] != args.expected_channels:
        raise ValueError(
            f"Train channels={train.shape[1]}, expected={args.expected_channels}"
        )
    if test.shape[1] != args.expected_channels:
        raise ValueError(
            f"Test channels={test.shape[1]}, expected={args.expected_channels}"
        )
    if test.shape[0] != label.shape[0]:
        raise ValueError("Test/label lengths differ.")
    if train.shape[0] < args.win_size or test.shape[0] < args.win_size:
        raise ValueError("Train or test is shorter than win_size.")
    if np.isinf(train).any() or np.isinf(test).any():
        raise ValueError("Train/test contains Inf.")
    unique = np.unique(label)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"Labels are not binary: {unique}")

    print("HAI data validation passed.")
    print(f"train shape          : {train.shape}, dtype={train.dtype}")
    print(f"test shape           : {test.shape}, dtype={test.dtype}")
    print(f"label shape          : {label.shape}, dtype={label.dtype}")
    print(f"channels             : {train.shape[1]}")
    print(f"train NaN count      : {int(np.isnan(train).sum())}")
    print(f"test NaN count       : {int(np.isnan(test).sum())}")
    print(f"test attack points   : {int(label.sum())}")
    print(f"test attack ratio    : {label.mean() * 100.0:.6f}%")
    print(f"win_size             : {args.win_size}")


if __name__ == "__main__":
    main()
