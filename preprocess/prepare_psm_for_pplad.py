from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _read_feature_csv(path: Path) -> Tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV 为空：{path}")

    # PSM 官方 CSV 的第一列是时间/索引列，不参与建模。
    feature_frame = frame.iloc[:, 1:].copy()
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.dropna(axis=1, how="all")
    if feature_frame.shape[1] == 0:
        raise ValueError(f"未在 {path} 中找到数值特征列。")

    values = feature_frame.to_numpy(dtype=np.float32, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values, [str(c) for c in feature_frame.columns]


def _read_label_csv(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV 为空：{path}")

    label_frame = frame.iloc[:, 1:].copy()
    label_frame = label_frame.apply(pd.to_numeric, errors="coerce")
    label_frame = label_frame.dropna(axis=1, how="all")
    if label_frame.shape[1] == 0:
        # 极少数文件没有时间列，回退为读取全部列。
        label_frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if label_frame.shape[1] == 0:
        raise ValueError(f"未在 {path} 中找到标签列。")

    values = label_frame.to_numpy(dtype=np.float32, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    # 若存在多个标签列，任一列异常即视为异常。
    labels = (values.max(axis=1) > 0).astype(np.float32)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="将 PSM CSV 转换为 PPLAD 使用的 NPY 文件")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/PSM"))
    parser.add_argument("--force", action="store_true", help="覆盖已经存在的 NPY 文件")
    args = parser.parse_args()

    data_dir = args.data_dir
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    label_csv = data_dir / "test_label.csv"
    for path in (train_csv, test_csv, label_csv):
        if not path.exists():
            raise FileNotFoundError(f"缺少文件：{path}")

    outputs = {
        "train": data_dir / "PSM_train.npy",
        "test": data_dir / "PSM_test.npy",
        "label": data_dir / "PSM_test_label.npy",
    }
    if not args.force and all(path.exists() for path in outputs.values()):
        train = np.load(outputs["train"], mmap_mode="r")
        test = np.load(outputs["test"], mmap_mode="r")
        labels = np.load(outputs["label"], mmap_mode="r")
        print("PSM NPY 文件已存在，跳过转换。")
    else:
        train, train_columns = _read_feature_csv(train_csv)
        test, test_columns = _read_feature_csv(test_csv)
        labels = _read_label_csv(label_csv)

        if train.shape[1] != test.shape[1]:
            raise ValueError(
                f"训练/测试特征维度不一致：train={train.shape[1]}, test={test.shape[1]}"
            )
        if train_columns != test_columns:
            raise ValueError("训练集和测试集的特征列名或顺序不一致。")
        if test.shape[0] != labels.shape[0]:
            raise ValueError(
                f"测试长度与标签长度不一致：test={test.shape[0]}, labels={labels.shape[0]}"
            )

        np.save(outputs["train"], train)
        np.save(outputs["test"], test)
        np.save(outputs["label"], labels)
        print("已生成：")
        for path in outputs.values():
            print(f"  {path}")

    anomaly_ratio = 100.0 * float(np.asarray(labels).mean())
    print(f"train shape       : {tuple(train.shape)}")
    print(f"test shape        : {tuple(test.shape)}")
    print(f"label shape       : {tuple(labels.shape)}")
    print(f"input channels    : {int(train.shape[1])}")
    print(f"true anomaly ratio: {anomaly_ratio:.4f}%")
    print("benchmark threshold ratio: 1.0%（与常见 PSM 基准脚本保持一致）")


if __name__ == "__main__":
    main()
