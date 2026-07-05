from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_label(path: Path) -> np.ndarray:
    labels = np.load(path, mmap_mode="r")
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    elif labels.ndim != 1:
        labels = labels.reshape(-1)
    return (labels > 0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 SMD 是否满足 PPLAD 的 NPY 输入格式"
    )
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/SMD"))
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.5,
        help="仅用于打印基准阈值比例，不修改数据",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    expected = {
        "train": data_dir / "SMD_train.npy",
        "test": data_dir / "SMD_test.npy",
        "label": data_dir / "SMD_test_label.npy",
    }
    missing = [str(path) for path in expected.values() if not path.exists()]
    if missing:
        raw_dirs = [data_dir / name for name in ("train", "test", "test_label")]
        raw_found = all(path.is_dir() for path in raw_dirs)
        message = [
            "缺少 PPLAD 所需的 SMD NPY 文件：",
            *[f"  {path}" for path in missing],
        ]
        if raw_found:
            message.extend(
                [
                    "",
                    "检测到按机器划分的原始 SMD 目录。",
                    "本脚本不会自动把不同机器直接拼接，因为机器边界会产生人工跳变，",
                    "并且实体级训练/测试与整体拼接属于不同评估协议。",
                    "请优先使用论文中明确说明的实体级协议，或提供已经按既定协议生成的 NPY 文件。",
                ]
            )
        else:
            message.extend(
                [
                    "",
                    "期望目录内容：",
                    "  dataset/SMD/SMD_train.npy",
                    "  dataset/SMD/SMD_test.npy",
                    "  dataset/SMD/SMD_test_label.npy",
                ]
            )
        raise FileNotFoundError("\n".join(message))

    train = np.load(expected["train"], mmap_mode="r")
    test = np.load(expected["test"], mmap_mode="r")
    labels = _load_label(expected["label"])

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(
            f"训练集和测试集必须是二维数组；train={train.shape}, test={test.shape}"
        )
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            f"训练/测试通道数不一致：train={train.shape[1]}, test={test.shape[1]}"
        )
    if test.shape[0] != labels.shape[0]:
        raise ValueError(
            f"测试长度与标签长度不一致：test={test.shape[0]}, labels={labels.shape[0]}"
        )
    if train.shape[0] < 100 or test.shape[0] < 100:
        raise ValueError("数据长度小于 win_size=100，无法执行当前实验。")

    print("SMD 数据检查通过。")
    print(f"train shape       : {tuple(train.shape)}")
    print(f"test shape        : {tuple(test.shape)}")
    print(f"label shape       : {tuple(labels.shape)}")
    print(f"input channels    : {int(train.shape[1])}")
    print(f"true anomaly ratio: {100.0 * float(labels.mean()):.4f}%")
    print(
        f"benchmark threshold ratio: {args.threshold_ratio}%"
        "（阈值超参数，不是真实异常占比）"
    )


if __name__ == "__main__":
    main()
