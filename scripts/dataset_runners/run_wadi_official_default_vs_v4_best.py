#!/usr/bin/env python3
from __future__ import annotations

"""
WADI 主对比训练与检测程序。

比较口径
--------
1. Original PPLAD：官方仓库没有独立 scripts/WADI.sh，本程序使用
   官方 main.py 中声明的仓库默认参数，并仅把数据集和通道数替换为 WADI。
2. V4：使用统一 V4 训练预算（win=100、batch=128、10 epochs），
   联合搜索 score mode（gap/total/combined）和 anormly_ratio，
   选择 PA-F1 最高结果。
3. V4 阈值搜索属于 oracle best / best-over-grid（测试集标签选优）
   口径，必须在论文中明确说明。
4. WADI 官方数据加载器使用名称 WaDi 和 WaDi_*.npy；本程序兼容
   WaDi_*.npy、WADI_*.npy 和 wadi_*.npy 三种文件命名。
"""

import argparse
import csv
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



_WADI_ARRAY_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def resolve_wadi_files(data_dir: Path) -> Dict[str, Path]:
    """兼容官方 WaDi 前缀和项目常用 WADI 前缀。"""
    data_dir = data_dir.resolve()
    candidates = {
        "train": ["WaDi_train.npy", "WADI_train.npy", "wadi_train.npy"],
        "test": ["WaDi_test.npy", "WADI_test.npy", "wadi_test.npy"],
        "label": [
            "WaDi_test_label.npy",
            "WADI_test_label.npy",
            "wadi_test_label.npy",
        ],
    }
    resolved: Dict[str, Path] = {}
    for key, names in candidates.items():
        for name in names:
            path = data_dir / name
            if path.exists():
                resolved[key] = path
                break
        if key not in resolved:
            raise FileNotFoundError(
                f"缺少 WADI {key} 文件。已检查："
                + ", ".join(str(data_dir / name) for name in names)
            )
    return resolved


def normalize_wadi_labels(labels: np.ndarray) -> np.ndarray:
    """
    处理后的 npy 通常为 0=正常、1=异常。
    若仍是 WADI 原始的 1=正常、-1=攻击，则转换为 0/1。
    """
    values = np.asarray(labels).reshape(-1)
    unique = set(np.unique(values).tolist())
    if unique.issubset({0, 1, 0.0, 1.0}):
        return values.astype(np.float32, copy=False)
    if unique.issubset({-1, 1, -1.0, 1.0}):
        return (values < 0).astype(np.float32)
    raise ValueError(
        f"WADI 标签必须为 0/1 或 -1/1，实际唯一值为：{sorted(unique)}"
    )


def install_flexible_wadi_loader(data_loader_module):
    """
    替换官方 WaDiSegLoader：
    - 兼容 WaDi/WADI/wadi 文件名前缀；
    - 四个 train/val/test/thre loader 共享标准化数组，降低 WADI 内存占用；
    - 同时接受 dataset='WADI' 和 dataset='WaDi'。
    """
    original_get_loader_segment = data_loader_module.get_loader_segment

    class FlexibleWaDiSegLoader:
        def __init__(
            self,
            data_path: str,
            win_size: int,
            step: int,
            mode: str = "train",
        ) -> None:
            self.mode = mode
            self.step = step
            self.win_size = win_size

            key = str(Path(data_path).resolve())
            if key not in _WADI_ARRAY_CACHE:
                paths = resolve_wadi_files(Path(data_path))
                train = np.load(paths["train"])
                test = np.load(paths["test"])
                labels = normalize_wadi_labels(np.load(paths["label"]))

                train = np.nan_to_num(train).astype(np.float32, copy=False)
                test = np.nan_to_num(test).astype(np.float32, copy=False)

                scaler = data_loader_module.StandardScaler()
                scaler.fit(train)
                train = scaler.transform(train).astype(np.float32, copy=False)
                test = scaler.transform(test).astype(np.float32, copy=False)

                _WADI_ARRAY_CACHE[key] = (train, test, labels)

            train, test, labels = _WADI_ARRAY_CACHE[key]
            self.train = train
            self.test = test
            self.val = test
            self.test_labels = labels

        def __len__(self) -> int:
            if self.mode == "train":
                return (self.train.shape[0] - self.win_size) // self.step + 1
            if self.mode in {"val", "test"}:
                return (self.test.shape[0] - self.win_size) // self.step + 1
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

        def __getitem__(self, index: int):
            index = index * self.step
            if self.mode == "train":
                return (
                    np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]),
                )
            if self.mode == "val":
                return (
                    np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]),
                )
            if self.mode == "test":
                return (
                    np.float32(self.test[index:index + self.win_size]),
                    np.float32(
                        self.test_labels[index:index + self.win_size]
                    ),
                )

            start = index // self.step * self.win_size
            end = start + self.win_size
            return (
                np.float32(self.test[start:end]),
                np.float32(self.test_labels[start:end]),
            )

    data_loader_module.WaDiSegLoader = FlexibleWaDiSegLoader

    def flexible_get_loader_segment(
        index,
        data_path,
        batch_size,
        win_size=100,
        step=100,
        mode="train",
        dataset="KDD",
    ):
        loader_dataset = "WaDi" if str(dataset).upper() == "WADI" else dataset
        return original_get_loader_segment(
            index,
            data_path,
            batch_size,
            win_size=win_size,
            step=step,
            mode=mode,
            dataset=loader_dataset,
        )

    data_loader_module.get_loader_segment = flexible_get_loader_segment
    return flexible_get_loader_segment


def bootstrap_project(root: Path):
    """
    项目根目录提供 data_loader/solver/main.py；
    BaselineModels/PPLAD-main 提供官方 model.PPLAD。
    """
    root = root.resolve()
    baseline = root / "BaselineModels" / "PPLAD-main"
    pplad_file = baseline / "model" / "PPLAD.py"
    if not pplad_file.exists():
        raise FileNotFoundError(f"找不到官方 PPLAD：{pplad_file}")

    baseline_str = str(baseline)
    if baseline_str in sys.path:
        sys.path.remove(baseline_str)
    sys.path.insert(0, baseline_str)

    importlib.import_module("model.RevIN")
    importlib.import_module("model.PPLAD")

    if baseline_str in sys.path:
        sys.path.remove(baseline_str)

    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    data_loader_module = importlib.import_module("data_factory.data_loader")
    flexible_loader = install_flexible_wadi_loader(data_loader_module)

    solver_module = importlib.import_module("solver")
    official_pplad_module = importlib.import_module("model.PPLAD")
    solver_module.PPLAD = official_pplad_module.PPLAD
    solver_module.get_loader_segment = flexible_loader

    main_module = importlib.import_module("main")
    main_module.get_loader_segment = flexible_loader

    Solver = solver_module.Solver
    V4Solver = main_module.AdaptiveSparseAnchorSolverV4
    return Solver, V4Solver


def verify_data(root: Path) -> int:
    data_dir = root / "dataset" / "WADI"
    paths = resolve_wadi_files(data_dir)

    train = np.load(paths["train"], mmap_mode="r")
    test = np.load(paths["test"], mmap_mode="r")
    raw_labels = np.load(paths["label"], mmap_mode="r")
    labels = normalize_wadi_labels(raw_labels)

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(
            f"WADI 数据应为二维：train={train.shape}, test={test.shape}"
        )
    if train.shape[1] != test.shape[1]:
        raise ValueError("WADI train/test 通道数不一致。")
    if len(test) != len(labels):
        raise ValueError(
            f"WADI test 与 label 长度不一致：{len(test)} vs {len(labels)}"
        )

    channels = int(train.shape[1])
    print("WADI data check passed")
    print("train file  :", paths["train"])
    print("test file   :", paths["test"])
    print("label file  :", paths["label"])
    print("train shape :", tuple(train.shape))
    print("test shape  :", tuple(test.shape))
    print("label shape :", tuple(labels.shape))
    print("channels    :", channels)
    print(
        "true anomaly ratio: {:.6f}%".format(
            float(np.asarray(labels).mean()) * 100.0
        )
    )
    return channels


def original_config(root: Path, channels: int) -> dict:
    """
    官方仓库 main.py 默认参数应用于 WADI。

    注意：官方仓库没有独立 scripts/WADI.sh，因此这不是经过
    WADI 专门调优的“官方 WADI shell 配置”。
    """
    return {
        "index": 137,
        "dataset": "WADI",
        "data_path": "WADI",
        "input_c": channels,
        "output_c": channels,
        "win_size": 60,
        "batch_size": 128,
        "num_epochs": 3,
        "lr": 1e-4,
        "local_size": [3],
        "global_size": [20],
        "d_model": 128,
        "r": 0.5,
        "similar": "MSE",
        "loss_fuc": "MSE",
        "anormly_ratio": 0.50,
        "model_save_path": str(
            root
            / "checkpoints"
            / "WADI_OFFICIAL_DEFAULT_VS_V4_BEST"
            / "ORIGINAL"
        ),
        "mode": "train",
        "rec_timeseries": True,
        "use_gpu": torch.cuda.is_available(),
        "use_multi_gpu": False,
        "gpu": 0,
        "devices": "0",
    }


def v4_config(
    root: Path,
    channels: int,
    seed: int,
    epochs: int,
) -> dict:
    """
    WADI V4 统一训练预算：
    win=100, batch=128, epochs=10,
    local candidates=1..8 / top2,
    global candidates=[12,16,20,24,28,32,40,48] / top4。

    历史统一脚本暂用 anormly_ratio=0.50；本程序不固定该值，
    而是在预定义网格内联合搜索分数模式和阈值。
    """
    return {
        "dataset": "WADI",
        "data_path": "WADI",
        "input_c": channels,
        "output_c": channels,
        "win_size": 100,
        "batch_size": 128,
        "num_epochs": epochs,
        "lr": 1e-3,
        "anormly_ratio": 0.50,
        "index": 137,
        "mode": "train",
        "seed": seed,
        "local_candidate_lags": [1, 2, 3, 4, 5, 6, 7, 8],
        "global_candidate_lags": [12, 16, 20, 24, 28, 32, 40, 48],
        "local_topk": 2,
        "global_topk": 4,
        "selector_hidden": 8,
        "fitter_hidden": 8,
        "selector_temperature": 0.5,
        "similarity_tau": 1.0,
        "sigma_min": 0.03,
        "sigma_max": 1.50,
        "area_weight": 0.1,
        "selector_balance_weight": 0.05,
        "gap_weight": 1.0,
        "relation_input": "instance",
        "score_modes": ["gap", "total", "combined"],
        "primary_score": "total",
        "score_normalization": "official",
        "threshold_source": "original",
        "quantile_method": "exact",
        "quantile_buffer": 50000,
        # 兼容父 Solver 初始化；不参与 V4 前向结构。
        "local_size": 3,
        "global_size": [20],
        "d_model": 8,
        "loss_fuc": "MSE",
        "r": 0.5,
        "similar": "MSE",
        "rec_timeseries": True,
        "model_save_path": str(
            root
            / "checkpoints"
            / "WADI_OFFICIAL_DEFAULT_VS_V4_BEST"
            / "V4"
        ),
        "result_path": str(
            root
            / "results"
            / "WADI_OFFICIAL_DEFAULT_VS_V4_BEST"
            / "V4"
        ),
        "use_gpu": torch.cuda.is_available(),
        "use_multi_gpu": False,
        "gpu": 0,
        "devices": "0",
    }


def count_trainable(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def point_adjust(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


def binary_metrics(gt: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt, pred, average="binary", zero_division=0
    )
    return {
        f"{prefix}_accuracy": float(accuracy_score(gt, pred)),
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": float(f1),
        f"{prefix}_mcc": float(matthews_corrcoef(gt, pred)),
    }


@torch.no_grad()
def collect_v4_scores(runner, loader, include_labels: bool) -> Tuple[Dict[str, np.ndarray], np.ndarray | None]:
    runner.model.eval()
    parts: Dict[str, List[np.ndarray]] = {
        mode: [] for mode in runner.score_modes
    }
    label_parts: List[np.ndarray] = []

    for input_data, labels in loader:
        _, details = runner._forward_batch(input_data)
        score_dict = runner._score_dict(details)
        for mode in runner.score_modes:
            values = score_dict[mode].detach().cpu().numpy().reshape(-1)
            if not np.isfinite(values).all():
                raise RuntimeError(f"{mode} 分数包含 NaN/Inf。")
            parts[mode].append(values)
        if include_labels:
            label_parts.append(labels.detach().cpu().numpy().reshape(-1))

    scores = {
        mode: np.concatenate(values, axis=0)
        for mode, values in parts.items()
    }
    labels_out = (
        np.concatenate(label_parts, axis=0).astype(int)
        if include_labels else None
    )
    return scores, labels_out


def metrics_from_confusion(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
    prefix: str,
) -> Dict[str, np.ndarray]:
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    tn = np.asarray(tn, dtype=np.float64)

    total = tp + fp + fn + tn
    precision_den = tp + fp
    recall_den = tp + fn
    f1_den = 2.0 * tp + fp + fn

    accuracy = np.divide(
        tp + tn, total, out=np.zeros_like(tp), where=total > 0
    )
    precision = np.divide(
        tp, precision_den, out=np.zeros_like(tp), where=precision_den > 0
    )
    recall = np.divide(
        tp, recall_den, out=np.zeros_like(tp), where=recall_den > 0
    )
    f1 = np.divide(
        2.0 * tp, f1_den, out=np.zeros_like(tp), where=f1_den > 0
    )

    mcc_den = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = np.divide(
        tp * tn - fp * fn,
        mcc_den,
        out=np.zeros_like(tp),
        where=mcc_den > 0,
    )

    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_mcc": mcc,
    }


def contiguous_anomaly_segments(gt: np.ndarray) -> List[Tuple[int, int]]:
    gt = np.asarray(gt, dtype=np.int8).reshape(-1)
    padded = np.pad(gt, (1, 1), mode="constant")
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def fast_threshold_metrics(
    scores: np.ndarray,
    gt: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    与 raw prediction + point adjustment 完全等价，但只排序一次。

    对正常点，PA 不改变预测，因此 FP/TN 可由正常分数排序后
    使用 searchsorted 计算。
    对每个连续异常区间，只要区间内任一点分数超过阈值，
    PA 就把整个区间标为异常。因此只需保存每段最大分数和长度。
    """
    scores = np.asarray(scores).reshape(-1)
    gt = np.asarray(gt, dtype=np.int8).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)

    if scores.size != gt.size:
        raise ValueError(
            f"score/label 长度不一致：{scores.size} != {gt.size}"
        )

    normal_scores = np.sort(scores[gt == 0])
    anomaly_scores = np.sort(scores[gt == 1])

    normal_count = int(normal_scores.size)
    anomaly_count = int(anomaly_scores.size)

    fp = normal_count - np.searchsorted(
        normal_scores, thresholds, side="right"
    )
    tn = normal_count - fp

    raw_tp = anomaly_count - np.searchsorted(
        anomaly_scores, thresholds, side="right"
    )
    raw_fn = anomaly_count - raw_tp

    raw = metrics_from_confusion(raw_tp, fp, raw_fn, tn, "raw")

    segments = contiguous_anomaly_segments(gt)
    if segments:
        segment_max = np.asarray(
            [float(np.max(scores[start:end])) for start, end in segments],
            dtype=np.float64,
        )
        segment_lengths = np.asarray(
            [end - start for start, end in segments],
            dtype=np.int64,
        )
        order = np.argsort(segment_max, kind="mergesort")
        sorted_max = segment_max[order]
        sorted_lengths = segment_lengths[order]
        cumulative_lengths = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(sorted_lengths)]
        )
        missed_segment_count = np.searchsorted(
            sorted_max, thresholds, side="right"
        )
        pa_fn = cumulative_lengths[missed_segment_count]
        pa_tp = anomaly_count - pa_fn
    else:
        pa_tp = np.zeros_like(thresholds, dtype=np.int64)
        pa_fn = np.zeros_like(thresholds, dtype=np.int64)

    pa = metrics_from_confusion(pa_tp, fp, pa_fn, tn, "pa")
    return {**raw, **pa}


def build_ratio_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    if minimum <= 0 or maximum <= minimum or step <= 0:
        raise ValueError("必须满足 0 < ratio_min < ratio_max 且 ratio_step > 0。")
    count = int(round((maximum - minimum) / step))
    grid = minimum + np.arange(count + 1, dtype=np.float64) * step
    grid = grid[grid <= maximum + 1e-12]
    return np.round(grid, 10)


def search_v4_best(
    runner,
    ratios: np.ndarray,
    output_dir: Path,
) -> Dict[str, object]:
    """
    WADI 分数样本较多。每个 score mode 单独收集、评估并释放；
    对全部阈值使用排序+二分查找计算指标，避免逐阈值扫描全量数据。
    """
    print("\nSearching WADI V4 score modes and thresholds ...")
    rows: List[Dict[str, object]] = []
    best_global: Dict[str, object] | None = None
    best_score_values: np.ndarray | None = None
    best_gt: np.ndarray | None = None
    percentiles = 100.0 - ratios

    original_modes = list(runner.score_modes)

    for mode in original_modes:
        print(f"Collecting scores for mode={mode} ...")
        runner.score_modes = [mode]

        train_scores, _ = collect_v4_scores(
            runner, runner.train_loader, include_labels=False
        )
        test_scores, gt = collect_v4_scores(
            runner, runner.thre_loader, include_labels=True
        )
        if gt is None:
            raise RuntimeError("未收集到 WADI 标签。")

        train_values = train_scores[mode]
        test_values = test_scores[mode]
        combined = np.concatenate([train_values, test_values], axis=0)

        # NumPy 精确百分位。一次调用计算全部阈值。
        thresholds = np.percentile(combined, percentiles)
        train_count = int(train_values.size)
        test_count = int(test_values.size)

        del combined, train_scores, test_scores, train_values

        metric_arrays = fast_threshold_metrics(
            test_values, gt, thresholds
        )

        for index, (ratio, threshold) in enumerate(zip(ratios, thresholds)):
            row: Dict[str, object] = {
                "score_mode": mode,
                "anormly_ratio": float(ratio),
                "threshold": float(threshold),
                "threshold_source": "train+test",
                "score_normalization": runner.score_normalization,
                "train_score_count": train_count,
                "test_score_count": test_count,
            }
            for key, values in metric_arrays.items():
                row[key] = float(values[index])
            rows.append(row)

            if best_global is None or (
                float(row["pa_f1"]),
                float(row["pa_precision"]),
                float(row["pa_recall"]),
                float(row["pa_accuracy"]),
            ) > (
                float(best_global["pa_f1"]),
                float(best_global["pa_precision"]),
                float(best_global["pa_recall"]),
                float(best_global["pa_accuracy"]),
            ):
                best_global = dict(row)
                best_score_values = test_values.copy()
                best_gt = gt.copy()

        del thresholds, metric_arrays, test_values, gt

    runner.score_modes = original_modes

    if best_global is None or best_score_values is None or best_gt is None:
        raise RuntimeError("WADI 阈值搜索未得到有效结果。")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "v4_threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_threshold = float(best_global["threshold"])
    best_pred_raw = (best_score_values > best_threshold).astype(np.int8)
    best_pred_pa = point_adjust(best_pred_raw, best_gt)

    np.savetxt(output_dir / "v4_best_score.txt", best_score_values, fmt="%.10f")
    np.savetxt(output_dir / "v4_best_pred_raw.txt", best_pred_raw, fmt="%d")
    np.savetxt(output_dir / "v4_best_pred_pa.txt", best_pred_pa, fmt="%d")
    np.savetxt(output_dir / "v4_label.txt", best_gt, fmt="%d")

    result: Dict[str, object] = {
        "selection_protocol": "oracle_best_over_predefined_test_threshold_grid",
        "search_objective": "maximum PA-F1",
        "search_modes": original_modes,
        "threshold_evaluation": "exact_sorted_search_equivalent_to_point_adjustment",
        "ratio_grid": {
            "min": float(ratios.min()),
            "max": float(ratios.max()),
            "step": float(ratios[1] - ratios[0]) if ratios.size > 1 else None,
            "count": int(ratios.size),
        },
        "best": best_global,
        "sweep_csv": str(csv_path),
    }

    (output_dir / "v4_best_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n================ V4 ORACLE BEST ================")
    print(f"score_mode    : {best_global['score_mode']}")
    print(f"anormly_ratio : {best_global['anormly_ratio']}")
    print(f"threshold     : {best_global['threshold']}")
    print(
        "PA Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, "
        "F1={:.4f}, MCC={:.4f}".format(
            float(best_global["pa_accuracy"]),
            float(best_global["pa_precision"]),
            float(best_global["pa_recall"]),
            float(best_global["pa_f1"]),
            float(best_global["pa_mcc"]),
        )
    )
    print(f"Saved sweep: {csv_path}")
    return result


def run_original(Solver, root: Path, channels: int) -> None:
    config = original_config(root, channels)
    output_dir = root / "results" / "WADI_OFFICIAL_DEFAULT_VS_V4_BEST"
    checkpoint_dir = Path(config["model_save_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("================ Original PPLAD / OFFICIAL-REPOSITORY DEFAULTS / WADI ================")
    for key in sorted(config):
        print(f"{key}: {config[key]}")

    runner = Solver(config)
    params = count_trainable(runner.model)
    print(f"Trainable parameters: {params:,}")

    start = time.perf_counter()
    runner.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start

    checkpoint = checkpoint_dir / "WADI_original_official_default_state.pt"
    torch.save({"model": runner.model.state_dict(), "config": config}, checkpoint)
    print(f"Saved checkpoint: {checkpoint}")

    accuracy, precision, recall, f1 = runner.test()
    metrics = {
        "dataset": "WADI",
        "model": "Original PPLAD",
        "protocol": "official_repository_defaults_from_main.py_applied_to_WADI",
        "config": config,
        "trainable_params": params,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint),
        "pa_accuracy": float(accuracy),
        "pa_precision": float(precision),
        "pa_recall": float(recall),
        "pa_f1": float(f1),
    }
    metrics_path = output_dir / "original_official_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved metrics: {metrics_path}")


def run_v4(
    V4Solver,
    root: Path,
    channels: int,
    seed: int,
    epochs: int,
    ratio_min: float,
    ratio_max: float,
    ratio_step: float,
) -> None:
    config = v4_config(root, channels, seed, epochs)
    output_dir = root / "results" / "WADI_OFFICIAL_DEFAULT_VS_V4_BEST"
    Path(config["model_save_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["result_path"]).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================ V4 / BEST-RESULT SEARCH ================")
    for key in sorted(config):
        print(f"{key}: {config[key]}")

    runner = V4Solver(config)
    params = count_trainable(runner.model)
    print(f"Trainable parameters: {params:,}")

    start = time.perf_counter()
    runner.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start

    ratios = build_ratio_grid(ratio_min, ratio_max, ratio_step)
    result = search_v4_best(runner, ratios, output_dir)

    checkpoint = Path(runner.checkpoint_path)
    summary = {
        "dataset": "WADI",
        "model": "V4 WADI best",
        "protocol": "model-specific best result",
        "config": config,
        "trainable_params": params,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint),
        **result,
    }
    summary_path = output_dir / "v4_best_run.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved V4 run summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["original", "v4"], required=True)
    parser.add_argument("--root", default="/mnt/c/Users/DING/Desktop/Experiment/CODE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v4-epochs", type=int, default=10)
    parser.add_argument("--ratio-min", type=float, default=0.10)
    parser.add_argument("--ratio-max", type=float, default=8.00)
    parser.add_argument("--ratio-step", type=float, default=0.01)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    os.chdir(root)

    # 原版 solver.py 中存在：
    #     if self.data_path == 'UCR' or 'UCR_AUG':
    # 该条件恒为 True，因此所有数据集测试结束后都会尝试写 result/<dataset>.csv。
    # 这里提前创建目录，避免 WADI 在指标已计算完成后因目录不存在而中断。
    (root / "result").mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    channels = verify_data(root)
    Solver, V4Solver = bootstrap_project(root)

    if args.model == "original":
        run_original(Solver, root, channels)
    else:
        run_v4(
            V4Solver,
            root,
            channels,
            args.seed,
            args.v4_epochs,
            args.ratio_min,
            args.ratio_max,
            args.ratio_step,
        )


if __name__ == "__main__":
    main()
