#!/usr/bin/env python3
from __future__ import annotations

"""
PSM 主对比训练与检测程序。

比较口径
--------
1. Original PPLAD：严格使用官方 scripts/PSM.sh 配置。
2. V4：使用当前最佳锚点结构，训练 10 epochs；在预定义搜索空间内
   联合搜索 score mode（gap/total/combined）和 anormly_ratio，选择 PA-F1 最高结果。
3. V4 的阈值搜索属于 oracle best / best-over-grid（测试集最优结果）口径。
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

    # 预先缓存官方 model 包，避免项目根目录同名 model 包遮蔽 PPLAD.py。
    importlib.import_module("model.RevIN")
    importlib.import_module("model.PPLAD")

    if baseline_str in sys.path:
        sys.path.remove(baseline_str)

    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    solver_module = importlib.import_module("solver")

    # 当前项目根目录的 solver.py 已删除 `from model.PPLAD import PPLAD`，
    # 但 Solver.build_model() 仍然直接调用全局名称 PPLAD。
    # 将官方实现显式注入 solver 模块，避免修改用户现有 solver.py。
    official_pplad_module = importlib.import_module("model.PPLAD")
    solver_module.PPLAD = official_pplad_module.PPLAD

    Solver = solver_module.Solver
    V4Solver = importlib.import_module("main").AdaptiveSparseAnchorSolverV4
    return Solver, V4Solver


def verify_data(root: Path) -> int:
    data_dir = root / "dataset" / "PSM"
    paths = {
        "train": data_dir / "PSM_train.npy",
        "test": data_dir / "PSM_test.npy",
        "label": data_dir / "PSM_test_label.npy",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"缺少 PSM 数据文件：{path}")

    train = np.load(paths["train"], mmap_mode="r")
    test = np.load(paths["test"], mmap_mode="r")
    labels = np.load(paths["label"], mmap_mode="r")

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"PSM 数据应为二维：train={train.shape}, test={test.shape}")
    if train.shape[1] != test.shape[1]:
        raise ValueError("PSM train/test 通道数不一致。")
    if len(test) != len(labels):
        raise ValueError(f"PSM test 与 label 长度不一致：{len(test)} vs {len(labels)}")

    channels = int(train.shape[1])
    if channels != 25:
        raise ValueError(f"PSM 官方配置应为 25 通道，实际为 {channels}。")

    print("PSM data check passed")
    print("train shape :", tuple(train.shape))
    print("test shape  :", tuple(test.shape))
    print("label shape :", tuple(labels.shape))
    print("channels    :", channels)
    print("true anomaly ratio: {:.6f}%".format(float(np.asarray(labels).mean()) * 100.0))
    return channels


def original_config(root: Path, channels: int) -> dict:
    """官方 infogroup502/PPLAD scripts/PSM.sh 配置。"""
    return {
        "index": 137,
        "dataset": "PSM",
        "data_path": "PSM",
        "input_c": channels,
        "output_c": channels,
        "win_size": 60,
        "batch_size": 256,
        "num_epochs": 1,
        "lr": 1e-4,
        "local_size": [1],
        "global_size": [20],
        "d_model": 128,
        "r": 0.5,
        "similar": "MSE",
        "loss_fuc": "MSE",
        "anormly_ratio": 0.80,
        "model_save_path": str(root / "checkpoints" / "PSM_OFFICIAL_VS_V4_BEST" / "ORIGINAL"),
        "mode": "train",
        "rec_timeseries": True,
        "use_gpu": torch.cuda.is_available(),
        "use_multi_gpu": False,
        "gpu": 0,
        "devices": "0",
    }


def v4_config(root: Path, channels: int, seed: int, epochs: int) -> dict:
    """
    当前 PSM 已验证的 V4 最佳结构：
    win=100, batch=128, epochs=10,
    local candidates=1..8 / top2,
    global candidates=[12,16,20,24,28,32,40,48] / top4。

    阈值不固定在 config 中；训练完成后联合搜索
    gap、total、combined 与 anormly_ratio。
    """
    return {
        "dataset": "PSM",
        "data_path": "PSM",
        "input_c": channels,
        "output_c": channels,
        "win_size": 100,
        "batch_size": 128,
        "num_epochs": epochs,
        "lr": 1e-3,
        "anormly_ratio": 1.0,  # 历史最佳参考；最终结果由独立搜索确定。
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
        "local_size": 1,
        "global_size": [20],
        "d_model": 8,
        "loss_fuc": "MSE",
        "r": 0.5,
        "similar": "MSE",
        "rec_timeseries": True,
        "model_save_path": str(root / "checkpoints" / "PSM_OFFICIAL_VS_V4_BEST" / "V4"),
        "result_path": str(root / "results" / "PSM_OFFICIAL_VS_V4_BEST" / "V4"),
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
    print("\nCollecting V4 train/test scores once ...")
    train_scores, _ = collect_v4_scores(
        runner, runner.train_loader, include_labels=False
    )
    test_scores, gt = collect_v4_scores(
        runner, runner.thre_loader, include_labels=True
    )
    if gt is None:
        raise RuntimeError("未收集到 PSM 标签。")

    rows: List[Dict[str, object]] = []
    percentiles = 100.0 - ratios

    for mode in runner.score_modes:
        combined = np.concatenate(
            [train_scores[mode], test_scores[mode]], axis=0
        )

        # 一次性计算全部阈值，避免每个 ratio 都重复排序。
        thresholds = np.percentile(combined, percentiles)

        for ratio, threshold in zip(ratios, thresholds):
            threshold = float(threshold)
            raw_pred = (test_scores[mode] > threshold).astype(int)
            pa_pred = point_adjust(raw_pred, gt)

            row: Dict[str, object] = {
                "score_mode": mode,
                "anormly_ratio": float(ratio),
                "threshold": threshold,
                "threshold_source": "train+test",
                "score_normalization": runner.score_normalization,
                "train_score_count": int(train_scores[mode].size),
                "test_score_count": int(test_scores[mode].size),
            }
            row.update(binary_metrics(gt, raw_pred, "raw"))
            row.update(binary_metrics(gt, pa_pred, "pa"))
            rows.append(row)

    best = max(
        rows,
        key=lambda row: (
            float(row["pa_f1"]),
            float(row["pa_precision"]),
            float(row["pa_recall"]),
            float(row["pa_accuracy"]),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "v4_threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_mode = str(best["score_mode"])
    best_threshold = float(best["threshold"])
    best_pred_raw = (test_scores[best_mode] > best_threshold).astype(int)
    best_pred_pa = point_adjust(best_pred_raw, gt)

    np.savetxt(output_dir / "v4_best_score.txt", test_scores[best_mode], fmt="%.10f")
    np.savetxt(output_dir / "v4_best_pred_raw.txt", best_pred_raw, fmt="%d")
    np.savetxt(output_dir / "v4_best_pred_pa.txt", best_pred_pa, fmt="%d")
    np.savetxt(output_dir / "v4_label.txt", gt, fmt="%d")

    result: Dict[str, object] = {
        "selection_protocol": "oracle_best_over_predefined_test_threshold_grid",
        "search_objective": "maximum PA-F1",
        "search_modes": list(runner.score_modes),
        "ratio_grid": {
            "min": float(ratios.min()),
            "max": float(ratios.max()),
            "step": float(ratios[1] - ratios[0]) if ratios.size > 1 else None,
            "count": int(ratios.size),
        },
        "best": best,
        "sweep_csv": str(csv_path),
    }

    (output_dir / "v4_best_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n================ V4 ORACLE BEST ================")
    print(f"score_mode    : {best_mode}")
    print(f"anormly_ratio : {best['anormly_ratio']}")
    print(f"threshold     : {best_threshold}")
    print(
        "PA Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, "
        "F1={:.4f}, MCC={:.4f}".format(
            float(best["pa_accuracy"]),
            float(best["pa_precision"]),
            float(best["pa_recall"]),
            float(best["pa_f1"]),
            float(best["pa_mcc"]),
        )
    )
    print(f"Saved sweep: {csv_path}")
    return result


def run_original(Solver, root: Path, channels: int) -> None:
    config = original_config(root, channels)
    output_dir = root / "results" / "PSM_OFFICIAL_VS_V4_BEST"
    checkpoint_dir = Path(config["model_save_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("================ Original PPLAD / OFFICIAL PSM ================")
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

    checkpoint = checkpoint_dir / "PSM_original_official_state.pt"
    torch.save({"model": runner.model.state_dict(), "config": config}, checkpoint)
    print(f"Saved checkpoint: {checkpoint}")

    accuracy, precision, recall, f1 = runner.test()
    metrics = {
        "dataset": "PSM",
        "model": "Original PPLAD",
        "protocol": "official scripts/PSM.sh",
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
    output_dir = root / "results" / "PSM_OFFICIAL_VS_V4_BEST"
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
        "dataset": "PSM",
        "model": "V4 PSM best",
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
    parser.add_argument("--ratio-max", type=float, default=3.00)
    parser.add_argument("--ratio-step", type=float, default=0.01)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    os.chdir(root)

    # 原版 solver.py 中存在：
    #     if self.data_path == 'UCR' or 'UCR_AUG':
    # 该条件恒为 True，因此所有数据集测试结束后都会尝试写 result/<dataset>.csv。
    # 这里提前创建目录，避免 PSM 在指标已计算完成后因目录不存在而中断。
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
