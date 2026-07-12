#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speed-oriented ASCA-AD V4 experiment for SMD.

用途：
1. 训练 / 评估 ASCA-AD-Speed-A 或 ASCA-AD-Speed-B。
2. 使用 fixed-combined 协议输出 PA-F1。
3. 测 batch=1、batch=128 和 full-test 推理速度。

放置位置：
    /mnt/c/Users/DING/Desktop/Experiment/CODE/scripts/benchmarks/run_smd_v4_speed_ab.py

运行示例：
    python -u scripts/benchmarks/run_smd_v4_speed_ab.py --variant speed_a --mode train-eval --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asca_ad.model import AdaptiveSparseAnchorSolverV4, set_seed  # noqa: E402


def max_rss_mib() -> float:
    """Linux/WSL 下 ru_maxrss 单位是 KB。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_trainable(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_total(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def variant_lags(variant: str) -> Tuple[list[int], list[int], int, int]:
    """
    speed_a:
        candidate 从 16 降到 8，优先测试。
    speed_b:
        candidate 从 16 降到 6，更激进，速度更可能快，但精度风险更大。
    """
    if variant == "speed_a":
        return [1, 2, 4, 8], [12, 24, 36, 48], 1, 2
    if variant == "speed_b":
        return [1, 2, 4], [12, 24, 48], 1, 1
    raise ValueError(f"Unknown variant: {variant}")


def build_config(args: argparse.Namespace) -> dict:
    local_lags, global_lags, local_topk, global_topk = variant_lags(args.variant)

    out_tag = f"SMD_V4_{args.variant.upper()}"
    return {
        "dataset": "SMD",
        "data_path": "SMD",
        "input_c": args.channels,
        "output_c": args.channels,
        "win_size": args.seq_len,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "lr": args.lr,
        # 注意：SMD 当前 ASCA-AD 主配置里是 0.9。
        # 如果你想和 TSLib iTransformer 的 anomaly_ratio=0.5 对齐，可运行时传 --anormly-ratio 0.5。
        "anormly_ratio": args.anormly_ratio,
        "index": 137,
        "mode": "train",
        "seed": args.seed,

        # 速度版核心改动：减少 candidate anchor 数量。
        "local_candidate_lags": local_lags,
        "global_candidate_lags": global_lags,
        "local_topk": local_topk,
        "global_topk": global_topk,

        # 保持原 V4 小 MLP 设置，不增加参数。
        "selector_hidden": args.selector_hidden,
        "fitter_hidden": args.fitter_hidden,
        "selector_temperature": args.selector_temperature,
        "similarity_tau": args.similarity_tau,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "area_weight": args.area_weight,
        "selector_balance_weight": args.selector_balance_weight,
        "gap_weight": args.gap_weight,
        "relation_input": args.relation_input,

        # fixed-combined 主协议：只测 combined score。
        "score_modes": ["combined"],
        "primary_score": "combined",
        "score_normalization": args.score_normalization,
        # original 表示 train + test score 的 percentile 阈值。
        "threshold_source": args.threshold_source,
        "quantile_method": "exact",
        "quantile_buffer": 50000,

        # 以下字段是兼容父 Solver 初始化；V4 前向结构不使用这些 PPLAD 字段。
        "local_size": 7,
        "global_size": [11],
        "d_model": 8,
        "loss_fuc": "MSE",
        "r": 0.5,
        "similar": "MSE",
        "rec_timeseries": True,

        "model_save_path": str(ROOT / "checkpoints" / out_tag / "V4"),
        "result_path": str(ROOT / "results" / out_tag / "V4"),
        "use_gpu": torch.cuda.is_available(),
        "use_multi_gpu": False,
        "gpu": 0,
        "devices": "0",
    }


@torch.inference_mode()
def benchmark_batch(
    solver: AdaptiveSparseAnchorSolverV4,
    raw_batch: torch.Tensor,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    model = solver.model
    model.eval()

    x = raw_batch[:batch_size].contiguous()
    if x.shape[0] < batch_size:
        raise RuntimeError(f"Batch too small: got {x.shape[0]}, need {batch_size}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        _ = solver._forward_batch(x)
    sync()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = solver._forward_batch(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    out = {
        "batch_size": int(batch_size),
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_std": float(arr.std()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
        "cpu_max_rss_mib": max_rss_mib(),
    }
    if torch.cuda.is_available():
        out["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        out["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024 / 1024
    return out


@torch.inference_mode()
def benchmark_full_test(
    solver: AdaptiveSparseAnchorSolverV4,
    repeats: int,
) -> Dict[str, float]:
    solver.model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    run_seconds = []
    total_points_list = []

    for _ in range(repeats):
        total_points = 0
        t0 = time.perf_counter()
        for input_data, _ in solver.thre_loader:
            score, _ = solver._forward_batch(input_data)
            total_points += int(score.numel())
        sync()
        run_seconds.append(time.perf_counter() - t0)
        total_points_list.append(total_points)

    arr = np.asarray(run_seconds, dtype=np.float64)
    points = int(np.mean(total_points_list))
    out = {
        "full_test_repeats": int(repeats),
        "full_test_seconds_mean": float(arr.mean()),
        "full_test_seconds_std": float(arr.std()),
        "full_test_seconds_p50": float(np.percentile(arr, 50)),
        "full_test_points": points,
        "full_test_points_per_second": float(points / arr.mean()),
        "cpu_max_rss_mib": max_rss_mib(),
    }
    if torch.cuda.is_available():
        out["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        out["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024 / 1024
    return out


def write_summary(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"

    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    pa = metrics["pa_metrics"]
    b1 = metrics["latency"]["batch_1"]
    bn = metrics["latency"]["batch_n"]
    ft = metrics["latency"]["full_test"]
    cfg = metrics["config_brief"]

    lines = []
    lines.append(f"# {metrics['variant']} on SMD\n")
    lines.append("## Config\n")
    lines.append(f"- local_candidate_lags: `{cfg['local_candidate_lags']}`")
    lines.append(f"- global_candidate_lags: `{cfg['global_candidate_lags']}`")
    lines.append(f"- local_topk/global_topk: `{cfg['local_topk']} / {cfg['global_topk']}`")
    lines.append(f"- seq_len: `{cfg['win_size']}`")
    lines.append(f"- batch_size: `{cfg['batch_size']}`")
    lines.append(f"- epochs: `{cfg['num_epochs']}`")
    lines.append(f"- anormly_ratio: `{cfg['anormly_ratio']}`")
    lines.append(f"- checkpoint: `{metrics['checkpoint_path']}`\n")

    lines.append("## Accuracy\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| PA Accuracy | {pa['pa_accuracy']:.6f} |")
    lines.append(f"| PA Precision | {pa['pa_precision']:.6f} |")
    lines.append(f"| PA Recall | {pa['pa_recall']:.6f} |")
    lines.append(f"| PA F1 | {pa['pa_f1']:.6f} |")
    lines.append("")

    lines.append("## Lightweight / Latency\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Trainable Params | {metrics['params']['trainable_params']} |")
    lines.append(f"| Total Params | {metrics['params']['total_params']} |")
    lines.append(f"| Batch=1 Mean Latency ms | {b1['latency_ms_mean']:.6f} |")
    lines.append(f"| Batch=1 P95 Latency ms | {b1['latency_ms_p95']:.6f} |")
    lines.append(f"| Batch={bn['batch_size']} Mean Latency ms | {bn['latency_ms_mean']:.6f} |")
    lines.append(f"| Batch={bn['batch_size']} P95 Latency ms | {bn['latency_ms_p95']:.6f} |")
    lines.append(f"| Full-test Seconds Mean | {ft['full_test_seconds_mean']:.6f} |")
    lines.append(f"| Full-test Throughput points/s | {ft['full_test_points_per_second']:.2f} |")
    lines.append(f"| CPU Max RSS MiB | {metrics['cpu_max_rss_mib']:.2f} |")
    if "gpu_peak_allocated_mib" in ft:
        lines.append(f"| Full-test GPU Peak Allocated MiB | {ft['gpu_peak_allocated_mib']:.2f} |")
        lines.append(f"| Full-test GPU Peak Reserved MiB | {ft['gpu_peak_reserved_mib']:.2f} |")
    lines.append("")
    lines.append("## Reference Target\n")
    lines.append("- 当前 iTransformer / SMD / batch=128 mean latency 约为 `1.967984 ms`。")
    lines.append("- 如果 Speed-A 的 batch=128 mean latency 小于该值，则速度超过当前 iTransformer 小配置。")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["speed_a", "speed_b"], default="speed_a")
    parser.add_argument("--mode", choices=["train", "eval", "train-eval"], default="train-eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--channels", type=int, default=38)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anormly-ratio", dest="anormly_ratio", type=float, default=0.9)

    parser.add_argument("--selector-hidden", type=int, default=8)
    parser.add_argument("--fitter-hidden", type=int, default=8)
    parser.add_argument("--selector-temperature", type=float, default=0.5)
    parser.add_argument("--similarity-tau", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.03)
    parser.add_argument("--sigma-max", type=float, default=1.50)
    parser.add_argument("--area-weight", type=float, default=0.1)
    parser.add_argument("--selector-balance-weight", type=float, default=0.05)
    parser.add_argument("--gap-weight", type=float, default=1.0)
    parser.add_argument("--relation-input", choices=["instance", "standardized"], default="instance")
    parser.add_argument("--score-normalization", choices=["official", "raw"], default="official")
    parser.add_argument("--threshold-source", choices=["original", "train"], default="original")

    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--full-test-repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = build_config(args)
    print("[variant]", args.variant)
    print("[config]", json.dumps({
        "local_candidate_lags": cfg["local_candidate_lags"],
        "global_candidate_lags": cfg["global_candidate_lags"],
        "local_topk": cfg["local_topk"],
        "global_topk": cfg["global_topk"],
        "anormly_ratio": cfg["anormly_ratio"],
        "epochs": cfg["num_epochs"],
    }, ensure_ascii=False, indent=2))

    solver = AdaptiveSparseAnchorSolverV4(cfg)

    if args.mode in ["train", "train-eval"]:
        solver.train()

    if args.mode == "eval":
        solver.load_checkpoint()

    pa_accuracy = pa_precision = pa_recall = pa_f1 = None
    if args.mode in ["eval", "train-eval"]:
        pa_accuracy, pa_precision, pa_recall, pa_f1 = solver.test()

    first_batch = next(iter(solver.thre_loader))[0]
    b1 = benchmark_batch(solver, first_batch, 1, args.warmup, args.repeats)
    bn = benchmark_batch(
        solver,
        first_batch,
        min(args.batch_size, int(first_batch.shape[0])),
        args.warmup,
        args.repeats,
    )
    ft = benchmark_full_test(solver, args.full_test_repeats)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"SMD_V4_{args.variant.upper()}"
    metrics = {
        "variant": args.variant,
        "dataset": "SMD",
        "checkpoint_path": solver.checkpoint_path,
        "config_brief": {
            "local_candidate_lags": cfg["local_candidate_lags"],
            "global_candidate_lags": cfg["global_candidate_lags"],
            "local_topk": cfg["local_topk"],
            "global_topk": cfg["global_topk"],
            "win_size": cfg["win_size"],
            "batch_size": cfg["batch_size"],
            "num_epochs": cfg["num_epochs"],
            "anormly_ratio": cfg["anormly_ratio"],
            "primary_score": cfg["primary_score"],
            "score_normalization": cfg["score_normalization"],
            "threshold_source": cfg["threshold_source"],
        },
        "pa_metrics": {
            "pa_accuracy": float(pa_accuracy) if pa_accuracy is not None else None,
            "pa_precision": float(pa_precision) if pa_precision is not None else None,
            "pa_recall": float(pa_recall) if pa_recall is not None else None,
            "pa_f1": float(pa_f1) if pa_f1 is not None else None,
        },
        "params": {
            "trainable_params": count_trainable(solver.model),
            "total_params": count_total(solver.model),
        },
        "latency": {
            "batch_1": b1,
            "batch_n": bn,
            "full_test": ft,
        },
        "cpu_max_rss_mib": max_rss_mib(),
    }

    write_summary(metrics, out_dir)

    print("\n[done]")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
