#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASCA-AD score-only fast inference benchmark.

目标：不改模型结构、不改参数、不改训练，只给原版 ASCA-AD 增加一个只计算
score_combined 的快速推理路径，并验证：
1) score-only 输出与原始 forward 的 score_combined 等价；
2) 检测指标不变；
3) batch=1 / batch=N / full-test 推理时间是否降低。

用法示例：
CUDA_VISIBLE_DEVICES=0 python -u scripts/benchmarks/benchmark_asca_score_only.py \
  --dataset SMD \
  --channels 38 \
  --anormly-ratio 0.9 \
  --checkpoint checkpoints/SMD/SMD_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_g12-16-20-24-28-32-40-48_kl2_kg4.pt \
  --out-dir results/SMD_ASCA_SCORE_ONLY_FAST | tee logs_SMD_ASCA_SCORE_ONLY_FAST.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asca_ad.model import AdaptiveSparseAnchorSolverV4, set_seed  # noqa: E402


def max_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "param_bytes": int(param_bytes),
        "param_mib": float(param_bytes / 1024 / 1024),
    }


def build_solver_config(args: argparse.Namespace) -> dict:
    return {
        "dataset": args.dataset,
        "data_path": args.dataset,
        "input_c": args.channels,
        "output_c": args.channels,
        "win_size": args.seq_len,
        "batch_size": args.batch_size,
        "num_epochs": 1,
        "lr": 1e-3,
        "anormly_ratio": args.anormly_ratio,
        "index": 137,
        "mode": "test",
        "seed": args.seed,
        "local_candidate_lags": args.local_lags,
        "global_candidate_lags": args.global_lags,
        "local_topk": args.local_topk,
        "global_topk": args.global_topk,
        "selector_hidden": args.selector_hidden,
        "fitter_hidden": args.fitter_hidden,
        "selector_temperature": args.selector_temperature,
        "similarity_tau": args.similarity_tau,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "area_weight": 0.1,
        "selector_balance_weight": 0.05,
        "gap_weight": args.gap_weight,
        "relation_input": args.relation_input,
        "score_modes": ["combined"],
        "primary_score": "combined",
        "score_normalization": args.score_normalization,
        "threshold_source": args.threshold_source,
        "quantile_method": "exact",
        "quantile_buffer": 50000,
        # 兼容原 PPLAD Solver 初始化。
        "local_size": 7,
        "global_size": [11],
        "d_model": 8,
        "loss_fuc": "MSE",
        "r": 0.5,
        "similar": "MSE",
        "rec_timeseries": True,
        "model_save_path": str(ROOT / "checkpoints" / args.dataset),
        "result_path": str(ROOT / "results" / f"{args.dataset}_ASCA_SCORE_ONLY_FAST"),
        "use_gpu": torch.cuda.is_available(),
        "use_multi_gpu": False,
        "gpu": 0,
        "devices": "0",
    }


def load_checkpoint(solver: AdaptiveSparseAnchorSolverV4, checkpoint: str | None) -> str:
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
        payload = torch.load(str(ckpt_path), map_location=solver.device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        solver.model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")
        return str(ckpt_path)

    solver.load_checkpoint()
    return str(solver.checkpoint_path)


@torch.inference_mode()
def asca_score_only(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """只计算原版 forward 中的 score_combined，不构建 details dict。"""
    if x.ndim != 3:
        raise ValueError("输入必须为 [B,L,M]。")

    sketch = model._sequence_sketch(x)
    local = model._candidate_group(x, sketch, model.local_lags, group_value=0.0)
    global_ = model._candidate_group(x, sketch, model.global_lags, group_value=1.0)

    local_gate, local_weights, _ = model._straight_through_topk(local["logits"], model.local_topk)
    global_gate, global_weights, _ = model._straight_through_topk(global_["logits"], model.global_topk)

    center_affinity = torch.ones_like(local["affinity"][..., :1])
    joint_affinity = torch.cat(
        [center_affinity, local["affinity"], global_["affinity"]], dim=-1
    )
    joint_target = joint_affinity / joint_affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    local_count = model.local_lags.numel()
    target_center = joint_target[..., :1]
    target_local = joint_target[..., 1 : 1 + local_count]
    target_global = joint_target[..., 1 + local_count :]

    local_summary = model._weighted_summary(
        target_local, local["lag_norm"], local_gate, local_weights
    )
    global_summary = model._weighted_summary(
        target_global, global_["lag_norm"], global_gate, global_weights
    )
    sigma = model.fitter(torch.cat([local_summary, global_summary], dim=-1))

    sqrt_two_pi = math.sqrt(2.0 * math.pi)
    center_pdf = 1.0 / (sqrt_two_pi * sigma).unsqueeze(-1)
    local_pdf = 2.0 / (sqrt_two_pi * sigma.unsqueeze(-1)) * torch.exp(
        -0.5 * (local["lag_norm"] / sigma.unsqueeze(-1)).square()
    )
    global_pdf = 2.0 / (sqrt_two_pi * sigma.unsqueeze(-1)) * torch.exp(
        -0.5 * (global_["lag_norm"] / sigma.unsqueeze(-1)).square()
    )
    joint_pdf = torch.cat([center_pdf, local_pdf, global_pdf], dim=-1)
    joint_gaussian = joint_pdf / joint_pdf.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    pred_center = joint_gaussian[..., :1]
    pred_local = joint_gaussian[..., 1 : 1 + local_count]
    pred_global = joint_gaussian[..., 1 + local_count :]

    center_weight = local_weights.new_full(
        (*local_weights.shape[:-1], 1), 1.0 / float(model.local_topk + 1)
    )
    local_pair_weights = local_weights * (float(model.local_topk) / float(model.local_topk + 1))
    local_fit = (
        center_weight.squeeze(-1) * (target_center.squeeze(-1) - pred_center.squeeze(-1)).square()
        + model._weighted_fit_error(target_local, pred_local, local_pair_weights)
    )
    global_fit = model._weighted_fit_error(target_global, pred_global, global_weights)

    score_gap = torch.abs(local_fit - global_fit)
    score_total = local_fit + global_fit
    score_combined = score_total + model.gap_weight * score_gap
    return score_combined


@torch.inference_mode()
def original_score_only(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    _, details = model(x)
    return details["score_combined"]


def normalize_score(score: torch.Tensor, score_normalization: str) -> torch.Tensor:
    if score_normalization == "raw":
        return score
    if score_normalization == "official":
        minimum = score.min(dim=-1, keepdim=True).values
        maximum = score.max(dim=-1, keepdim=True).values
        scaled = (score - minimum) / (maximum - minimum + 1e-5)
        return torch.softmax(scaled, dim=-1)
    raise ValueError(f"未知 score_normalization: {score_normalization}")


@torch.inference_mode()
def prepare_input(solver: AdaptiveSparseAnchorSolverV4, input_data: torch.Tensor) -> torch.Tensor:
    return solver._prepare_input(input_data)


@torch.inference_mode()
def collect_scores(
    solver: AdaptiveSparseAnchorSolverV4,
    score_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    loader: Iterable,
    score_normalization: str,
    collect_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    solver.model.eval()
    scores = []
    labels_all = []
    for input_data, labels in loader:
        x = prepare_input(solver, input_data)
        raw = score_fn(solver.model, x)
        score = normalize_score(raw, score_normalization)
        scores.append(score.detach().cpu().numpy().reshape(-1))
        if collect_labels:
            labels_all.append(labels.detach().cpu().numpy().reshape(-1))
    score_arr = np.concatenate(scores, axis=0)
    label_arr = np.concatenate(labels_all, axis=0).astype(int) if collect_labels else None
    return score_arr, label_arr


def evaluate_scores(train_scores: np.ndarray, test_scores: np.ndarray, gt: np.ndarray, anormly_ratio: float) -> dict:
    threshold_scores = np.concatenate([train_scores, test_scores], axis=0)
    threshold = float(np.percentile(threshold_scores, 100.0 - float(anormly_ratio)))
    pred = (test_scores > threshold).astype(int)

    raw_accuracy = float(accuracy_score(gt, pred))
    raw_precision, raw_recall, raw_f1, _ = precision_recall_fscore_support(
        gt, pred, average="binary", zero_division=0
    )
    raw_mcc = float(matthews_corrcoef(gt, pred))

    adjusted = point_adjust(pred, gt)
    pa_accuracy = float(accuracy_score(gt, adjusted))
    pa_precision, pa_recall, pa_f1, _ = precision_recall_fscore_support(
        gt, adjusted, average="binary", zero_division=0
    )
    pa_mcc = float(matthews_corrcoef(gt, adjusted))

    return {
        "threshold": threshold,
        "raw_accuracy": raw_accuracy,
        "raw_precision": float(raw_precision),
        "raw_recall": float(raw_recall),
        "raw_f1": float(raw_f1),
        "raw_mcc": raw_mcc,
        "pa_accuracy": pa_accuracy,
        "pa_precision": float(pa_precision),
        "pa_recall": float(pa_recall),
        "pa_f1": float(pa_f1),
        "pa_mcc": pa_mcc,
    }


@torch.inference_mode()
def equivalence_check(solver: AdaptiveSparseAnchorSolverV4, max_batches: int) -> dict:
    solver.model.eval()
    max_abs = 0.0
    max_rel = 0.0
    checked = 0
    for input_data, _ in solver.thre_loader:
        x = prepare_input(solver, input_data)
        s1 = original_score_only(solver.model, x)
        s2 = asca_score_only(solver.model, x)
        abs_diff = (s1 - s2).abs()
        rel_diff = abs_diff / s1.abs().clamp_min(1e-12)
        max_abs = max(max_abs, float(abs_diff.max().detach().cpu().item()))
        max_rel = max(max_rel, float(rel_diff.max().detach().cpu().item()))
        checked += 1
        if checked >= max_batches:
            break
    return {"checked_batches": checked, "max_abs_diff": max_abs, "max_rel_diff": max_rel}


@torch.inference_mode()
def benchmark_batch(
    solver: AdaptiveSparseAnchorSolverV4,
    score_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    raw_batch: torch.Tensor,
    batch_size: int,
    score_normalization: str,
    warmup: int,
    repeats: int,
) -> dict:
    solver.model.eval()
    batch = raw_batch[:batch_size].contiguous()
    if batch.shape[0] < batch_size:
        raise RuntimeError(f"Batch too small: got {batch.shape[0]}, need {batch_size}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        x = prepare_input(solver, batch)
        score = score_fn(solver.model, x)
        _ = normalize_score(score, score_normalization)
    sync()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        x = prepare_input(solver, batch)
        score = score_fn(solver.model, x)
        _ = normalize_score(score, score_normalization)
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
    }
    if torch.cuda.is_available():
        out["gpu_peak_allocated_mib"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        out["gpu_peak_reserved_mib"] = float(torch.cuda.max_memory_reserved() / 1024 / 1024)
    return out


@torch.inference_mode()
def benchmark_full_test(
    solver: AdaptiveSparseAnchorSolverV4,
    score_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    score_normalization: str,
    repeats: int,
) -> dict:
    solver.model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    seconds = []
    points_list = []
    for _ in range(repeats):
        points = 0
        t0 = time.perf_counter()
        for input_data, _ in solver.thre_loader:
            x = prepare_input(solver, input_data)
            score = score_fn(solver.model, x)
            score = normalize_score(score, score_normalization)
            points += int(score.numel())
        sync()
        seconds.append(time.perf_counter() - t0)
        points_list.append(points)

    arr = np.asarray(seconds, dtype=np.float64)
    points = int(np.mean(points_list))
    out = {
        "full_test_repeats": int(repeats),
        "full_test_seconds_mean": float(arr.mean()),
        "full_test_seconds_std": float(arr.std()),
        "full_test_seconds_p50": float(np.percentile(arr, 50)),
        "full_test_points": points,
        "full_test_points_per_second": float(points / arr.mean()),
    }
    if torch.cuda.is_available():
        out["gpu_peak_allocated_mib"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        out["gpu_peak_reserved_mib"] = float(torch.cuda.max_memory_reserved() / 1024 / 1024)
    return out


def speedup(old: float, new: float) -> float:
    if new <= 0:
        return float("nan")
    return float(old / new)


def write_outputs(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    p = metrics["params"]
    eq = metrics["equivalence"]
    orig_eval = metrics["evaluation"]["original"]
    fast_eval = metrics["evaluation"]["score_only"]
    lat_o = metrics["latency"]["original"]
    lat_f = metrics["latency"]["score_only"]

    lines = []
    lines.append(f"# ASCA-AD Score-only Fast Inference on {metrics['dataset']}\n")
    lines.append("## Checkpoint\n")
    lines.append(f"`{metrics['checkpoint']}`\n")

    lines.append("## Parameters\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Trainable Params | {p['trainable_params']} |")
    lines.append(f"| Total Params | {p['total_params']} |")
    lines.append(f"| Param Size MiB | {p['param_mib']:.6f} |")
    lines.append("")

    lines.append("## Equivalence Check\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Checked Batches | {eq['checked_batches']} |")
    lines.append(f"| Max Abs Diff | {eq['max_abs_diff']:.12g} |")
    lines.append(f"| Max Rel Diff | {eq['max_rel_diff']:.12g} |")
    lines.append("")

    lines.append("## Detection Metrics\n")
    lines.append("| Path | PA Precision | PA Recall | PA F1 | PA Accuracy | RAW F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Original forward | {orig_eval['pa_precision']:.6f} | {orig_eval['pa_recall']:.6f} | "
        f"{orig_eval['pa_f1']:.6f} | {orig_eval['pa_accuracy']:.6f} | {orig_eval['raw_f1']:.6f} |"
    )
    lines.append(
        f"| Score-only fast | {fast_eval['pa_precision']:.6f} | {fast_eval['pa_recall']:.6f} | "
        f"{fast_eval['pa_f1']:.6f} | {fast_eval['pa_accuracy']:.6f} | {fast_eval['raw_f1']:.6f} |"
    )
    lines.append("")

    lines.append("## Latency / Throughput\n")
    lines.append("| Metric | Original forward | Score-only fast | Speedup |")
    lines.append("|---|---:|---:|---:|")
    for key, label in [("batch_1", "Batch=1 Mean ms"), ("batch_n", f"Batch={lat_o['batch_n']['batch_size']} Mean ms")]:
        o = lat_o[key]["latency_ms_mean"]
        f = lat_f[key]["latency_ms_mean"]
        lines.append(f"| {label} | {o:.6f} | {f:.6f} | {speedup(o, f):.3f}x |")
    o = lat_o["full_test"]["full_test_seconds_mean"]
    f = lat_f["full_test"]["full_test_seconds_mean"]
    lines.append(f"| Full-test Seconds | {o:.6f} | {f:.6f} | {speedup(o, f):.3f}x |")
    o = lat_o["full_test"]["full_test_points_per_second"]
    f = lat_f["full_test"]["full_test_points_per_second"]
    lines.append(f"| Full-test Throughput points/s | {o:.2f} | {f:.2f} | {speedup(f, o):.3f}x |")
    if "gpu_peak_allocated_mib" in lat_f["full_test"]:
        lines.append(
            f"| Full-test GPU Peak Allocated MiB | {lat_o['full_test']['gpu_peak_allocated_mib']:.2f} | "
            f"{lat_f['full_test']['gpu_peak_allocated_mib']:.2f} | - |"
        )
        lines.append(
            f"| Full-test GPU Peak Reserved MiB | {lat_o['full_test']['gpu_peak_reserved_mib']:.2f} | "
            f"{lat_f['full_test']['gpu_peak_reserved_mib']:.2f} | - |"
        )
    lines.append(f"| CPU Max RSS MiB | {metrics['cpu_max_rss_mib']:.2f} | {metrics['cpu_max_rss_mib']:.2f} | - |")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_dir / 'summary.json'}")
    print(f"[saved] {out_dir / 'summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--anormly-ratio", dest="anormly_ratio", type=float, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--relation-input", choices=["instance", "standardized"], default="instance")
    parser.add_argument("--score-normalization", choices=["official", "raw"], default="official")
    parser.add_argument("--threshold-source", choices=["original"], default="original")

    parser.add_argument("--local-lags", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--global-lags", type=int, nargs="+", default=[12, 16, 20, 24, 28, 32, 40, 48])
    parser.add_argument("--local-topk", type=int, default=2)
    parser.add_argument("--global-topk", type=int, default=4)
    parser.add_argument("--selector-hidden", type=int, default=8)
    parser.add_argument("--fitter-hidden", type=int, default=8)
    parser.add_argument("--selector-temperature", type=float, default=0.5)
    parser.add_argument("--similarity-tau", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.03)
    parser.add_argument("--sigma-max", type=float, default=1.50)
    parser.add_argument("--gap-weight", type=float, default=1.0)

    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--full-test-repeats", type=int, default=3)
    parser.add_argument("--equiv-batches", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[root] {ROOT}")
    print(f"[out_dir] {out_dir}")

    solver = AdaptiveSparseAnchorSolverV4(build_solver_config(args))
    checkpoint = load_checkpoint(solver, args.checkpoint)
    solver.model.eval()

    params = count_params(solver.model)
    print("[params]", json.dumps(params, indent=2))

    eq = equivalence_check(solver, args.equiv_batches)
    print("[equivalence]", json.dumps(eq, indent=2))

    train_o, _ = collect_scores(solver, original_score_only, solver.train_loader, args.score_normalization)
    test_o, gt = collect_scores(solver, original_score_only, solver.thre_loader, args.score_normalization, collect_labels=True)
    train_f, _ = collect_scores(solver, asca_score_only, solver.train_loader, args.score_normalization)
    test_f, gt2 = collect_scores(solver, asca_score_only, solver.thre_loader, args.score_normalization, collect_labels=True)
    assert gt is not None and gt2 is not None
    if not np.array_equal(gt, gt2):
        raise RuntimeError("Original/Fast labels are different; this should not happen.")

    eval_o = evaluate_scores(train_o, test_o, gt, args.anormly_ratio)
    eval_f = evaluate_scores(train_f, test_f, gt, args.anormly_ratio)

    first_batch = next(iter(solver.thre_loader))[0]
    batch_n = min(args.batch_size, int(first_batch.shape[0]))

    lat_o = {
        "batch_1": benchmark_batch(solver, original_score_only, first_batch, 1, args.score_normalization, args.warmup, args.repeats),
        "batch_n": benchmark_batch(solver, original_score_only, first_batch, batch_n, args.score_normalization, args.warmup, args.repeats),
        "full_test": benchmark_full_test(solver, original_score_only, args.score_normalization, args.full_test_repeats),
    }
    lat_f = {
        "batch_1": benchmark_batch(solver, asca_score_only, first_batch, 1, args.score_normalization, args.warmup, args.repeats),
        "batch_n": benchmark_batch(solver, asca_score_only, first_batch, batch_n, args.score_normalization, args.warmup, args.repeats),
        "full_test": benchmark_full_test(solver, asca_score_only, args.score_normalization, args.full_test_repeats),
    }

    metrics = {
        "dataset": args.dataset,
        "checkpoint": checkpoint,
        "score_normalization": args.score_normalization,
        "anormly_ratio": args.anormly_ratio,
        "params": params,
        "equivalence": eq,
        "evaluation": {"original": eval_o, "score_only": eval_f},
        "latency": {"original": lat_o, "score_only": lat_f},
        "cpu_max_rss_mib": max_rss_mib(),
    }

    write_outputs(metrics, out_dir)
    print("\n[done]")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
