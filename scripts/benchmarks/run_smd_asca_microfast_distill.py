#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASCA-AD-MicroFast: parameter-budgeted distilled fast model for SMD.

目标：
- 允许比 ASCA-AD 原版 146 参数略多；
- 但必须小于 PPLAD 的 2,561 参数；
- 用 depthwise-separable Conv1d（深度可分离卷积）替代普通 Conv1d，避免参数量涨到几万；
- 用原 ASCA-AD V4 作为 teacher（教师模型），训练 MicroFast student（学生模型）拟合 teacher 的 combined score；
- 训练过程不使用标签，标签只用于最终评价。

推荐参数：
    hidden=32, kernel_size=9
    参数量约 1.6K，小于 PPLAD 2,561。

放置位置：
    /mnt/c/Users/DING/Desktop/Experiment/CODE/scripts/benchmarks/run_smd_asca_microfast_distill.py

推荐运行：
    CUDA_VISIBLE_DEVICES=0 python -u scripts/benchmarks/run_smd_asca_microfast_distill.py \
      --teacher-checkpoint checkpoints/SMD/SMD_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_g12-16-20-24-28-32-40-48_kl2_kg4.pt \
      --epochs 10 \
      --hidden 32 \
      --kernel-size 9 \
      --batch-size 128 \
      --seq-len 100 \
      --channels 38 \
      --anormly-ratio 0.9 \
      --out-dir results/SMD_ASCA_MICROFAST_H32_K9 | tee logs_SMD_ASCA_MICROFAST_H32_K9.txt
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
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asca_ad.model import AdaptiveSparseAnchorCompetitiveModelV4, AdaptiveSparseAnchorSolverV4, set_seed  # noqa: E402


PPLAD_PARAM_BUDGET = 2561


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


def parse_lags_from_checkpoint_name(path: str) -> Tuple[List[int], List[int], int, int]:
    name = Path(path).name
    try:
        l_part = name.split("_l", 1)[1].split("_g", 1)[0]
        g_part = name.split("_g", 1)[1].split("_kl", 1)[0]
        kl_part = name.split("_kl", 1)[1].split("_kg", 1)[0]
        kg_part = name.split("_kg", 1)[1].split(".pt", 1)[0]
        local_lags = [int(v) for v in l_part.split("-") if v]
        global_lags = [int(v) for v in g_part.split("-") if v]
        return local_lags, global_lags, int(kl_part), int(kg_part)
    except Exception as exc:
        raise ValueError(f"无法从 checkpoint 文件名解析 lags/topk: {name}") from exc


def score_normalize(score: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return score
    if mode == "official":
        minimum = score.min(dim=-1, keepdim=True).values
        maximum = score.max(dim=-1, keepdim=True).values
        scaled = (score - minimum) / (maximum - minimum + 1e-5)
        return torch.softmax(scaled, dim=-1)
    raise ValueError(f"Unknown score normalization: {mode}")


class MicroFastScoreStudent(nn.Module):
    """
    小参数量 GPU-friendly score head。

    结构：
        depthwise temporal conv:
            Conv1d(C -> C, kernel_size=k, groups=C)
        pointwise channel mixer:
            Conv1d(C -> hidden, kernel_size=1)
        GELU
        pointwise output:
            Conv1d(hidden -> 1, kernel_size=1)

    参数量近似：
        C*k + C              # depthwise
      + C*hidden + hidden    # pointwise in
      + hidden + 1           # pointwise out
    SMD: C=38, hidden=32, k=9
        38*9+38 + 38*32+32 + 32+1 = 1661
    小于 PPLAD 2561。
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 32,
        kernel_size: int = 9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size 建议使用奇数，例如 5/7/9/11。")
        if hidden <= 0:
            raise ValueError("hidden 必须大于 0。")

        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=True,
            ),
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Conv1d(hidden, 1, kernel_size=1, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, C, L] -> [B, L]
        return self.net(x.transpose(1, 2)).squeeze(1)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "param_bytes": int(param_bytes),
        "param_mib": float(param_bytes / 1024 / 1024),
    }


def build_teacher_kwargs(args: argparse.Namespace) -> dict:
    local_lags, global_lags, local_topk, global_topk = parse_lags_from_checkpoint_name(
        args.teacher_checkpoint
    )
    return {
        "local_candidate_lags": local_lags,
        "global_candidate_lags": global_lags,
        "local_topk": local_topk,
        "global_topk": global_topk,
        "selector_hidden": args.selector_hidden,
        "fitter_hidden": args.fitter_hidden,
        "selector_temperature": args.selector_temperature,
        "similarity_tau": args.similarity_tau,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "gap_weight": args.gap_weight,
    }


def build_solver_config(args: argparse.Namespace, teacher_kwargs: dict) -> dict:
    return {
        "dataset": "SMD",
        "data_path": "SMD",
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

        "local_candidate_lags": teacher_kwargs["local_candidate_lags"],
        "global_candidate_lags": teacher_kwargs["global_candidate_lags"],
        "local_topk": teacher_kwargs["local_topk"],
        "global_topk": teacher_kwargs["global_topk"],
        "selector_hidden": teacher_kwargs["selector_hidden"],
        "fitter_hidden": teacher_kwargs["fitter_hidden"],
        "selector_temperature": teacher_kwargs["selector_temperature"],
        "similarity_tau": teacher_kwargs["similarity_tau"],
        "sigma_min": teacher_kwargs["sigma_min"],
        "sigma_max": teacher_kwargs["sigma_max"],
        "area_weight": 0.1,
        "selector_balance_weight": 0.05,
        "gap_weight": teacher_kwargs["gap_weight"],
        "relation_input": args.relation_input,

        "score_modes": ["combined"],
        "primary_score": "combined",
        "score_normalization": args.score_normalization,
        "threshold_source": args.threshold_source,
        "quantile_method": "exact",
        "quantile_buffer": 50000,

        # 兼容父 Solver 初始化；V4 前向结构不使用这些 PPLAD 字段。
        "local_size": 7,
        "global_size": [11],
        "d_model": 8,
        "loss_fuc": "MSE",
        "r": 0.5,
        "similar": "MSE",
        "rec_timeseries": True,

        "model_save_path": str(ROOT / "checkpoints" / "SMD_ASCA_MICROFAST" / "TEACHER"),
        "result_path": str(ROOT / "results" / "SMD_ASCA_MICROFAST" / "TEACHER"),
        "use_gpu": torch.cuda.is_available(),
        "use_multi_gpu": False,
        "gpu": 0,
        "devices": "0",
    }


def load_teacher(args: argparse.Namespace) -> Tuple[AdaptiveSparseAnchorCompetitiveModelV4, dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_kwargs = build_teacher_kwargs(args)
    teacher = AdaptiveSparseAnchorCompetitiveModelV4(**teacher_kwargs).to(device)

    ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    teacher.load_state_dict(state)
    teacher.eval()
    return teacher, teacher_kwargs


def prepare_input(solver: AdaptiveSparseAnchorSolverV4, input_data: torch.Tensor) -> torch.Tensor:
    return solver._prepare_input(input_data)


def teacher_score(
    teacher: AdaptiveSparseAnchorCompetitiveModelV4,
    x: torch.Tensor,
    normalization: str,
) -> torch.Tensor:
    _, details = teacher(x)
    return score_normalize(details["score_combined"], normalization)


def train_student(
    args: argparse.Namespace,
    solver: AdaptiveSparseAnchorSolverV4,
    teacher: AdaptiveSparseAnchorCompetitiveModelV4,
    student: MicroFastScoreStudent,
) -> list[dict]:
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []

    print("[training MicroFast student]")
    for epoch in range(1, args.epochs + 1):
        student.train()
        teacher.eval()
        t0 = time.perf_counter()
        losses = []

        for input_data, _ in solver.train_loader:
            x = prepare_input(solver, input_data)

            with torch.no_grad():
                target = teacher_score(
                    teacher,
                    x,
                    normalization=args.teacher_score_normalization,
                ).detach().clone()

            raw = student(x)
            pred = score_normalize(raw, args.student_score_normalization)

            loss_mse = F.mse_loss(pred, target)
            loss = loss_mse * args.loss_scale

            # 可选：防止 score 过度平滑。默认关闭。
            if args.variance_weight > 0:
                loss = loss - args.variance_weight * pred.var(dim=-1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss_mse.detach().cpu().item()))

        sync()
        elapsed = time.perf_counter() - t0
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        record = {"epoch": epoch, "teacher_mse": avg_loss, "seconds": elapsed}
        history.append(record)
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] teacher_mse={avg_loss:.10f}, time={elapsed:.3f}s")

    return history


@torch.inference_mode()
def collect_student_scores(
    solver: AdaptiveSparseAnchorSolverV4,
    student: nn.Module,
    loader: Iterable,
    normalization: str,
    collect_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    student.eval()
    score_parts = []
    label_parts = []

    for input_data, labels in loader:
        x = prepare_input(solver, input_data)
        raw = student(x)
        score = score_normalize(raw, normalization)
        score_parts.append(score.detach().cpu().numpy().reshape(-1))
        if collect_labels:
            label_parts.append(labels.detach().cpu().numpy().reshape(-1))

    scores = np.concatenate(score_parts, axis=0)
    labels_out = np.concatenate(label_parts, axis=0).astype(int) if collect_labels else None
    return scores, labels_out


@torch.inference_mode()
def evaluate_student(
    args: argparse.Namespace,
    solver: AdaptiveSparseAnchorSolverV4,
    student: nn.Module,
) -> dict:
    train_scores, _ = collect_student_scores(
        solver,
        student,
        solver.train_loader,
        normalization=args.student_score_normalization,
        collect_labels=False,
    )

    if args.threshold_source == "train":
        threshold_scores = train_scores
    elif args.threshold_source == "original":
        test_scores_for_threshold, _ = collect_student_scores(
            solver,
            student,
            solver.thre_loader,
            normalization=args.student_score_normalization,
            collect_labels=False,
        )
        threshold_scores = np.concatenate([train_scores, test_scores_for_threshold], axis=0)
    else:
        raise ValueError(f"Unknown threshold_source: {args.threshold_source}")

    threshold = float(np.percentile(threshold_scores, 100.0 - args.anormly_ratio))

    test_scores, gt = collect_student_scores(
        solver,
        student,
        solver.thre_loader,
        normalization=args.student_score_normalization,
        collect_labels=True,
    )
    if gt is None:
        raise RuntimeError("gt is None.")

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
        "threshold_samples": int(threshold_scores.size),
        "test_points": int(test_scores.size),
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
def benchmark_batch(
    solver: AdaptiveSparseAnchorSolverV4,
    model: nn.Module,
    raw_batch: torch.Tensor,
    batch_size: int,
    normalization: str,
    warmup: int,
    repeats: int,
) -> dict:
    model.eval()
    batch = raw_batch[:batch_size].contiguous()
    if batch.shape[0] < batch_size:
        raise RuntimeError(f"Batch too small: got {batch.shape[0]}, need {batch_size}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        x = prepare_input(solver, batch)
        raw = model(x)
        _ = score_normalize(raw, normalization)
    sync()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        x = prepare_input(solver, batch)
        raw = model(x)
        _ = score_normalize(raw, normalization)
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
        out["gpu_peak_allocated_mib"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        out["gpu_peak_reserved_mib"] = float(torch.cuda.max_memory_reserved() / 1024 / 1024)
    return out


@torch.inference_mode()
def benchmark_full_test(
    solver: AdaptiveSparseAnchorSolverV4,
    model: nn.Module,
    normalization: str,
    repeats: int,
) -> dict:
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    seconds = []
    points_list = []

    for _ in range(repeats):
        total_points = 0
        t0 = time.perf_counter()

        for input_data, _ in solver.thre_loader:
            x = prepare_input(solver, input_data)
            raw = model(x)
            score = score_normalize(raw, normalization)
            total_points += int(score.numel())

        sync()
        seconds.append(time.perf_counter() - t0)
        points_list.append(total_points)

    arr = np.asarray(seconds, dtype=np.float64)
    points = int(np.mean(points_list))
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
        out["gpu_peak_allocated_mib"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        out["gpu_peak_reserved_mib"] = float(torch.cuda.max_memory_reserved() / 1024 / 1024)
    return out


def save_student_checkpoint(args: argparse.Namespace, student: nn.Module, out_dir: Path) -> str:
    ckpt_path = out_dir / "asca_microfast_student.pt"
    payload = {
        "model": student.state_dict(),
        "config": {
            "channels": args.channels,
            "hidden": args.hidden,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "teacher_checkpoint": args.teacher_checkpoint,
            "teacher_score_normalization": args.teacher_score_normalization,
            "student_score_normalization": args.student_score_normalization,
            "param_budget": PPLAD_PARAM_BUDGET,
        },
    }
    torch.save(payload, ckpt_path)
    return str(ckpt_path)


def write_outputs(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"

    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    ev = metrics["evaluation"]
    b1 = metrics["latency"]["batch_1"]
    bn = metrics["latency"]["batch_n"]
    ft = metrics["latency"]["full_test"]

    lines = []
    lines.append("# ASCA-AD-MicroFast Distillation on SMD\n")

    lines.append("## Parameter Budget\n")
    lines.append(f"- PPLAD parameter budget: `{PPLAD_PARAM_BUDGET}`")
    lines.append(f"- MicroFast trainable params: `{metrics['params']['trainable_params']}`")
    lines.append(f"- Under PPLAD budget: `{metrics['under_pplad_budget']}`")
    lines.append("")

    lines.append("## Config\n")
    lines.append(f"- teacher_checkpoint: `{metrics['teacher_checkpoint']}`")
    lines.append(f"- student_checkpoint: `{metrics['student_checkpoint']}`")
    lines.append(f"- hidden: `{metrics['student_config']['hidden']}`")
    lines.append(f"- kernel_size: `{metrics['student_config']['kernel_size']}`")
    lines.append(f"- train_epochs: `{metrics['train_epochs']}`")
    lines.append(f"- anormly_ratio: `{metrics['anormly_ratio']}`")
    lines.append(f"- teacher_score_normalization: `{metrics['teacher_score_normalization']}`")
    lines.append(f"- student_score_normalization: `{metrics['student_score_normalization']}`")
    lines.append("")

    lines.append("## Accuracy\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Threshold | {ev['threshold']:.10f} |")
    lines.append(f"| RAW Accuracy | {ev['raw_accuracy']:.6f} |")
    lines.append(f"| RAW Precision | {ev['raw_precision']:.6f} |")
    lines.append(f"| RAW Recall | {ev['raw_recall']:.6f} |")
    lines.append(f"| RAW F1 | {ev['raw_f1']:.6f} |")
    lines.append(f"| PA Accuracy | {ev['pa_accuracy']:.6f} |")
    lines.append(f"| PA Precision | {ev['pa_precision']:.6f} |")
    lines.append(f"| PA Recall | {ev['pa_recall']:.6f} |")
    lines.append(f"| PA F1 | {ev['pa_f1']:.6f} |")
    lines.append("")

    lines.append("## Lightweight / Latency\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Trainable Params | {metrics['params']['trainable_params']} |")
    lines.append(f"| Total Params | {metrics['params']['total_params']} |")
    lines.append(f"| Param Size MiB | {metrics['params']['param_mib']:.6f} |")
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

    lines.append("## Reference\n")
    lines.append("- 当前 iTransformer / SMD / batch=128 mean latency 约为 `1.967984 ms`。")
    lines.append("- 若 Batch=128 Mean Latency ms 小于该值，则 ASCA-AD-MicroFast 在该口径下超过 iTransformer。")
    lines.append("- 同时要求 Trainable Params 小于 PPLAD 的 2,561，避免失去轻量化优势。")
    lines.append("")

    lines.append("## Training History\n")
    lines.append("```json")
    lines.append(json.dumps(metrics["training_history"], ensure_ascii=False, indent=2))
    lines.append("```")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--channels", type=int, default=38)
    parser.add_argument("--anormly-ratio", dest="anormly_ratio", type=float, default=0.9)
    parser.add_argument("--relation-input", choices=["instance", "standardized"], default="instance")
    parser.add_argument("--threshold-source", choices=["original", "train"], default="original")
    parser.add_argument("--score-normalization", choices=["official", "raw"], default="official")

    # teacher ASCA-AD architecture kwargs
    parser.add_argument("--selector-hidden", type=int, default=8)
    parser.add_argument("--fitter-hidden", type=int, default=8)
    parser.add_argument("--selector-temperature", type=float, default=0.5)
    parser.add_argument("--similarity-tau", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.03)
    parser.add_argument("--sigma-max", type=float, default=1.50)
    parser.add_argument("--gap-weight", type=float, default=1.0)

    # MicroFast student config
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.0)

    # distillation config
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-scale", type=float, default=1000.0)
    parser.add_argument("--variance-weight", type=float, default=0.0)
    parser.add_argument("--teacher-score-normalization", choices=["official", "raw"], default="official")
    parser.add_argument("--student-score-normalization", choices=["official", "raw"], default="official")

    # benchmark
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--full-test-repeats", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    args.teacher_checkpoint = (
        str((ROOT / args.teacher_checkpoint).resolve())
        if not os.path.isabs(args.teacher_checkpoint)
        else args.teacher_checkpoint
    )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[root] {ROOT}")
    print(f"[teacher_checkpoint] {args.teacher_checkpoint}")
    print(f"[out_dir] {out_dir}")

    teacher, teacher_kwargs = load_teacher(args)
    cfg = build_solver_config(args, teacher_kwargs)
    solver = AdaptiveSparseAnchorSolverV4(cfg)
    solver.model.load_state_dict(teacher.state_dict())
    solver.model.eval()

    student = MicroFastScoreStudent(
        channels=args.channels,
        hidden=args.hidden,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(solver.device)

    params = count_params(student)
    print("[student params]", json.dumps(params, indent=2))
    if params["trainable_params"] > PPLAD_PARAM_BUDGET:
        raise RuntimeError(
            f"MicroFast 参数量 {params['trainable_params']} 超过 PPLAD 参数预算 {PPLAD_PARAM_BUDGET}。"
            "请降低 --hidden 或 --kernel-size。"
        )

    history = train_student(args, solver, teacher, student)

    if args.compile:
        print("[compile] trying torch.compile(student)")
        try:
            student = torch.compile(student)  # type: ignore[assignment]
        except Exception as exc:
            print(f"[compile failed] {exc}")

    evaluation = evaluate_student(args, solver, student)

    first_batch = next(iter(solver.thre_loader))[0]

    b1 = benchmark_batch(
        solver,
        student,
        first_batch,
        1,
        normalization=args.student_score_normalization,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    bn = benchmark_batch(
        solver,
        student,
        first_batch,
        min(args.batch_size, int(first_batch.shape[0])),
        normalization=args.student_score_normalization,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    ft = benchmark_full_test(
        solver,
        student,
        normalization=args.student_score_normalization,
        repeats=args.full_test_repeats,
    )

    save_model = student._orig_mod if hasattr(student, "_orig_mod") else student  # type: ignore[attr-defined]
    student_ckpt = save_student_checkpoint(args, save_model, out_dir)

    metrics = {
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_checkpoint": student_ckpt,
        "student_config": {
            "hidden": args.hidden,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
        },
        "teacher_score_normalization": args.teacher_score_normalization,
        "student_score_normalization": args.student_score_normalization,
        "threshold_source": args.threshold_source,
        "anormly_ratio": args.anormly_ratio,
        "train_epochs": args.epochs,
        "training_history": history,
        "evaluation": evaluation,
        "params": count_params(save_model),
        "under_pplad_budget": count_params(save_model)["trainable_params"] < PPLAD_PARAM_BUDGET,
        "latency": {
            "batch_1": b1,
            "batch_n": bn,
            "full_test": ft,
        },
        "cpu_max_rss_mib": max_rss_mib(),
    }

    write_outputs(metrics, out_dir)

    print("\n[done]")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
