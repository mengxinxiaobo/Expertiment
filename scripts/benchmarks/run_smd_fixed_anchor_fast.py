#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-anchor fast inference test for ASCA-AD V4 on SMD.

核心思想：
1. 读取已经训练好的原始 ASCA-AD V4 checkpoint。
2. 在训练集上统计原模型最常选中的 local/global anchor lag。
3. 推理时固定这些 lag，跳过 selector / softmax / top-k。
4. 重新计算 fixed-combined 协议下的 PA-F1，并测 batch=1、batch=128、full-test 推理速度。

放置位置建议：
    /mnt/c/Users/DING/Desktop/Experiment/CODE/scripts/benchmarks/run_smd_fixed_anchor_fast.py

推荐命令：
    CUDA_VISIBLE_DEVICES=0 python -u scripts/benchmarks/run_smd_fixed_anchor_fast.py \
      --dataset SMD \
      --checkpoint checkpoints/SMD/SMD_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_g12-16-20-24-28-32-40-48_kl2_kg4.pt \
      --fixed-local-k 1 \
      --fixed-global-k 2 \
      --batch-size 128 \
      --seq-len 100 \
      --channels 38 \
      --anormly-ratio 0.9 \
      --out-dir results/SMD_FIXED_ANCHOR_FAST_L1_G2 | tee logs_SMD_FIXED_ANCHOR_FAST_L1_G2.txt
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
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asca_ad.model import AdaptiveSparseAnchorCompetitiveModelV4, AdaptiveSparseAnchorSolverV4, set_seed  # noqa: E402


def max_rss_mib() -> float:
    """Linux/WSL 下 ru_maxrss 的单位是 KB。"""
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
    """
    从文件名中解析：
    SMD_adaptive_anchor_v4_l1-2-3_g12-24-48_kl1_kg2.pt
    """
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


def build_solver_config(args: argparse.Namespace, model_kwargs: dict) -> dict:
    tag = "SMD_FIXED_ANCHOR_FAST"
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

        "local_candidate_lags": model_kwargs["local_candidate_lags"],
        "global_candidate_lags": model_kwargs["global_candidate_lags"],
        "local_topk": model_kwargs["local_topk"],
        "global_topk": model_kwargs["global_topk"],
        "selector_hidden": model_kwargs["selector_hidden"],
        "fitter_hidden": model_kwargs["fitter_hidden"],
        "selector_temperature": model_kwargs["selector_temperature"],
        "similarity_tau": model_kwargs["similarity_tau"],
        "sigma_min": model_kwargs["sigma_min"],
        "sigma_max": model_kwargs["sigma_max"],
        "area_weight": 0.1,
        "selector_balance_weight": 0.05,
        "gap_weight": model_kwargs["gap_weight"],
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

        "model_save_path": str(ROOT / "checkpoints" / tag / "V4"),
        "result_path": str(ROOT / "results" / tag / "V4"),
        "use_gpu": torch.cuda.is_available(),
        "use_multi_gpu": False,
        "gpu": 0,
        "devices": "0",
    }


def load_original_model(args: argparse.Namespace) -> Tuple[AdaptiveSparseAnchorCompetitiveModelV4, dict]:
    local_lags, global_lags, local_topk, global_topk = parse_lags_from_checkpoint_name(args.checkpoint)

    model_kwargs = {
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AdaptiveSparseAnchorCompetitiveModelV4(**model_kwargs).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    return model, model_kwargs


class FixedAnchorFastModel(nn.Module):
    """
    ASCA-AD-Fast 推理近似版：
    - 使用原模型训练好的 fitter；
    - 固定 local/global lags；
    - 跳过 selector、selector feature、softmax、top-k 和 gate；
    - 只输出 combined score 所需的最小 details。
    """

    def __init__(
        self,
        base_model: AdaptiveSparseAnchorCompetitiveModelV4,
        fixed_local_lags: List[int],
        fixed_global_lags: List[int],
    ) -> None:
        super().__init__()
        if not fixed_local_lags or not fixed_global_lags:
            raise ValueError("fixed_local_lags 和 fixed_global_lags 都不能为空。")

        self.fitter = base_model.fitter
        self.similarity_tau = float(base_model.similarity_tau)
        self.gap_weight = float(base_model.gap_weight)
        self.max_lag = float(base_model.max_lag)
        self.local_edge = float(base_model.local_edge)

        self.register_buffer("local_lags", torch.tensor(fixed_local_lags, dtype=torch.long))
        self.register_buffer("global_lags", torch.tensor(fixed_global_lags, dtype=torch.long))

    @staticmethod
    def _symmetric_gather(x: torch.Tensor, lags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length = x.shape[:2]
        time_index = torch.arange(length, device=x.device).view(1, length, 1)
        lag_view = lags.view(1, 1, -1)
        left_index = (time_index - lag_view).clamp(0, length - 1).expand(batch, -1, -1)
        right_index = (time_index + lag_view).clamp(0, length - 1).expand(batch, -1, -1)
        batch_index = torch.arange(batch, device=x.device).view(batch, 1, 1).expand_as(left_index)
        return x[batch_index, left_index], x[batch_index, right_index]

    def _candidate_affinity(self, x: torch.Tensor, lags: torch.Tensor) -> Dict[str, torch.Tensor]:
        left_x, right_x = self._symmetric_gather(x, lags)
        current_x = x.unsqueeze(2)

        left_distance = (current_x - left_x).square().mean(dim=-1)
        right_distance = (current_x - right_x).square().mean(dim=-1)

        affinity = torch.exp(-left_distance / self.similarity_tau) + torch.exp(
            -right_distance / self.similarity_tau
        )

        batch, length, count = affinity.shape
        lag_norm = (lags.to(x.dtype).view(1, 1, count) / self.max_lag).expand(batch, length, -1)
        weights = torch.full_like(affinity, 1.0 / float(count))
        return {"affinity": affinity, "lag_norm": lag_norm, "weights": weights}

    @staticmethod
    def _weighted_summary(
        target: torch.Tensor,
        lag_norm: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        mean = torch.sum(weights * target, dim=-1)
        variance = torch.sum(weights * (target - mean.unsqueeze(-1)).square(), dim=-1)
        std = torch.sqrt(variance + 1e-8)
        mean_lag = torch.sum(weights * lag_norm, dim=-1)
        selected_mass = torch.sum(target, dim=-1)
        return torch.stack([mean, std, mean_lag, selected_mass], dim=-1)

    @staticmethod
    def _weighted_fit_error(
        target: torch.Tensor,
        prediction: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(weights * (target - prediction).square(), dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError("输入必须是 [B,L,C]。")

        local = self._candidate_affinity(x, self.local_lags)
        global_ = self._candidate_affinity(x, self.global_lags)

        center_affinity = torch.ones_like(local["affinity"][..., :1])
        joint_affinity = torch.cat(
            [center_affinity, local["affinity"], global_["affinity"]], dim=-1
        )
        joint_target = joint_affinity / joint_affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        local_count = self.local_lags.numel()
        target_center = joint_target[..., :1]
        target_local = joint_target[..., 1 : 1 + local_count]
        target_global = joint_target[..., 1 + local_count :]

        local_summary = self._weighted_summary(
            target_local, local["lag_norm"], local["weights"]
        )
        global_summary = self._weighted_summary(
            target_global, global_["lag_norm"], global_["weights"]
        )

        sigma = self.fitter(torch.cat([local_summary, global_summary], dim=-1))

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

        center_weight = local["weights"].new_full(
            (*local["weights"].shape[:-1], 1), 1.0 / float(local_count + 1)
        )
        local_pair_weights = local["weights"] * (float(local_count) / float(local_count + 1))

        local_fit = (
            center_weight.squeeze(-1) * (target_center.squeeze(-1) - pred_center.squeeze(-1)).square()
            + self._weighted_fit_error(target_local, pred_local, local_pair_weights)
        )
        global_fit = self._weighted_fit_error(target_global, pred_global, global_["weights"])

        score_gap = torch.abs(local_fit - global_fit)
        score_total = local_fit + global_fit
        score_combined = score_total + self.gap_weight * score_gap

        details = {
            "score_gap": score_gap,
            "score_total": score_total,
            "score_combined": score_combined,
            "local_fit": local_fit,
            "global_fit": global_fit,
            "sigma": sigma,
        }
        return score_combined, details


@torch.inference_mode()
def choose_fixed_lags(
    solver: AdaptiveSparseAnchorSolverV4,
    fixed_local_k: int,
    fixed_global_k: int,
    max_batches: int | None = None,
) -> Tuple[List[int], List[int], Dict[str, list]]:
    """
    用原始模型在训练集上的 selected_index 统计 anchor 占用率，然后选 Top-K lag。
    """
    solver.model.eval()

    local_count = np.zeros(len(solver.local_candidate_lags), dtype=np.int64)
    global_count = np.zeros(len(solver.global_candidate_lags), dtype=np.int64)

    for batch_idx, (input_data, _) in enumerate(solver.train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        _, details = solver._forward_batch(input_data)

        local_index = details["local_selected_index"].detach().cpu().numpy().reshape(-1)
        global_index = details["global_selected_index"].detach().cpu().numpy().reshape(-1)

        local_count += np.bincount(local_index, minlength=local_count.size)
        global_count += np.bincount(global_index, minlength=global_count.size)

    local_order = np.argsort(-local_count)[:fixed_local_k]
    global_order = np.argsort(-global_count)[:fixed_global_k]

    fixed_local = sorted([solver.local_candidate_lags[int(i)] for i in local_order])
    fixed_global = sorted([solver.global_candidate_lags[int(i)] for i in global_order])

    stats = {
        "local_candidate_lags": list(map(int, solver.local_candidate_lags)),
        "global_candidate_lags": list(map(int, solver.global_candidate_lags)),
        "local_count": local_count.astype(int).tolist(),
        "global_count": global_count.astype(int).tolist(),
        "local_frequency": (local_count / max(local_count.sum(), 1)).astype(float).tolist(),
        "global_frequency": (global_count / max(global_count.sum(), 1)).astype(float).tolist(),
    }
    return fixed_local, fixed_global, stats


def prepare_input(solver: AdaptiveSparseAnchorSolverV4, input_data: torch.Tensor) -> torch.Tensor:
    return solver._prepare_input(input_data)


def score_dict_from_details(details: Dict[str, torch.Tensor], normalization: str) -> Dict[str, torch.Tensor]:
    scores = {
        "combined": details["score_combined"],
    }

    if normalization == "raw":
        return scores

    if normalization == "official":
        normalized = {}
        for name, score in scores.items():
            minimum = score.min(dim=-1, keepdim=True).values
            maximum = score.max(dim=-1, keepdim=True).values
            scaled = (score - minimum) / (maximum - minimum + 1e-5)
            normalized[name] = torch.softmax(scaled, dim=-1)
        return normalized

    raise ValueError(f"未知 score_normalization: {normalization}")


@torch.inference_mode()
def collect_scores(
    fixed_model: FixedAnchorFastModel,
    solver: AdaptiveSparseAnchorSolverV4,
    loader: Iterable,
    normalization: str,
    collect_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    fixed_model.eval()

    score_parts = []
    label_parts = []

    for input_data, labels in loader:
        x = prepare_input(solver, input_data)
        _, details = fixed_model(x)
        score_dict = score_dict_from_details(details, normalization=normalization)
        score_parts.append(score_dict["combined"].detach().cpu().numpy().reshape(-1))
        if collect_labels:
            label_parts.append(labels.detach().cpu().numpy().reshape(-1))

    scores = np.concatenate(score_parts, axis=0)
    labels_out = np.concatenate(label_parts, axis=0).astype(int) if collect_labels else None
    return scores, labels_out


@torch.inference_mode()
def evaluate_fixed_model(
    fixed_model: FixedAnchorFastModel,
    solver: AdaptiveSparseAnchorSolverV4,
    threshold_source: str,
    anormly_ratio: float,
    normalization: str,
) -> Dict[str, float]:
    percentile = 100.0 - float(anormly_ratio)

    train_scores, _ = collect_scores(
        fixed_model, solver, solver.train_loader, normalization=normalization, collect_labels=False
    )
    if threshold_source == "train":
        threshold_scores = train_scores
    elif threshold_source == "original":
        test_scores_for_threshold, _ = collect_scores(
            fixed_model, solver, solver.thre_loader, normalization=normalization, collect_labels=False
        )
        threshold_scores = np.concatenate([train_scores, test_scores_for_threshold], axis=0)
    else:
        raise ValueError(f"Unknown threshold_source: {threshold_source}")

    threshold = float(np.percentile(threshold_scores, percentile))

    test_scores, gt = collect_scores(
        fixed_model, solver, solver.thre_loader, normalization=normalization, collect_labels=True
    )
    if gt is None:
        raise RuntimeError("gt should not be None.")

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
def benchmark_batch_fixed(
    fixed_model: FixedAnchorFastModel,
    solver: AdaptiveSparseAnchorSolverV4,
    raw_batch: torch.Tensor,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    fixed_model.eval()
    batch = raw_batch[:batch_size].contiguous()
    if batch.shape[0] < batch_size:
        raise RuntimeError(f"Batch too small: got {batch.shape[0]}, need {batch_size}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        x = prepare_input(solver, batch)
        _ = fixed_model(x)
    sync()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        x = prepare_input(solver, batch)
        _ = fixed_model(x)
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
def benchmark_full_test_fixed(
    fixed_model: FixedAnchorFastModel,
    solver: AdaptiveSparseAnchorSolverV4,
    repeats: int,
) -> Dict[str, float]:
    fixed_model.eval()

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
            score, _ = fixed_model(x)
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


@torch.inference_mode()
def benchmark_batch_original(
    solver: AdaptiveSparseAnchorSolverV4,
    raw_batch: torch.Tensor,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    solver.model.eval()
    batch = raw_batch[:batch_size].contiguous()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        _ = solver._forward_batch(batch)
    sync()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = solver._forward_batch(batch)
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


def count_active_params(fixed_model: FixedAnchorFastModel) -> int:
    return int(sum(p.numel() for p in fixed_model.parameters() if p.requires_grad))


def write_outputs(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"

    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    ev = metrics["evaluation"]
    b1 = metrics["latency"]["fixed_batch_1"]
    bn = metrics["latency"]["fixed_batch_n"]
    ft = metrics["latency"]["fixed_full_test"]
    ob = metrics["latency"].get("original_batch_n")

    lines = []
    lines.append(f"# ASCA-AD Fixed-Anchor Fast: {metrics['dataset']}\n")

    lines.append("## Fixed Anchors\n")
    lines.append(f"- fixed_local_lags: `{metrics['fixed_local_lags']}`")
    lines.append(f"- fixed_global_lags: `{metrics['fixed_global_lags']}`")
    lines.append(f"- checkpoint: `{metrics['checkpoint']}`")
    lines.append(f"- score_normalization: `{metrics['score_normalization']}`")
    lines.append(f"- threshold_source: `{metrics['threshold_source']}`")
    lines.append(f"- anormly_ratio: `{metrics['anormly_ratio']}`\n")

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
    lines.append(f"| Active Params in Fast Model | {metrics['active_params']} |")
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
    if ob is not None:
        speedup = ob["latency_ms_mean"] / max(bn["latency_ms_mean"], 1e-12)
        lines.append(f"| Speedup vs Original Batch={bn['batch_size']} | {speedup:.3f}x |")
    lines.append("")

    lines.append("## Reference\n")
    lines.append("- 当前 iTransformer / SMD / batch=128 mean latency 约为 `1.967984 ms`。")
    lines.append("- 若本脚本 Batch=128 Mean Latency ms 小于该值，则固定锚点快速版在该口径下超过 iTransformer。")
    lines.append("")

    lines.append("## Anchor Occupancy\n")
    lines.append("```json")
    lines.append(json.dumps(metrics["anchor_stats"], ensure_ascii=False, indent=2))
    lines.append("```")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--channels", type=int, default=38)
    parser.add_argument("--anormly-ratio", dest="anormly_ratio", type=float, default=0.9)
    parser.add_argument("--score-normalization", choices=["official", "raw"], default="official")
    parser.add_argument("--threshold-source", choices=["original", "train"], default="original")
    parser.add_argument("--relation-input", choices=["instance", "standardized"], default="instance")

    parser.add_argument("--fixed-local-k", type=int, default=1)
    parser.add_argument("--fixed-global-k", type=int, default=2)
    parser.add_argument("--manual-local-lags", nargs="*", type=int, default=None)
    parser.add_argument("--manual-global-lags", nargs="*", type=int, default=None)
    parser.add_argument("--occupancy-max-batches", type=int, default=None)

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
    parser.add_argument("--skip-original-benchmark", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    args.checkpoint = str((ROOT / args.checkpoint).resolve()) if not os.path.isabs(args.checkpoint) else args.checkpoint

    print(f"[root] {ROOT}")
    print(f"[checkpoint] {args.checkpoint}")

    original_model, model_kwargs = load_original_model(args)
    cfg = build_solver_config(args, model_kwargs)
    solver = AdaptiveSparseAnchorSolverV4(cfg)
    solver.model.load_state_dict(original_model.state_dict())
    solver.model.eval()

    print("[ok] loaded original ASCA-AD model")
    print("[original model kwargs]")
    print(json.dumps(model_kwargs, ensure_ascii=False, indent=2))

    if args.manual_local_lags is not None and len(args.manual_local_lags) > 0:
        fixed_local = sorted(args.manual_local_lags)
        fixed_global = sorted(args.manual_global_lags or [])
        if not fixed_global:
            raise ValueError("--manual-global-lags 不能为空。")
        anchor_stats = {"manual": True}
    else:
        fixed_local, fixed_global, anchor_stats = choose_fixed_lags(
            solver,
            fixed_local_k=args.fixed_local_k,
            fixed_global_k=args.fixed_global_k,
            max_batches=args.occupancy_max_batches,
        )

    print(f"[fixed_local_lags] {fixed_local}")
    print(f"[fixed_global_lags] {fixed_global}")

    fixed_model = FixedAnchorFastModel(solver.model, fixed_local, fixed_global).to(solver.device)
    fixed_model.eval()

    evaluation = evaluate_fixed_model(
        fixed_model=fixed_model,
        solver=solver,
        threshold_source=args.threshold_source,
        anormly_ratio=args.anormly_ratio,
        normalization=args.score_normalization,
    )

    print("[evaluation]")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))

    first_batch = next(iter(solver.thre_loader))[0]

    fixed_batch_1 = benchmark_batch_fixed(
        fixed_model, solver, first_batch, 1, args.warmup, args.repeats
    )
    fixed_batch_n = benchmark_batch_fixed(
        fixed_model,
        solver,
        first_batch,
        min(args.batch_size, int(first_batch.shape[0])),
        args.warmup,
        args.repeats,
    )
    fixed_full_test = benchmark_full_test_fixed(
        fixed_model, solver, args.full_test_repeats
    )

    latency = {
        "fixed_batch_1": fixed_batch_1,
        "fixed_batch_n": fixed_batch_n,
        "fixed_full_test": fixed_full_test,
    }

    if not args.skip_original_benchmark:
        original_batch_n = benchmark_batch_original(
            solver,
            first_batch,
            min(args.batch_size, int(first_batch.shape[0])),
            args.warmup,
            args.repeats,
        )
        latency["original_batch_n"] = original_batch_n

    metrics = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "fixed_local_lags": fixed_local,
        "fixed_global_lags": fixed_global,
        "anormly_ratio": args.anormly_ratio,
        "score_normalization": args.score_normalization,
        "threshold_source": args.threshold_source,
        "evaluation": evaluation,
        "active_params": count_active_params(fixed_model),
        "latency": latency,
        "anchor_stats": anchor_stats,
        "cpu_max_rss_mib": max_rss_mib(),
    }

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    write_outputs(metrics, out_dir)

    print("\n[done]")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
