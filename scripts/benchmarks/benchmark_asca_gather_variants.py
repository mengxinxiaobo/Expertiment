#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark ASCA-AD V4 symmetric-gather implementations.

比较三种等价 gather 后端：
1) advanced_index: 原始高级索引实现
2) cached_index  : 缓存 index tensor，减少重复构造索引
3) shifted_stack : 用规则 shift/stack 替代高级索引

要求：先把 /mnt/data/model_v4_gather_optimized.py 复制到 asca_ad/model.py。
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asca_ad.model import AdaptiveSparseAnchorSolverV4, set_seed  # noqa: E402


BACKENDS = ["advanced_index", "cached_index", "shifted_stack"]


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
        "gather_backend": args.initial_backend,
        "relation_input": args.relation_input,
        "score_modes": ["combined"],
        "primary_score": "combined",
        "score_normalization": args.score_normalization,
        "threshold_source": args.threshold_source,
        "quantile_method": "exact",
        "quantile_buffer": 50000,
        "local_size": 7,
        "global_size": [11],
        "d_model": 8,
        "loss_fuc": "MSE",
        "r": 0.5,
        "similar": "MSE",
        "rec_timeseries": True,
        "model_save_path": str(ROOT / "checkpoints" / args.dataset),
        "result_path": str(ROOT / "results" / f"{args.dataset}_ASCA_GATHER_VARIANTS"),
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
    joint_affinity = torch.cat([center_affinity, local["affinity"], global_["affinity"]], dim=-1)
    joint_target = joint_affinity / joint_affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    local_count = model.local_lags.numel()
    target_center = joint_target[..., :1]
    target_local = joint_target[..., 1 : 1 + local_count]
    target_global = joint_target[..., 1 + local_count :]

    local_summary = model._weighted_summary(target_local, local["lag_norm"], local_gate, local_weights)
    global_summary = model._weighted_summary(target_global, global_["lag_norm"], global_gate, global_weights)
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
    return score_total + model.gap_weight * score_gap


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
    backend: str,
    loader: Iterable,
    score_normalization: str,
    collect_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    solver.model.eval()
    solver.model.set_gather_backend(backend)
    scores = []
    labels_all = []
    for input_data, labels in loader:
        x = prepare_input(solver, input_data)
        raw = asca_score_only(solver.model, x)
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
def equivalence_check(solver: AdaptiveSparseAnchorSolverV4, backends: list[str], max_batches: int) -> dict:
    solver.model.eval()
    reference = backends[0]
    out = {backend: {"checked_batches": 0, "max_abs_diff": 0.0, "max_rel_diff": 0.0} for backend in backends[1:]}
    checked = 0
    for input_data, _ in solver.thre_loader:
        x = prepare_input(solver, input_data)
        solver.model.set_gather_backend(reference)
        ref = asca_score_only(solver.model, x)
        for backend in backends[1:]:
            solver.model.set_gather_backend(backend)
            cur = asca_score_only(solver.model, x)
            abs_diff = (ref - cur).abs()
            rel_diff = abs_diff / ref.abs().clamp_min(1e-12)
            out[backend]["max_abs_diff"] = max(out[backend]["max_abs_diff"], float(abs_diff.max().detach().cpu().item()))
            out[backend]["max_rel_diff"] = max(out[backend]["max_rel_diff"], float(rel_diff.max().detach().cpu().item()))
            out[backend]["checked_batches"] += 1
        checked += 1
        if checked >= max_batches:
            break
    return {"reference": reference, "variants": out}


@torch.inference_mode()
def benchmark_batch(
    solver: AdaptiveSparseAnchorSolverV4,
    backend: str,
    raw_batch: torch.Tensor,
    batch_size: int,
    score_normalization: str,
    warmup: int,
    repeats: int,
) -> dict:
    solver.model.eval()
    solver.model.set_gather_backend(backend)
    if hasattr(solver.model, "clear_gather_cache"):
        solver.model.clear_gather_cache()
    batch = raw_batch[:batch_size].contiguous()
    if batch.shape[0] < batch_size:
        raise RuntimeError(f"Batch too small: got {batch.shape[0]}, need {batch_size}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        x = prepare_input(solver, batch)
        score = asca_score_only(solver.model, x)
        _ = normalize_score(score, score_normalization)
    sync()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        x = prepare_input(solver, batch)
        score = asca_score_only(solver.model, x)
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
    backend: str,
    score_normalization: str,
    repeats: int,
) -> dict:
    solver.model.eval()
    solver.model.set_gather_backend(backend)
    if hasattr(solver.model, "clear_gather_cache"):
        solver.model.clear_gather_cache()
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
            score = asca_score_only(solver.model, x)
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


def ratio(ref: float, cur: float) -> float:
    return float(ref / cur) if cur > 0 else float("nan")


def write_outputs(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    params = metrics["params"]
    ref = metrics["backends"][0]
    lines = []
    lines.append(f"# ASCA-AD V4 Gather Backend Benchmark on {metrics['dataset']}\n")
    lines.append("## Checkpoint\n")
    lines.append(f"`{metrics['checkpoint']}`\n")
    lines.append("## Parameters\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Trainable Params | {params['trainable_params']} |")
    lines.append(f"| Total Params | {params['total_params']} |")
    lines.append(f"| Param Size MiB | {params['param_mib']:.6f} |")
    lines.append("")

    lines.append("## Equivalence Check\n")
    lines.append(f"Reference backend: `{ref}`\n")
    lines.append("| Backend | Checked Batches | Max Abs Diff | Max Rel Diff |")
    lines.append("|---|---:|---:|---:|")
    for backend, item in metrics["equivalence"]["variants"].items():
        lines.append(f"| {backend} | {item['checked_batches']} | {item['max_abs_diff']:.12g} | {item['max_rel_diff']:.12g} |")
    lines.append("")

    lines.append("## Detection Metrics\n")
    lines.append("| Backend | PA Precision | PA Recall | PA F1 | PA Accuracy | RAW F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for backend in metrics["backends"]:
        ev = metrics["evaluation"][backend]
        lines.append(
            f"| {backend} | {ev['pa_precision']:.6f} | {ev['pa_recall']:.6f} | "
            f"{ev['pa_f1']:.6f} | {ev['pa_accuracy']:.6f} | {ev['raw_f1']:.6f} |"
        )
    lines.append("")

    lines.append("## Latency / Throughput\n")
    lines.append("| Backend | Batch=1 ms | Speedup vs ref | Batch=N ms | Speedup vs ref | Full-test s | Speedup vs ref | GPU Peak MiB |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    ref_lat = metrics["latency"][ref]
    for backend in metrics["backends"]:
        lat = metrics["latency"][backend]
        b1 = lat["batch_1"]["latency_ms_mean"]
        bn = lat["batch_n"]["latency_ms_mean"]
        ft = lat["full_test"]["full_test_seconds_mean"]
        ref_b1 = ref_lat["batch_1"]["latency_ms_mean"]
        ref_bn = ref_lat["batch_n"]["latency_ms_mean"]
        ref_ft = ref_lat["full_test"]["full_test_seconds_mean"]
        gpu = lat["full_test"].get("gpu_peak_allocated_mib", float("nan"))
        lines.append(
            f"| {backend} | {b1:.6f} | {ratio(ref_b1, b1):.3f}x | "
            f"{bn:.6f} | {ratio(ref_bn, bn):.3f}x | "
            f"{ft:.6f} | {ratio(ref_ft, ft):.3f}x | {gpu:.2f} |"
        )
    lines.append("")
    lines.append(f"CPU Max RSS MiB: `{metrics['cpu_max_rss_mib']:.2f}`\n")

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

    parser.add_argument("--initial-backend", choices=BACKENDS, default="advanced_index")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
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
    print(f"[backends] {args.backends}")

    solver = AdaptiveSparseAnchorSolverV4(build_solver_config(args))
    if not hasattr(solver.model, "set_gather_backend"):
        raise RuntimeError("当前 asca_ad/model.py 还没有 set_gather_backend。请先替换为 model_v4_gather_optimized.py。")
    checkpoint = load_checkpoint(solver, args.checkpoint)
    solver.model.eval()

    params = count_params(solver.model)
    print("[params]", json.dumps(params, indent=2))

    eq = equivalence_check(solver, args.backends, args.equiv_batches)
    print("[equivalence]", json.dumps(eq, indent=2))

    evaluation = {}
    gt_ref = None
    for backend in args.backends:
        print(f"[collect/eval] backend={backend}")
        train_s, _ = collect_scores(solver, backend, solver.train_loader, args.score_normalization)
        test_s, gt = collect_scores(solver, backend, solver.thre_loader, args.score_normalization, collect_labels=True)
        assert gt is not None
        if gt_ref is None:
            gt_ref = gt
        elif not np.array_equal(gt_ref, gt):
            raise RuntimeError("Labels changed across backends; this should not happen.")
        evaluation[backend] = evaluate_scores(train_s, test_s, gt, args.anormly_ratio)

    first_batch = next(iter(solver.thre_loader))[0]
    batch_n = min(args.batch_size, int(first_batch.shape[0]))
    latency = {}
    for backend in args.backends:
        print(f"[benchmark] backend={backend}")
        latency[backend] = {
            "batch_1": benchmark_batch(solver, backend, first_batch, 1, args.score_normalization, args.warmup, args.repeats),
            "batch_n": benchmark_batch(solver, backend, first_batch, batch_n, args.score_normalization, args.warmup, args.repeats),
            "full_test": benchmark_full_test(solver, backend, args.score_normalization, args.full_test_repeats),
        }

    metrics = {
        "dataset": args.dataset,
        "checkpoint": checkpoint,
        "score_normalization": args.score_normalization,
        "anormly_ratio": args.anormly_ratio,
        "backends": args.backends,
        "params": params,
        "equivalence": eq,
        "evaluation": evaluation,
        "latency": latency,
        "cpu_max_rss_mib": max_rss_mib(),
    }
    write_outputs(metrics, out_dir)
    print("\n[done]")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
