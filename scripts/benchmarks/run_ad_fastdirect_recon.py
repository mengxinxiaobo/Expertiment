#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASCA-AD-FastDirect / MicroRecon-AD
One-step unsupervised fast anomaly detector.
No teacher, no distillation: train a tiny reconstruction model on train windows;
use reconstruction error as anomaly score.
"""
from __future__ import annotations

import argparse, json, os, resource, sys, time
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MicroReconAD(nn.Module):
    """Depthwise-separable reconstruction model."""
    def __init__(self, channels: int, hidden: int = 24, kernel_size: int = 9,
                 dropout: float = 0.0, residual: bool = False):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.residual = residual
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=True),
            nn.Conv1d(channels, hidden, 1, bias=True),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Conv1d(hidden, channels, 1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return x + y if self.residual else y


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"total_params": int(total), "trainable_params": int(trainable),
            "param_bytes": int(bytes_), "param_mib": float(bytes_ / 1024 / 1024)}


def build_solver_config(args: argparse.Namespace) -> dict:
    return {
        "dataset": args.dataset, "data_path": args.dataset,
        "input_c": args.channels, "output_c": args.channels,
        "win_size": args.seq_len, "batch_size": args.batch_size,
        "num_epochs": 1, "lr": args.lr, "anormly_ratio": args.anormly_ratio,
        "index": 137, "mode": "train", "seed": args.seed,
        "local_candidate_lags": [1,2,3,4,5,6,7,8],
        "global_candidate_lags": [12,16,20,24,28,32,40,48],
        "local_topk": 2, "global_topk": 4,
        "selector_hidden": 8, "fitter_hidden": 8,
        "selector_temperature": 0.5, "similarity_tau": 1.0,
        "sigma_min": 0.03, "sigma_max": 1.5,
        "area_weight": 0.1, "selector_balance_weight": 0.05,
        "gap_weight": 1.0, "relation_input": args.relation_input,
        "score_modes": ["combined"], "primary_score": "combined",
        "score_normalization": "official", "threshold_source": args.threshold_source,
        "quantile_method": "exact", "quantile_buffer": 50000,
        "local_size": 7, "global_size": [11], "d_model": 8,
        "loss_fuc": "MSE", "r": 0.5, "similar": "MSE", "rec_timeseries": True,
        "model_save_path": str(ROOT / "checkpoints" / f"{args.dataset}_FASTDIRECT" / "RECON"),
        "result_path": str(ROOT / "results" / f"{args.dataset}_FASTDIRECT" / "RECON"),
        "use_gpu": torch.cuda.is_available(), "use_multi_gpu": False,
        "gpu": 0, "devices": "0",
    }


def prepare_input(solver, input_data: torch.Tensor) -> torch.Tensor:
    return solver._prepare_input(input_data)


def corrupt_input(x: torch.Tensor, mask_ratio: float, noise_std: float) -> torch.Tensor:
    out = x
    if noise_std > 0:
        scale = x.detach().std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        out = out + torch.randn_like(out) * scale * noise_std
    if mask_ratio > 0:
        out = out.masked_fill(torch.rand_like(out) < mask_ratio, 0.0)
    return out


def recon_loss(recon: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(recon, target)
    if name == "huber":
        return F.smooth_l1_loss(recon, target, beta=0.5)
    raise ValueError(name)


def recon_score(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (recon - x).square().mean(dim=-1)


def norm_score(score: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return score
    if mode == "official":
        mn = score.min(dim=-1, keepdim=True).values
        mx = score.max(dim=-1, keepdim=True).values
        scaled = (score - mn) / (mx - mn + 1e-5)
        return torch.softmax(scaled, dim=-1)
    raise ValueError(mode)


def train(args, solver, model) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train(); losses = []; t0 = time.perf_counter()
        for input_data, _ in solver.train_loader:
            x = prepare_input(solver, input_data)
            x_in = corrupt_input(x, args.mask_ratio, args.noise_std)
            recon = model(x_in)
            loss = recon_loss(recon, x, args.loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sync()
        rec = {"epoch": ep, "train_recon_loss": float(np.mean(losses)),
               "seconds": float(time.perf_counter() - t0)}
        hist.append(rec)
        print(f"Epoch [{ep:03d}/{args.epochs:03d}] recon_loss={rec['train_recon_loss']:.8f}, time={rec['seconds']:.3f}s")
    return hist


@torch.inference_mode()
def collect_scores(solver, model, loader: Iterable, modes: list[str], collect_labels=False) -> Tuple[dict, np.ndarray | None]:
    model.eval(); parts = {m: [] for m in modes}; label_parts = []
    for input_data, labels in loader:
        x = prepare_input(solver, input_data)
        raw = recon_score(model(x), x)
        for m in modes:
            parts[m].append(norm_score(raw, m).detach().cpu().numpy().reshape(-1))
        if collect_labels:
            label_parts.append(labels.detach().cpu().numpy().reshape(-1))
    scores = {m: np.concatenate(v) for m, v in parts.items()}
    labels = np.concatenate(label_parts).astype(int) if collect_labels else None
    return scores, labels


def evaluate(args, solver, model) -> dict:
    train_scores, _ = collect_scores(solver, model, solver.train_loader, args.score_modes, False)
    if args.threshold_source == "train":
        threshold_scores = train_scores
    else:
        test_thr, _ = collect_scores(solver, model, solver.thre_loader, args.score_modes, False)
        threshold_scores = {m: np.concatenate([train_scores[m], test_thr[m]]) for m in args.score_modes}
    test_scores, gt = collect_scores(solver, model, solver.thre_loader, args.score_modes, True)
    assert gt is not None
    results = {}
    for m in args.score_modes:
        threshold = float(np.percentile(threshold_scores[m], 100.0 - args.anormly_ratio))
        pred = (test_scores[m] > threshold).astype(int)
        raw_acc = float(accuracy_score(gt, pred))
        raw_p, raw_r, raw_f1, _ = precision_recall_fscore_support(gt, pred, average="binary", zero_division=0)
        raw_mcc = float(matthews_corrcoef(gt, pred))
        pa = point_adjust(pred, gt)
        pa_acc = float(accuracy_score(gt, pa))
        pa_p, pa_r, pa_f1, _ = precision_recall_fscore_support(gt, pa, average="binary", zero_division=0)
        pa_mcc = float(matthews_corrcoef(gt, pa))
        results[m] = {"threshold": threshold, "threshold_samples": int(threshold_scores[m].size),
                      "test_points": int(test_scores[m].size),
                      "raw_accuracy": raw_acc, "raw_precision": float(raw_p),
                      "raw_recall": float(raw_r), "raw_f1": float(raw_f1), "raw_mcc": raw_mcc,
                      "pa_accuracy": pa_acc, "pa_precision": float(pa_p),
                      "pa_recall": float(pa_r), "pa_f1": float(pa_f1), "pa_mcc": pa_mcc}
    return results


@torch.inference_mode()
def bench_batch(args, solver, model, raw_batch, batch_size: int) -> dict:
    model.eval(); batch = raw_batch[:batch_size].contiguous()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        x = prepare_input(solver, batch); _ = norm_score(recon_score(model(x), x), args.primary_score)
    sync(); times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        x = prepare_input(solver, batch); _ = norm_score(recon_score(model(x), x), args.primary_score)
        sync(); times.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(times)
    out = {"batch_size": int(batch_size), "latency_ms_mean": float(arr.mean()),
           "latency_ms_std": float(arr.std()), "latency_ms_p50": float(np.percentile(arr, 50)),
           "latency_ms_p95": float(np.percentile(arr, 95)), "latency_ms_p99": float(np.percentile(arr, 99)),
           "cpu_max_rss_mib": max_rss_mib()}
    if torch.cuda.is_available():
        out["gpu_peak_allocated_mib"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        out["gpu_peak_reserved_mib"] = float(torch.cuda.max_memory_reserved() / 1024 / 1024)
    return out


@torch.inference_mode()
def bench_full(args, solver, model) -> dict:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    secs, pts = [], []
    for _ in range(args.full_test_repeats):
        total = 0; t0 = time.perf_counter()
        for input_data, _ in solver.thre_loader:
            x = prepare_input(solver, input_data); s = norm_score(recon_score(model(x), x), args.primary_score)
            total += int(s.numel())
        sync(); secs.append(time.perf_counter() - t0); pts.append(total)
    arr = np.asarray(secs); points = int(np.mean(pts))
    out = {"full_test_repeats": int(args.full_test_repeats),
           "full_test_seconds_mean": float(arr.mean()), "full_test_seconds_std": float(arr.std()),
           "full_test_seconds_p50": float(np.percentile(arr, 50)), "full_test_points": points,
           "full_test_points_per_second": float(points / arr.mean()), "cpu_max_rss_mib": max_rss_mib()}
    if torch.cuda.is_available():
        out["gpu_peak_allocated_mib"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        out["gpu_peak_reserved_mib"] = float(torch.cuda.max_memory_reserved() / 1024 / 1024)
    return out


def save_ckpt(args, model, out_dir: Path) -> str:
    path = out_dir / "fastdirect_recon.pt"
    torch.save({"model": model.state_dict(), "config": vars(args)}, path)
    return str(path)


def write_outputs(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    primary = metrics["primary_score"]; ev = metrics["evaluation"][primary]
    b1 = metrics["latency"]["batch_1"]; bn = metrics["latency"]["batch_n"]; ft = metrics["latency"]["full_test"]
    lines = [f"# ASCA-AD-FastDirect / MicroRecon-AD on {metrics['dataset']}\n"]
    lines += ["## Config\n",
              f"- checkpoint: `{metrics['checkpoint']}`",
              f"- hidden: `{metrics['model_config']['hidden']}`",
              f"- kernel_size: `{metrics['model_config']['kernel_size']}`",
              f"- residual: `{metrics['model_config']['residual']}`",
              f"- mask_ratio: `{metrics['training_config']['mask_ratio']}`",
              f"- noise_std: `{metrics['training_config']['noise_std']}`",
              f"- train_epochs: `{metrics['train_epochs']}`",
              f"- anormly_ratio: `{metrics['anormly_ratio']}`",
              f"- score_modes: `{metrics['score_modes']}`",
              f"- primary_score: `{primary}`\n"]
    lines += ["## Parameter Budget\n", f"- Param budget: `{metrics['param_budget']}`",
              f"- Trainable params: `{metrics['params']['trainable_params']}`",
              f"- Under budget: `{metrics['under_budget']}`\n"]
    lines += [f"## Accuracy ({primary})\n", "| Metric | Value |", "|---|---:|",
              f"| Threshold | {ev['threshold']:.10f} |",
              f"| RAW Accuracy | {ev['raw_accuracy']:.6f} |",
              f"| RAW Precision | {ev['raw_precision']:.6f} |",
              f"| RAW Recall | {ev['raw_recall']:.6f} |",
              f"| RAW F1 | {ev['raw_f1']:.6f} |",
              f"| PA Accuracy | {ev['pa_accuracy']:.6f} |",
              f"| PA Precision | {ev['pa_precision']:.6f} |",
              f"| PA Recall | {ev['pa_recall']:.6f} |",
              f"| PA F1 | {ev['pa_f1']:.6f} |", ""]
    if len(metrics["evaluation"]) > 1:
        lines += ["## Accuracy by Score Mode\n", "| Score | PA Precision | PA Recall | PA F1 | PA Accuracy | RAW F1 |", "|---|---:|---:|---:|---:|---:|"]
        for mode, r in metrics["evaluation"].items():
            lines.append(f"| {mode} | {r['pa_precision']:.6f} | {r['pa_recall']:.6f} | {r['pa_f1']:.6f} | {r['pa_accuracy']:.6f} | {r['raw_f1']:.6f} |")
        lines.append("")
    lines += ["## Lightweight / Latency\n", "| Metric | Value |", "|---|---:|",
              f"| Trainable Params | {metrics['params']['trainable_params']} |",
              f"| Total Params | {metrics['params']['total_params']} |",
              f"| Param Size MiB | {metrics['params']['param_mib']:.6f} |",
              f"| Batch=1 Mean Latency ms | {b1['latency_ms_mean']:.6f} |",
              f"| Batch=1 P95 Latency ms | {b1['latency_ms_p95']:.6f} |",
              f"| Batch={bn['batch_size']} Mean Latency ms | {bn['latency_ms_mean']:.6f} |",
              f"| Batch={bn['batch_size']} P95 Latency ms | {bn['latency_ms_p95']:.6f} |",
              f"| Full-test Seconds Mean | {ft['full_test_seconds_mean']:.6f} |",
              f"| Full-test Throughput points/s | {ft['full_test_points_per_second']:.2f} |",
              f"| CPU Max RSS MiB | {metrics['cpu_max_rss_mib']:.2f} |"]
    if "gpu_peak_allocated_mib" in ft:
        lines += [f"| Full-test GPU Peak Allocated MiB | {ft['gpu_peak_allocated_mib']:.2f} |",
                  f"| Full-test GPU Peak Reserved MiB | {ft['gpu_peak_reserved_mib']:.2f} |"]
    lines += ["", "## Training History", "```json", json.dumps(metrics["training_history"], ensure_ascii=False, indent=2), "```"]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_dir / 'summary.json'}"); print(f"[saved] {out_dir / 'summary.md'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--channels", type=int, required=True); p.add_argument("--anormly-ratio", type=float, required=True)
    p.add_argument("--relation-input", choices=["instance", "standardized"], default="instance")
    p.add_argument("--threshold-source", choices=["original", "train"], default="original")
    p.add_argument("--hidden", type=int, default=24); p.add_argument("--kernel-size", type=int, default=9)
    p.add_argument("--dropout", type=float, default=0.0); p.add_argument("--residual", action="store_true")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--mask-ratio", type=float, default=0.10); p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--score-modes", nargs="+", choices=["raw", "official"], default=["raw", "official"])
    p.add_argument("--primary-score", choices=["raw", "official"], default="raw")
    p.add_argument("--param-budget", type=int, default=2561)
    p.add_argument("--warmup", type=int, default=30); p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--full-test-repeats", type=int, default=3); p.add_argument("--compile", action="store_true")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.primary_score not in args.score_modes:
        raise ValueError("--primary-score must be in --score-modes")
    set_seed(args.seed)
    if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir); out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[root] {ROOT}"); print(f"[out_dir] {out_dir}")
    solver = AdaptiveSparseAnchorSolverV4(build_solver_config(args))
    model = MicroReconAD(args.channels, args.hidden, args.kernel_size, args.dropout, args.residual).to(solver.device)
    params = count_params(model); print("[model params]", json.dumps(params, indent=2))
    if params["trainable_params"] > args.param_budget:
        raise RuntimeError(f"参数量 {params['trainable_params']} 超过预算 {args.param_budget}。")
    hist = train(args, solver, model)
    if args.compile:
        try: model = torch.compile(model)  # type: ignore
        except Exception as e: print(f"[compile failed] {e}")
    evaluation = evaluate(args, solver, model)
    first_batch = next(iter(solver.thre_loader))[0]
    b1 = bench_batch(args, solver, model, first_batch, 1)
    bn = bench_batch(args, solver, model, first_batch, min(args.batch_size, int(first_batch.shape[0])))
    ft = bench_full(args, solver, model)
    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model  # type: ignore
    ckpt = save_ckpt(args, save_model, out_dir)
    metrics = {"dataset": args.dataset, "checkpoint": ckpt,
               "model_config": {"channels": args.channels, "hidden": args.hidden, "kernel_size": args.kernel_size, "dropout": args.dropout, "residual": args.residual},
               "training_config": {"loss": args.loss, "mask_ratio": args.mask_ratio, "noise_std": args.noise_std, "lr": args.lr, "weight_decay": args.weight_decay},
               "score_modes": args.score_modes, "primary_score": args.primary_score,
               "threshold_source": args.threshold_source, "anormly_ratio": args.anormly_ratio,
               "train_epochs": args.epochs, "training_history": hist, "evaluation": evaluation,
               "params": count_params(save_model), "param_budget": int(args.param_budget),
               "under_budget": count_params(save_model)["trainable_params"] <= int(args.param_budget),
               "latency": {"batch_1": b1, "batch_n": bn, "full_test": ft}, "cpu_max_rss_mib": max_rss_mib()}
    write_outputs(metrics, out_dir)
    print("\n[done]"); print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
