#!/usr/bin/env python3
"""
Test ASCA-AD V4 fast inference optimizations.

测试内容：
1) original + torch.no_grad()
2) original + torch.inference_mode()
3) fast-gather + torch.inference_mode()

说明：
- 这两个优化只用于推理速度/显存测试，不会提升检测 F1。
- fast-gather 会用 unfold + index_select 替代原来的 batch advanced indexing。
- 脚本会先比较 original 与 fast-gather 的输出误差，误差足够小时再看延迟结果。

运行位置：项目根目录 /mnt/c/Users/DING/Desktop/Experiment/CODE
示例：
python -u scripts/benchmarks/test_asca_fast_inference.py \
  --dataset SMD --batch-size 128 --seq-len 100 --channels 38 \
  --out-dir results/FAST_INFERENCE_TEST/SMD
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asca_ad.model import AdaptiveSparseAnchorCompetitiveModelV4  # noqa: E402


class FastGatherAdaptiveSparseAnchorCompetitiveModelV4(AdaptiveSparseAnchorCompetitiveModelV4):
    """ASCA-AD V4 with faster symmetric gather for inference benchmark.

    原模型 _symmetric_gather 使用 batch advanced indexing。
    这里改成：
      1. 时间维 replicate padding；
      2. unfold 形成每个时间点周围的窗口视图；
      3. index_select 一次性取出所有 lag 对应的左右锚点。

    输入 x 仍然是 [B, L, M]，输出仍然是 [B, L, K, M]。
    """

    @staticmethod
    def _pad_time_replicate(x: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return x
        if x.ndim != 3:
            raise ValueError("fast gather only supports [B, L, C] tensors.")
        # F.pad 的 replicate 模式只能方便地 pad 最后一维，所以先转为 [B, C, L]。
        return F.pad(
            x.transpose(1, 2).contiguous(),
            (pad, pad),
            mode="replicate",
        ).transpose(1, 2).contiguous()

    @staticmethod
    def _symmetric_gather(x: torch.Tensor, lags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return left/right anchors with replicate boundary.

        Args:
            x: [B, L, C]
            lags: [K]
        Returns:
            left, right: [B, L, K, C]
        """
        if x.ndim != 3:
            raise ValueError("输入必须为 [B,L,C]。")
        if lags.numel() == 0:
            raise ValueError("lags 不能为空。")

        max_lag = int(lags.max().item())
        x_pad = FastGatherAdaptiveSparseAnchorCompetitiveModelV4._pad_time_replicate(x, max_lag)

        # windows: [B, L, C, 2*max_lag+1]
        windows = x_pad.unfold(dimension=1, size=2 * max_lag + 1, step=1)

        left_offsets = (max_lag - lags).to(device=x.device, dtype=torch.long)
        right_offsets = (max_lag + lags).to(device=x.device, dtype=torch.long)

        # index_select 后：[B, L, C, K]，转为 [B, L, K, C]
        left = windows.index_select(dim=-1, index=left_offsets).transpose(-1, -2).contiguous()
        right = windows.index_select(dim=-1, index=right_offsets).transpose(-1, -2).contiguous()
        return left, right


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def max_rss_mib() -> float:
    # Linux/WSL: ru_maxrss is KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def find_checkpoint(project_root: Path, dataset: str, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = project_root / p
        if not p.exists():
            raise FileNotFoundError(f"checkpoint 不存在：{p}")
        return p

    patterns = [
        str(project_root / "checkpoints" / "**" / f"{dataset}_adaptive_anchor_v4*.pt"),
        str(project_root / "checkpoints" / "**" / "*adaptive_anchor_v4*.pt"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    matches = sorted(set(matches), key=lambda x: os.path.getmtime(x), reverse=True)
    if not matches:
        return None
    return Path(matches[0])


def default_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
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
        "gap_weight": args.gap_weight,
    }


def kwargs_from_checkpoint_or_args(args: argparse.Namespace, checkpoint_obj: Optional[dict]) -> Dict[str, Any]:
    kwargs = default_model_kwargs(args)
    if isinstance(checkpoint_obj, dict) and isinstance(checkpoint_obj.get("config"), dict):
        cfg = checkpoint_obj["config"]
        for key in [
            "local_candidate_lags",
            "global_candidate_lags",
            "local_topk",
            "global_topk",
            "selector_hidden",
            "fitter_hidden",
        ]:
            if key in cfg:
                kwargs[key] = cfg[key]
    return kwargs


def load_state_dict_if_available(
    model: torch.nn.Module,
    checkpoint_obj: Optional[dict],
    checkpoint_path: Optional[Path],
) -> None:
    if checkpoint_obj is None:
        print("[warn] 未找到 checkpoint，使用随机初始化权重。只适合测试代码是否运行。")
        return

    if "model" in checkpoint_obj:
        state = checkpoint_obj["model"]
    else:
        state = checkpoint_obj
    model.load_state_dict(state, strict=True)
    print(f"[ok] loaded checkpoint: {checkpoint_path}")


def build_models(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module, Optional[Path]]:
    ckpt_path = find_checkpoint(PROJECT_ROOT, args.dataset, args.checkpoint)
    ckpt_obj = None
    if ckpt_path is not None:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    else:
        print("[warn] 没有自动找到 checkpoint。")

    kwargs = kwargs_from_checkpoint_or_args(args, ckpt_obj)
    print("[model kwargs]", kwargs)

    original = AdaptiveSparseAnchorCompetitiveModelV4(**kwargs)
    fast = FastGatherAdaptiveSparseAnchorCompetitiveModelV4(**kwargs)

    load_state_dict_if_available(original, ckpt_obj, ckpt_path)
    load_state_dict_if_available(fast, ckpt_obj, ckpt_path)

    original.to(device).eval()
    fast.to(device).eval()
    return original, fast, ckpt_path


def make_input(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    x = torch.randn(args.batch_size, args.seq_len, args.channels, device=device)
    if args.instance_norm:
        mean = x.mean(dim=1, keepdim=True).detach()
        var = x.var(dim=1, keepdim=True, unbiased=False).detach()
        x = (x - mean) / torch.sqrt(var + 1e-5)
    return x.contiguous()


def forward_once(model: torch.nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "no_grad":
        with torch.no_grad():
            score, _ = model(x)
            return score
    if mode == "inference_mode":
        with torch.inference_mode():
            score, _ = model(x)
            return score
    raise ValueError(f"unknown mode: {mode}")


def check_equivalence(
    original: torch.nn.Module,
    fast: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    sync(device)
    y0 = forward_once(original, x, "inference_mode")
    y1 = forward_once(fast, x, "inference_mode")
    sync(device)
    diff = (y0 - y1).detach().abs()
    denom = y0.detach().abs().clamp_min(1e-8)
    rel = diff / denom
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "max_rel_diff": float(rel.max().item()),
        "mean_rel_diff": float(rel.mean().item()),
    }


def benchmark(
    name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    mode: str,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        _ = forward_once(model, x, mode)
    sync(device)

    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = forward_once(model, x, mode)
        sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    result = {
        "name": name,
        "mode": mode,
        "batch_size": int(x.shape[0]),
        "seq_len": int(x.shape[1]),
        "channels": int(x.shape[2]),
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_std": float(arr.std()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
        "cpu_max_rss_mib": max_rss_mib(),
    }
    if device.type == "cuda":
        result.update(
            {
                "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024 / 1024),
                "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024 / 1024),
            }
        )
    return result


def write_report(metrics: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fast_inference_test.json"
    md_path = out_dir / "fast_inference_test.md"

    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    rows.append(f"# ASCA-AD Fast Inference Test: {metrics['dataset']}\n")
    rows.append(f"- Checkpoint: `{metrics.get('checkpoint')}`")
    rows.append(f"- Device: `{metrics['device']}`")
    rows.append(f"- Input: B={metrics['batch_size']}, L={metrics['seq_len']}, C={metrics['channels']}")
    rows.append("")
    rows.append("## Output Equivalence")
    rows.append("")
    rows.append("| Metric | Value |")
    rows.append("|---|---:|")
    for k, v in metrics["equivalence"].items():
        rows.append(f"| {k} | {v:.10g} |")
    rows.append("")
    rows.append("## Latency")
    rows.append("")
    rows.append("| Variant | Mean ms | P50 ms | P95 ms | P99 ms | GPU Peak Allocated MiB |")
    rows.append("|---|---:|---:|---:|---:|---:|")
    for item in metrics["benchmarks"]:
        rows.append(
            "| {name} | {mean:.6f} | {p50:.6f} | {p95:.6f} | {p99:.6f} | {gpu:.2f} |".format(
                name=item["name"],
                mean=item["latency_ms_mean"],
                p50=item["latency_ms_p50"],
                p95=item["latency_ms_p95"],
                p99=item["latency_ms_p99"],
                gpu=item.get("gpu_peak_allocated_mib", float("nan")),
            )
        )

    base = metrics["benchmarks"][0]["latency_ms_mean"]
    fast = metrics["benchmarks"][-1]["latency_ms_mean"]
    speedup = base / fast if fast > 0 else float("inf")
    rows.append("")
    rows.append(f"Speedup vs original no_grad: **{speedup:.3f}x**")
    rows.append("")
    md_path.write_text("\n".join(rows), encoding="utf-8")

    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")
    print(md_path.read_text(encoding="utf-8"))


def parse_lags(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--checkpoint", default=None, help="可选：显式指定 checkpoint .pt 路径")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--channels", type=int, default=38)
    parser.add_argument("--instance-norm", action="store_true", default=True)
    parser.add_argument("--no-instance-norm", action="store_false", dest="instance_norm")

    parser.add_argument("--local-lags", type=parse_lags, default=parse_lags("1 2 3 4 5 6 7 8"))
    parser.add_argument("--global-lags", type=parse_lags, default=parse_lags("12 16 20 24 28 32 40 48"))
    parser.add_argument("--local-topk", type=int, default=2)
    parser.add_argument("--global-topk", type=int, default=4)
    parser.add_argument("--selector-hidden", type=int, default=8)
    parser.add_argument("--fitter-hidden", type=int, default=8)
    parser.add_argument("--selector-temperature", type=float, default=0.5)
    parser.add_argument("--similarity-tau", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.03)
    parser.add_argument("--sigma-max", type=float, default=1.50)
    parser.add_argument("--gap-weight", type=float, default=1.0)

    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--out-dir", default="results/FAST_INFERENCE_TEST/SMD")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    original, fast, ckpt_path = build_models(args, device)
    x = make_input(args, device)

    equivalence = check_equivalence(original, fast, x, device)
    print("[equivalence]", equivalence)
    if equivalence["max_abs_diff"] > 1e-5:
        print("[warn] fast-gather 与原模型输出差异超过 1e-5，请不要直接替换正式模型。")

    b0 = benchmark(
        "original_no_grad",
        original,
        x,
        device,
        mode="no_grad",
        warmup=args.warmup,
        repeats=args.repeats,
    )
    b1 = benchmark(
        "original_inference_mode",
        original,
        x,
        device,
        mode="inference_mode",
        warmup=args.warmup,
        repeats=args.repeats,
    )
    b2 = benchmark(
        "fast_gather_inference_mode",
        fast,
        x,
        device,
        mode="inference_mode",
        warmup=args.warmup,
        repeats=args.repeats,
    )

    metrics = {
        "dataset": args.dataset,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "device": str(device),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "channels": args.channels,
        "equivalence": equivalence,
        "benchmarks": [b0, b1, b2],
    }
    write_report(metrics, PROJECT_ROOT / args.out_dir)


if __name__ == "__main__":
    main()
