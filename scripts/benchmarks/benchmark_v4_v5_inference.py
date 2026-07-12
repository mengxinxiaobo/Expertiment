from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import AdaptiveSparseAnchorCompetitiveModelV4
from src.legacy.main_hierarchical_asca_v5 import (
    HierarchicalAdaptiveSparseAnchorModelV5,
)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_state(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}\n"
            f"missing={missing}\nunexpected={unexpected}"
        )


@torch.no_grad()
def benchmark_cuda(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    model.eval()
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(x)
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))

    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[max(0, int(0.95 * len(samples)) - 1)],
        "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


@torch.no_grad()
def benchmark_cpu(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    model.eval()
    for _ in range(warmup):
        model(x)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        samples.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[max(0, int(0.95 * len(samples)) - 1)],
        "peak_mib": float("nan"),
    }


def build_v4(args: argparse.Namespace) -> torch.nn.Module:
    return AdaptiveSparseAnchorCompetitiveModelV4(
        local_candidate_lags=args.local_candidate_lags,
        global_candidate_lags=args.global_candidate_lags,
        local_topk=args.local_topk,
        global_topk=args.global_topk,
        selector_hidden=args.selector_hidden,
        fitter_hidden=args.fitter_hidden,
        selector_temperature=args.selector_temperature,
        similarity_tau=args.similarity_tau,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        gap_weight=args.gap_weight,
    )


def build_v5(args: argparse.Namespace) -> torch.nn.Module:
    return HierarchicalAdaptiveSparseAnchorModelV5(
        local_candidate_lags=args.local_candidate_lags,
        global_candidate_lags=args.global_candidate_lags,
        local_topk=args.local_topk,
        global_topk=args.global_topk,
        local_route_topk=args.local_route_topk,
        global_route_topk=args.global_route_topk,
        router_hidden=args.router_hidden,
        selector_hidden=args.selector_hidden,
        selector_temperature=args.selector_temperature,
        router_temperature=args.router_temperature,
        similarity_tau=args.similarity_tau,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        gap_weight=args.gap_weight,
        parameter_budget=args.parameter_budget,
    )


def run_one(
    name: str,
    model: torch.nn.Module,
    checkpoint: str,
    args: argparse.Namespace,
) -> None:
    load_state(model, checkpoint)
    model = model.to(args.device)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"checkpoint : {checkpoint}")
    print(f"parameters : {count_params(model):,}")

    bench = benchmark_cuda if args.device.startswith("cuda") else benchmark_cpu

    for batch in args.batch_sizes:
        x = torch.randn(
            batch,
            args.seq_len,
            args.channels,
            device=args.device,
            dtype=torch.float32,
        )
        result = bench(model, x, args.warmup, args.repeats)
        print(
            f"batch={batch:<4d} "
            f"mean={result['mean_ms']:.6f} ms  "
            f"median={result['median_ms']:.6f} ms  "
            f"p95={result['p95_ms']:.6f} ms  "
            f"peak={result['peak_mib']:.2f} MiB"
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare current ASCA-AD V4 and hierarchical V5 inference latency."
    )
    p.add_argument("--v4-checkpoint", required=True)
    p.add_argument("--v5-checkpoint", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--channels", type=int, default=38)
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 128])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=200)

    p.add_argument("--local-candidate-lags", nargs="+", type=int,
                   default=[1,2,3,4,5,6,7,8])
    p.add_argument("--global-candidate-lags", nargs="+", type=int,
                   default=[12,16,20,24,28,32,40,48])
    p.add_argument("--local-topk", type=int, default=2)
    p.add_argument("--global-topk", type=int, default=4)
    p.add_argument("--selector-hidden", type=int, default=8)
    p.add_argument("--fitter-hidden", type=int, default=8)
    p.add_argument("--selector-temperature", type=float, default=0.5)
    p.add_argument("--similarity-tau", type=float, default=1.0)
    p.add_argument("--sigma-min", type=float, default=0.03)
    p.add_argument("--sigma-max", type=float, default=1.50)
    p.add_argument("--gap-weight", type=float, default=1.0)

    p.add_argument("--local-route-topk", type=int, default=4)
    p.add_argument("--global-route-topk", type=int, default=4)
    p.add_argument("--router-hidden", type=int, default=12)
    p.add_argument("--router-temperature", type=float, default=0.75)
    p.add_argument("--parameter-budget", type=int, default=800)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("ASCA-AD inference benchmark")
    print(f"device      : {args.device}")
    print(f"input shape : [B,{args.seq_len},{args.channels}]")
    print(f"warmup      : {args.warmup}")
    print(f"repeats     : {args.repeats}")

    run_one("ASCA-AD V4", build_v4(args), args.v4_checkpoint, args)
    run_one("Hierarchical ASCA-AD V5", build_v5(args), args.v5_checkpoint, args)


if __name__ == "__main__":
    main()
