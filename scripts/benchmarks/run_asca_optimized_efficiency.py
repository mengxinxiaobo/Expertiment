#!/usr/bin/env python3
"""Independent-process old-vs-V4-IO inference efficiency benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from scripts.benchmarks.asca_optimization_common import (
    BATCH_SIZE, DATASETS, OUT, ROOT, WINDOW, build_optimized_from_old, cpu_batches,
    full_coverage_starts, load_old_model, protected_snapshot, require_cuda, save_json,
    set_seed, sha256, state_dict_kib,
)

WARMUP = 30
LATENCY_REPEATS = 200
FULL_REPEATS = 20


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def cuda_latency(call: Callable[[], torch.Tensor], inference_mode: bool) -> dict[str, float]:
    context = torch.inference_mode if inference_mode else torch.no_grad
    with context():
        for _ in range(WARMUP):
            call()
    torch.cuda.synchronize()
    samples = []
    with context():
        for _ in range(LATENCY_REPEATS):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); output = call(); end.record(); end.synchronize()
            samples.append(float(start.elapsed_time(end)))
            del output
    return {"mean_ms": statistics.mean(samples), "std_ms": statistics.pstdev(samples)}


def timed_score(score_call: Callable[[torch.Tensor], torch.Tensor], batch: torch.Tensor) -> tuple[float, int]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); output = score_call(batch); end.record(); end.synchronize()
    seconds = float(start.elapsed_time(end)) / 1000.0
    count = int(output.shape[0])
    del output
    return seconds, count


def worker(dataset: str, implementation: str) -> None:
    set_seed()
    device = require_cuda()
    before = protected_snapshot()
    protocol, old_model, old_score, checkpoint = load_old_model(dataset, device)
    if implementation == "old":
        model, score_call, optimized = old_model, old_score, False
        display = "old_implementation"
    else:
        model, score_call = build_optimized_from_old(old_model, checkpoint, device)
        del old_model
        optimized = True
        display = "optimized_implementation"
    model.eval()
    checkpoint_hash_before = sha256(checkpoint)
    test = protocol.load_scaled_test()
    if test.dtype != np.float32 or not np.isfinite(test).all():
        raise RuntimeError("formal scaled test must be finite float32")
    starts = full_coverage_starts(len(test), WINDOW)
    first_cpu = next(cpu_batches(test, starts))
    batch1 = first_cpu[:1].to(device)
    batch128 = batch1.repeat(BATCH_SIZE, 1, 1)
    print(f"[{dataset}/{display}] latency started {now()}", flush=True)
    latency1 = cuda_latency(lambda: score_call(batch1), optimized)
    latency128 = cuda_latency(lambda: score_call(batch128), optimized)
    del batch1, batch128, first_cpu
    gc.collect(); torch.cuda.empty_cache()

    context = torch.inference_mode if optimized else torch.no_grad
    first_cpu = next(cpu_batches(test, starts))
    warmup_batch = first_cpu.to(device)
    with context():
        for _ in range(WARMUP):
            score_call(warmup_batch)
    torch.cuda.synchronize(device)
    del warmup_batch, first_cpu
    gc.collect(); torch.cuda.empty_cache()

    full_samples = []
    print(f"[{dataset}/{display}] full test x{FULL_REPEATS} started {now()}", flush=True)
    with context():
        for repeat in range(1, FULL_REPEATS + 1):
            gpu_seconds = 0.0
            output_windows = 0
            for cpu_batch in cpu_batches(test, starts):
                gpu_batch = cpu_batch.to(device, non_blocking=False)
                seconds, count = timed_score(score_call, gpu_batch)
                gpu_seconds += seconds; output_windows += count
                del gpu_batch, cpu_batch
            if output_windows != len(starts):
                raise RuntimeError("full-test window coverage mismatch")
            full_samples.append(gpu_seconds)
            eta = statistics.mean(full_samples) * (FULL_REPEATS - repeat)
            print(
                f"Repeat {repeat}/{FULL_REPEATS} | GPU Time={gpu_seconds:.6f}s | ETA={eta:.1f}s",
                flush=True,
            )

    gc.collect(); torch.cuda.synchronize(device); torch.cuda.empty_cache(); gc.collect()
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    processed_windows = 0
    with context():
        for cpu_batch in cpu_batches(test, starts):
            gpu_batch = cpu_batch.to(device, non_blocking=False)
            output = score_call(gpu_batch)
            processed_windows += int(output.shape[0])
            del output, gpu_batch, cpu_batch
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    if processed_windows != len(starts):
        raise RuntimeError("GPU peak pass did not cover all windows")
    checkpoint_hash_after = sha256(checkpoint)
    after = protected_snapshot()
    full_mean = statistics.mean(full_samples)
    payload = {
        "status": "COMPLETE", "dataset": dataset, "implementation": display,
        "model": "ASCA-AD V4" if implementation == "old" else "ASCA-AD V4-IO",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "state_dict_kib": state_dict_kib(model),
        "latency_b1": latency1, "latency_b128": latency128,
        "full_test_time_mean_s": full_mean,
        "full_test_time_std_s": statistics.pstdev(full_samples),
        "throughput_points_per_s": len(test) / full_mean,
        "gpu_peak_mib": peak / 2**20,
        "gpu_incremental_mib": (peak - baseline) / 2**20,
        "processed_points": int(len(test)), "inference_windows": len(starts),
        "coverage_ratio": 1.0, "window": WINDOW, "batch_size": BATCH_SIZE,
        "tail_policy": "final full window anchored at N-window",
        "warmup": WARMUP, "latency_repeats": LATENCY_REPEATS,
        "full_test_repeats": FULL_REPEATS, "dtype": "float32",
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_hash_before,
        "checkpoint_modified": checkpoint_hash_before != checkpoint_hash_after,
        "source_integrity_unchanged": before == after,
        "training": False, "label_access": False, "threshold_computed": False,
        "evaluator_called": False, "all_windows_on_gpu": False,
        "host_to_device_transfer_excluded": True, "data_loading_excluded": True,
        "gpu_peak_includes": ["model parameters", "current input batch", "intermediate activations"],
        "peak_process_rss_mib": None, "peak_uss_mib": None,
    }
    if payload["checkpoint_modified"] or not payload["source_integrity_unchanged"]:
        payload["status"] = "FAILED"
    save_json(OUT / dataset / "efficiency" / f"efficiency_{implementation}.json", payload)
    if payload["status"] != "COMPLETE":
        raise SystemExit(f"{dataset}/{implementation} integrity audit failed")


def monitor(dataset: str, implementation: str) -> dict:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required for Peak RSS/USS monitoring") from exc
    script = Path(__file__).resolve()
    command = [sys.executable, str(script), "--worker", "--dataset", dataset, "--implementation", implementation]
    process = subprocess.Popen(command, cwd=ROOT)
    monitored = psutil.Process(process.pid)
    peak_rss = peak_uss = 0
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, int(monitored.memory_info().rss))
            peak_uss = max(peak_uss, int(monitored.memory_full_info().uss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.01)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    path = OUT / dataset / "efficiency" / f"efficiency_{implementation}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["peak_process_rss_mib"] = peak_rss / 2**20
    payload["peak_uss_mib"] = peak_uss / 2**20
    payload["process_memory_sampling_interval_s"] = 0.01
    save_json(path, payload)
    return payload


def compare(dataset: str, old: dict, optimized: dict) -> dict:
    lower = ("latency_b1", "latency_b128")
    result = {"dataset": dataset, "old": old, "optimized": optimized, "differences": {}}
    pairs = {
        "latency_b1_ms": (old["latency_b1"]["mean_ms"], optimized["latency_b1"]["mean_ms"]),
        "latency_b128_ms": (old["latency_b128"]["mean_ms"], optimized["latency_b128"]["mean_ms"]),
        "full_test_time_s": (old["full_test_time_mean_s"], optimized["full_test_time_mean_s"]),
        "gpu_peak_mib": (old["gpu_peak_mib"], optimized["gpu_peak_mib"]),
        "gpu_incremental_mib": (old["gpu_incremental_mib"], optimized["gpu_incremental_mib"]),
        "peak_rss_mib": (old["peak_process_rss_mib"], optimized["peak_process_rss_mib"]),
        "peak_uss_mib": (old["peak_uss_mib"], optimized["peak_uss_mib"]),
    }
    for name, (a, b) in pairs.items():
        result["differences"][name] = {
            "absolute_old_minus_optimized": a - b,
            "percentage_reduction": (a - b) / a * 100.0 if a else 0.0,
            "speedup_old_over_optimized": a / b if b else None,
        }
    result["differences"]["throughput_points_per_s"] = {
        "absolute_optimized_minus_old": optimized["throughput_points_per_s"] - old["throughput_points_per_s"],
        "percentage_improvement": (
            optimized["throughput_points_per_s"] / old["throughput_points_per_s"] - 1.0
        ) * 100.0,
    }
    result["status"] = "COMPLETE"
    save_json(OUT / dataset / "efficiency" / "efficiency_comparison.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--implementation", choices=("old", "optimized"))
    parser.add_argument("--old-only", action="store_true")
    parser.add_argument("--optimized-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if not args.implementation:
            raise SystemExit("--worker requires --implementation")
        worker(args.dataset, args.implementation)
        return
    if args.old_only and args.optimized_only:
        raise SystemExit("--old-only and --optimized-only are mutually exclusive")
    old = None if args.optimized_only else monitor(args.dataset, "old")
    optimized = None if args.old_only else monitor(args.dataset, "optimized")
    if old is not None and optimized is not None:
        compare(args.dataset, old, optimized)
    print(f"[{args.dataset}] efficiency benchmark completed", flush=True)


if __name__ == "__main__":
    main()
