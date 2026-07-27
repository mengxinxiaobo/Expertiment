#!/usr/bin/env python3
"""Inference-only SKAB efficiency benchmark for ASCA-AD V4, PPLAD and LTFAD."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "results" / "SKAB_EFFICIENCY"
WARMUP = 30
REPEATS = 200
FULL_TEST_REPEATS = 20
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("asca", "pplad", "ltfad"))
    return parser.parse_args()


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_scaled_test() -> np.ndarray:
    dataset = ROOT / "dataset" / "SKAB"
    train = np.asarray(np.load(dataset / "SKAB_train.npy", allow_pickle=False), dtype=np.float32)
    test = np.asarray(np.load(dataset / "SKAB_test.npy", allow_pickle=False), dtype=np.float32)
    if train.shape != (12450, 8) or test.shape != (5710, 8):
        raise RuntimeError(f"Unexpected SKAB shapes: train={train.shape}, test={test.shape}")
    scaler = StandardScaler()
    scaler.fit(train)
    return scaler.transform(test).astype(np.float32, copy=False)


def non_overlapping_windows(data: np.ndarray, window: int) -> torch.Tensor:
    starts = range(0, len(data) - window + 1, window)
    values = np.stack([data[start : start + window] for start in starts])
    return torch.from_numpy(np.ascontiguousarray(values))


def load_model(model_name: str, device: torch.device):
    if model_name == "asca":
        import scripts.benchmarks.generate_asca_v4_skab_score as generator

        model = generator.AdaptiveSparseAnchorCompetitiveModelV4(**generator.MODEL_CONFIG)
        state, checkpoint_config = generator.load_checkpoint_state(generator.CHECKPOINT)
        generator.validate_checkpoint_config(checkpoint_config)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        solver = generator.build_inference_solver(model, device)
        adapter = generator.ASCAV4ScoreAdapter(solver).eval()
        return "ASCA-AD V4", 100, model, adapter.window_scores, generator.CHECKPOINT

    if model_name == "pplad":
        import scripts.benchmarks.generate_pplad_skab_score as generator

        solver = generator.build_solver([], device, 128)
        checkpoint = (
            ROOT
            / "checkpoints"
            / "SKAB_OFFICIAL_DEFAULT_VS_V4_BEST"
            / "ORIGINAL"
            / "SKAB_original_official_default_state.pt"
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        solver.model.load_state_dict(payload["model"], strict=True)
        solver.model.eval()
        adapter = generator.PPLADScoreAdapter(solver, generator.official_solver).eval()
        return "PPLAD", 60, solver.model, adapter.window_scores, checkpoint

    import scripts.benchmarks.generate_ltfad_skab_score as generator

    solver = generator.build_solver([], device, 128)
    checkpoint = OUTPUT_ROOT.parent / "SKAB_PAPER_RESULTS" / "LTFAD" / "LTFAD_official_state_dict.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    solver.model.load_state_dict(payload["model"], strict=True)
    solver.model.eval()
    adapter = generator.LTFADScoreAdapter(solver, generator.official_solver).eval()
    return "LTFAD", 90, solver.model, adapter.window_scores, checkpoint


def cuda_latency_ms(call: Callable[[], torch.Tensor]) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(WARMUP):
            call()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    with torch.no_grad():
        for start, end in zip(starts, ends):
            start.record()
            call()
            end.record()
    torch.cuda.synchronize()
    values = np.asarray(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=np.float64,
    )
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def full_test_timing(
    score_call: Callable[[torch.Tensor], torch.Tensor],
    batches: list[torch.Tensor],
) -> tuple[dict[str, float], int]:
    def run_once() -> None:
        for batch in batches:
            score_call(batch)

    with torch.no_grad():
        for _ in range(WARMUP):
            run_once()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(FULL_TEST_REPEATS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(FULL_TEST_REPEATS)]
    with torch.no_grad():
        for start, end in zip(starts, ends):
            start.record()
            run_once()
            end.record()
    torch.cuda.synchronize()
    seconds = np.asarray(
        [start.elapsed_time(end) / 1000.0 for start, end in zip(starts, ends)],
        dtype=np.float64,
    )
    return (
        {
            "mean": float(seconds.mean()),
            "std": float(seconds.std()),
            "p50": float(np.percentile(seconds, 50)),
            "p95": float(np.percentile(seconds, 95)),
        },
        len(batches),
    )


def memory_measurement(
    score_call: Callable[[torch.Tensor], torch.Tensor],
    batches: list[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch in batches:
            score_call(batch)
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    mib = 1024.0**2
    return {
        "baseline_allocated_mib": baseline / mib,
        "peak_allocated_mib": peak / mib,
        "incremental_allocated_mib": max(0, peak - baseline) / mib,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_worker(model_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested CUDA-event benchmark")
    set_seed()
    device = torch.device("cuda:0")
    display, window, model, score_call, checkpoint = load_model(model_name, device)
    model.eval()
    total_parameters = int(sum(p.numel() for p in model.parameters()))
    trainable_parameters = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    model_dir = OUTPUT_ROOT / display.replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    state_path = model_dir / "state_dict.pt"
    torch.save(model.state_dict(), state_path)
    state_bytes = state_path.stat().st_size

    test = load_scaled_test()
    windows_cpu = non_overlapping_windows(test, window)
    full_batches = [
        item.to(device)
        for item in torch.split(windows_cpu, 128, dim=0)
    ]
    first = windows_cpu[0:1].to(device)
    batch1 = first
    batch128 = first.repeat(128, 1, 1)

    latency1 = cuda_latency_ms(lambda: score_call(batch1))
    latency128 = cuda_latency_ms(lambda: score_call(batch128))
    full_test, full_batches_count = full_test_timing(score_call, full_batches)
    full_points = int(windows_cpu.shape[0] * window)
    memory = memory_measurement(score_call, full_batches, device)

    payload = {
        "model": display,
        # Keep the legacy field for existing dataset aggregators.
        "parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "window_size": window,
        "channels": int(windows_cpu.shape[-1]),
        "input_shape": {
            "batch_1": list(batch1.shape),
            "batch_128": list(batch128.shape),
        },
        "device": str(device),
        "dtype": str(windows_cpu.dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": WARMUP,
        "repeat": REPEATS,
        "state_dict": {
            "path": str(state_path.relative_to(ROOT)),
            "bytes": state_bytes,
            "kib": state_bytes / 1024.0,
            "mib": state_bytes / 1024.0**2,
        },
        "latency_batch_1_ms": latency1,
        "latency_batch_128_ms": latency128,
        "full_test": {
            **full_test,
            "windows": int(windows_cpu.shape[0]),
            "batches": full_batches_count,
            "points": full_points,
            "throughput_points_per_second": full_points / full_test["mean"],
        },
        "gpu_memory": memory,
        "protocol": {
            "inference_scope": "standardized window -> model-specific anomaly score",
            "torch_no_grad": True,
            "model_eval": True,
            "cuda_event_timing": True,
            "warmup": WARMUP,
            "repeat": REPEATS,
            "full_test_repeat": FULL_TEST_REPEATS,
            "test_window_step": window,
            "uses_label": False,
            "calls_evaluator": False,
            "training": False,
            "label_access": False,
            "evaluator_called": False,
            "includes_data_loader_time": False,
            "includes_host_to_device_transfer": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    write_json(model_dir / "efficiency.json", payload)
    print(
        f"{display}: params={total_parameters}, trainable={trainable_parameters}, "
        f"state={state_bytes / 1024:.3f}KiB, "
        f"b1={latency1['mean']:.6f}ms, b128={latency128['mean']:.6f}ms, "
        f"full={full_test['mean']:.6f}s, points={full_points}"
    )


def aggregate() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for worker in ("asca", "pplad", "ltfad"):
        subprocess.run([sys.executable, str(script), "--worker", worker], check=True)
    paths = (
        OUTPUT_ROOT / "ASCA-AD_V4" / "efficiency.json",
        OUTPUT_ROOT / "PPLAD" / "efficiency.json",
        OUTPUT_ROOT / "LTFAD" / "efficiency.json",
    )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    fields = [
        "Model",
        "Parameters",
        "State Dict(KiB)",
        "Latency batch=1(ms)",
        "Latency batch=128(ms)",
        "Full Test Time(s)",
        "Throughput(points/s)",
        "GPU Peak Memory(MiB)",
        "GPU Incremental Memory(MiB)",
    ]
    csv_path = OUTPUT_ROOT / "efficiency_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "Model": item["model"],
                    "Parameters": item["parameters"],
                    "State Dict(KiB)": item["state_dict"]["kib"],
                    "Latency batch=1(ms)": item["latency_batch_1_ms"]["mean"],
                    "Latency batch=128(ms)": item["latency_batch_128_ms"]["mean"],
                    "Full Test Time(s)": item["full_test"]["mean"],
                    "Throughput(points/s)": item["full_test"]["throughput_points_per_second"],
                    "GPU Peak Memory(MiB)": item["gpu_memory"]["peak_allocated_mib"],
                    "GPU Incremental Memory(MiB)": item["gpu_memory"]["incremental_allocated_mib"],
                }
            )
    write_json(
        OUTPUT_ROOT / "efficiency_comparison.json",
        {
            "dataset": "SKAB",
            "metric_scope": "inference efficiency only",
            "models": rows,
        },
    )
    print(f"csv={csv_path}")


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args.worker)
    else:
        aggregate()


if __name__ == "__main__":
    main()
