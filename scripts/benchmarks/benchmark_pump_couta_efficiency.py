#!/usr/bin/env python3
"""Streaming, inference-only PUMP efficiency benchmark for official COUTA."""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.couta_dataset_adapter import (
    PUMPCOUTADataAdapter, load_pump_couta_config, restore_from_bundle, sha256,
)

OUT = ROOT / "results" / "PUMP_COUTA_RESULTS"
EFF = OUT / "Efficiency"
BUNDLE = OUT / "checkpoints" / "COUTA_PUMP_bundle.pt"
DATASET_NAME = "PUMP"
FULL_BATCH_SIZE = 64
WARMUP = 30
LATENCY_REPEAT = 200
FULL_TEST_REPEAT = 20


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def stage(name: str) -> float:
    print("=" * 72, flush=True)
    print(name, flush=True)
    print(f"Start Time: {now()}", flush=True)
    return time.perf_counter()


def finish(name: str, started: float) -> float:
    elapsed = time.perf_counter() - started
    print(f"{name} Finished", flush=True)
    print(f"End Time: {now()}", flush=True)
    print(f"Elapsed Time: {elapsed:.3f}s", flush=True)
    return elapsed


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def score_batch(net, center, batch: torch.Tensor) -> torch.Tensor:
    output = net(batch)
    return (torch.sum((output[0] - center) ** 2, dim=1) +
            torch.sum((output[1] - center) ** 2, dim=1))


def latency(net, center, shape: tuple[int, ...]) -> tuple[float, float]:
    tensor = torch.zeros(shape, dtype=torch.float32, device="cuda")
    for _ in range(WARMUP):
        score_batch(net, center, tensor)
    torch.cuda.synchronize()
    timings = []
    for _ in range(LATENCY_REPEAT):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        score_batch(net, center, tensor)
        end.record()
        end.synchronize()
        timings.append(float(begin.elapsed_time(end)))
    del tensor
    return statistics.mean(timings), statistics.pstdev(timings)


def cpu_window_view(test_scaled: np.ndarray, seq_len: int) -> torch.Tensor:
    # View only: no full GPU residency and no expanded NumPy window copy.
    base = torch.from_numpy(np.ascontiguousarray(test_scaled))
    return base.unfold(0, seq_len, 1).permute(0, 2, 1)


def full_test_once(net, center, windows: torch.Tensor) -> tuple[float, int]:
    """Run every window; sum only per-batch CUDA forward events (H2D excluded)."""
    gpu_seconds = 0.0
    produced = 0
    for start in range(0, windows.shape[0], FULL_BATCH_SIZE):
        stop = min(start + FULL_BATCH_SIZE, windows.shape[0])
        # CPU materialization and H2D both occur before the timed CUDA event.
        gpu_batch = windows[start:stop].contiguous().to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        scores = score_batch(net, center, gpu_batch)
        end.record()
        end.synchronize()
        gpu_seconds += float(begin.elapsed_time(end)) / 1000.0
        produced += int(scores.numel())
        del scores, gpu_batch
    return gpu_seconds, produced


def pick(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return default


def mean_std(mean: str, std: str) -> str:
    return f"{float(mean):.6f} ± {float(std):.6f}" if std else f"{float(mean):.6f} ± N/A"


def aggregate_efficiency() -> dict[str, object]:
    sources = [
        ROOT / "results" / "PUMP_PAPER_RESULTS" / "Efficiency" / "comparison_efficiency.csv",
        ROOT / "results" / "PUMP_TRANAD_RESULTS" / "Efficiency" / "comparison_efficiency.csv",
        EFF / "comparison_efficiency.csv",
    ]
    fields = ["Model", "Params", "State Dict (KiB)", "B=1 mean ± std (ms)",
              "B=128 mean ± std (ms)", "Full Test mean ± std (s)",
              "Throughput (points/s)", "GPU Peak (MiB)"]
    rows = []
    coverage: dict[str, int] = {}
    for path in sources:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                model = row["Model"]
                b1 = pick(row, "Latency_B1(ms)", "Latency B=1 Mean (ms)")
                b128 = pick(row, "Latency_B128(ms)", "Latency B=128 Mean (ms)")
                full = pick(row, "Full_Test_Time(s)", "Full Test Time Mean (s)")
                points = int(float(pick(row, "Actual Processed Points", default={
                    "ASCA-AD V4": "203100", "PPLAD": "203160", "LTFAD": "203130",
                    "COUTA": "203165"}.get(model, "203165"))))
                coverage[model] = points
                rows.append({
                    "Model": model,
                    "Params": pick(row, "Parameters", "Total Parameters"),
                    "State Dict (KiB)": pick(row, "State Dict(KiB)", "State Dict (KiB)"),
                    "B=1 mean ± std (ms)": mean_std(b1, pick(row, "Latency B=1 Std (ms)")),
                    "B=128 mean ± std (ms)": mean_std(b128, pick(row, "Latency B=128 Std (ms)")),
                    "Full Test mean ± std (s)": mean_std(full, pick(row, "Full Test Time Std (s)")),
                    "Throughput (points/s)": pick(row, "Throughput(points/s)", "Throughput (points/s)"),
                    "GPU Peak (MiB)": pick(row, "GPU_Peak(MiB)", "GPU Peak (MiB)"),
                })
    expected = {"ASCA-AD V4", "PPLAD", "LTFAD", "TranAD", "COUTA"}
    if {row["Model"] for row in rows} != expected:
        raise RuntimeError(f"Unexpected PUMP efficiency model set: {[row['Model'] for row in rows]}")
    output = OUT / "five_model_efficiency_comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    coverage_equal = len(set(coverage.values())) == 1
    if not coverage_equal:
        print("[WARNING] Existing legacy efficiency rows cover different point counts: "
              f"{coverage}. Table is generated transparently; strict full-test-time "
              "comparability is not claimed.", flush=True)
    return {"path": str(output.relative_to(ROOT)).replace("\\", "/"),
            "processed_points_by_model": coverage,
            "processed_points_equal": coverage_equal}


def main() -> None:
    total_started = stage(f"{DATASET_NAME} COUTA streaming inference efficiency benchmark")
    config = load_pump_couta_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not BUNDLE.exists():
        raise FileNotFoundError(f"Formal checkpoint missing: {BUNDLE}")
    checkpoint_before = sha256(BUNDLE)

    # All disk loading, scaling and CPU window construction occur before timing.
    adapter = PUMPCOUTADataAdapter("score")
    test = adapter.load_test()
    bundle = torch.load(BUNDLE, map_location="cuda:0", weights_only=False)
    model, scaler = restore_from_bundle(bundle, "cuda:0")
    net, center = model.net.eval(), model.c
    test_scaled = np.asarray(scaler.transform(test), dtype=np.float32)
    if not np.isfinite(test_scaled).all():
        raise RuntimeError(f"Non-finite scaled {DATASET_NAME} test")
    seq_len = int(config["model_config"]["seq_len"])
    input_c = int(config["expected_shapes"]["test"][1])
    expected_points = int(config["expected_shapes"]["test"][0])
    expected_windows = expected_points - seq_len + 1
    windows = cpu_window_view(test_scaled, seq_len)
    if tuple(windows.shape) != (expected_windows, seq_len, input_c):
        raise RuntimeError(f"Unexpected CPU windows: {windows.shape}")

    total_parameters = sum(parameter.numel() for parameter in net.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in net.parameters()
                               if parameter.requires_grad)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        temporary_state = Path(handle.name)
    try:
        torch.save(net.state_dict(), temporary_state)
        state_kib = temporary_state.stat().st_size / 1024.0
    finally:
        temporary_state.unlink(missing_ok=True)

    with torch.no_grad():
        started = stage("Stage 1: Native-window batch latency")
        b1_mean, b1_std = latency(net, center, (1, seq_len, input_c))
        b128_mean, b128_std = latency(net, center, (128, seq_len, input_c))
        finish("Stage 1", started)

        started = stage("Stage 2: Streaming full-test GPU-forward timing")
        warmup_batch = windows[:FULL_BATCH_SIZE].contiguous().to("cuda")
        for _ in range(WARMUP):
            score_batch(net, center, warmup_batch)
        torch.cuda.synchronize()
        del warmup_batch
        torch.cuda.empty_cache()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        full_times: list[float] = []
        produced_windows = 0
        for repeat in range(1, FULL_TEST_REPEAT + 1):
            seconds, produced_windows = full_test_once(net, center, windows)
            if produced_windows != expected_windows:
                raise RuntimeError(f"Full-test output mismatch: {produced_windows}/{expected_windows}")
            full_times.append(seconds)
            running_mean = statistics.mean(full_times)
            running_std = statistics.pstdev(full_times)
            print(f"Repeat {repeat}/{FULL_TEST_REPEAT} | Elapsed Time={seconds:.6f}s | "
                  f"Running Mean={running_mean:.6f}s | Running Std={running_std:.6f}s",
                  flush=True)
        peak_allocated = torch.cuda.max_memory_allocated()
        finish("Stage 2", started)

    full_mean = statistics.mean(full_times)
    full_std = statistics.pstdev(full_times)
    row = {
        "Model": "COUTA", "Total Parameters": total_parameters,
        "State Dict (KiB)": state_kib,
        "Latency B=1 Mean (ms)": b1_mean, "Latency B=1 Std (ms)": b1_std,
        "Latency B=128 Mean (ms)": b128_mean, "Latency B=128 Std (ms)": b128_std,
        "Full Test Time Mean (s)": full_mean, "Full Test Time Std (s)": full_std,
        "Throughput (points/s)": expected_points / full_mean,
        "GPU Peak (MiB)": peak_allocated / 2**20,
    }
    details = {
        **row, "Trainable Parameters": trainable_parameters,
        "Processed Points": expected_points, "Inference Windows": expected_windows,
        "GPU Incremental (MiB)": (peak_allocated - baseline_allocated) / 2**20,
        "Full Test Batch Size": FULL_BATCH_SIZE, "All Windows On GPU": False,
    }
    EFF.mkdir(parents=True, exist_ok=True)
    with (EFF / "comparison_efficiency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    save_json(EFF / "comparison_efficiency.json", details)

    checkpoint_after = sha256(BUNDLE)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("Formal COUTA checkpoint changed during efficiency benchmark")
    audit = {
        "training": False, "label_access": False, "evaluator_called": False,
        "threshold_computed": False, "prediction_generated": False,
        "checkpoint_modified": False, "model_source_modified": False,
        "data_loading_excluded": True, "scaler_excluded": True,
        "cpu_window_construction_excluded": True,
        "host_to_device_transfer_excluded": True, "all_windows_on_gpu": False,
    }
    protocol = {
        "dataset": DATASET_NAME, "model": "COUTA", "seed": 42, "dtype": "float32",
        "seq_len": seq_len, "input_c": input_c,
        "input_shapes": {"b1": [1, seq_len, input_c],
                         "b128": [128, seq_len, input_c],
                         "full_cpu_windows": list(windows.shape)},
        "processed_points": expected_points, "inference_windows": expected_windows,
        "full_test_batch_size": FULL_BATCH_SIZE, "all_windows_on_gpu": False,
        "warmup": WARMUP, "latency_repeat": LATENCY_REPEAT,
        "full_test_repeat": FULL_TEST_REPEAT,
        "checkpoint": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_before,
        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "timing_scope": "per-batch CUDA events around official COUTA net and dual-representation score",
        "gpu_peak_includes": ["model parameters", "current inference batch", "intermediate activations"],
        "gpu_peak_excludes": ["all full-test windows", "training", "labels", "threshold", "evaluator"],
        "gpu_incremental_mib_audit_only": details["GPU Incremental (MiB)"],
        "file_access": adapter.audit(), "audit": audit,
    }
    aggregation = aggregate_efficiency()
    protocol["five_model_aggregation"] = aggregation
    save_json(EFF / "protocol_efficiency.json", protocol)
    save_json(EFF / "COUTA" / "efficiency.json", {"metrics": details, "protocol": protocol})
    finish("Total benchmark", total_started)
    print(json.dumps(details, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
