#!/usr/bin/env python3
"""Inference-only efficiency benchmark for the formal PSM COUTA checkpoint."""

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
    PSMCOUTADataAdapter, load_couta_config, restore_from_bundle, sha256,
)

OUT = ROOT / "results" / "PSM_COUTA_RESULTS"
EFF = OUT / "Efficiency"
BUNDLE = OUT / "checkpoints" / "COUTA_PSM_bundle.pt"
DATASET_NAME = "PSM"
CSV_INCLUDE_INCREMENTAL = True


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def stage(name: str):
    print("=" * 72, flush=True); print(name, flush=True); print(f"Start Time: {now()}", flush=True)
    return time.perf_counter()


def finish(name: str, started: float):
    elapsed = time.perf_counter() - started
    print(f"{name} Finished\nEnd Time: {now()}\nElapsed Time: {elapsed:.3f}s", flush=True)


def score_batch(net, center, x):
    output = net(x)
    return torch.sum((output[0] - center) ** 2, dim=1) + torch.sum((output[1] - center) ** 2, dim=1)


def latency(net, center, shape, warmup, repeat):
    x = torch.zeros(shape, dtype=torch.float32, device="cuda")
    for _ in range(warmup): score_batch(net, center, x)
    torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record(); score_batch(net, center, x); end.record(); end.synchronize()
        values.append(float(begin.elapsed_time(end)))
    return statistics.mean(values), statistics.pstdev(values)


def windows_on_gpu(test_scaled: np.ndarray, seq_len: int) -> torch.Tensor:
    # torch.unfold produces [windows, channels, seq_len]; transpose to official [B,L,C].
    base = torch.from_numpy(np.ascontiguousarray(test_scaled)).to("cuda")
    return base.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()


def full_once(net, center, windows, batch_size, prefix):
    output = torch.empty(prefix + windows.shape[0], dtype=torch.float32, device="cuda")
    output[:prefix] = 0
    for start in range(0, windows.shape[0], batch_size):
        stop = min(start + batch_size, windows.shape[0])
        output[prefix + start:prefix + stop] = score_batch(net, center, windows[start:stop])
    return output


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    total_started = stage(f"{DATASET_NAME} COUTA inference efficiency benchmark")
    config = load_couta_config()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if not BUNDLE.exists(): raise FileNotFoundError(f"Formal checkpoint missing: {BUNDLE}")
    adapter = PSMCOUTADataAdapter("score")
    test = adapter.load_test()
    bundle = torch.load(BUNDLE, map_location="cuda:0", weights_only=False)
    model, scaler = restore_from_bundle(bundle, "cuda:0")
    net, center = model.net.eval(), model.c
    test_scaled = np.asarray(scaler.transform(test), dtype=np.float32)
    seq_len = config["model_config"]["seq_len"]
    prefix = seq_len - 1
    windows = windows_on_gpu(test_scaled, seq_len)
    expected_test = int(config["expected_shapes"]["test"][0])
    expected_input = int(config["expected_shapes"]["test"][1])
    expected_windows = expected_test - seq_len + 1
    if tuple(windows.shape) != (expected_windows, seq_len, expected_input):
        raise RuntimeError(windows.shape)
    total_params = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(net.state_dict(), temporary)
        state_kib = temporary.stat().st_size / 1024.0
    finally:
        temporary.unlink(missing_ok=True)

    warmup, latency_repeat, full_repeat = 30, 200, 20
    with torch.no_grad():
        t = stage("Stage 1: Batch latency")
        b1_mean, b1_std = latency(net, center, (1, seq_len, expected_input), warmup, latency_repeat)
        b128_mean, b128_std = latency(net, center, (128, seq_len, expected_input), warmup, latency_repeat)
        finish("Stage 1", t)

        t = stage("Stage 2: Full-test GPU-only inference")
        for _ in range(warmup): score_batch(net, center, windows[:64])
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        full_times = []
        for repeat in range(1, full_repeat + 1):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(); scores = full_once(net, center, windows, 64, prefix); end.record()
            end.synchronize(); seconds = begin.elapsed_time(end) / 1000.0
            if scores.numel() != expected_test: raise RuntimeError(scores.shape)
            full_times.append(seconds)
            average = sum(full_times) / len(full_times)
            eta = average * (full_repeat - repeat)
            print(f"Repeat {repeat}/{full_repeat} | Current full-test time={seconds:.6f}s | "
                  f"Completed={repeat}/{full_repeat} | Estimated remaining={eta:.1f}s", flush=True)
        peak = torch.cuda.max_memory_allocated()
        finish("Stage 2", t)

    full_mean, full_std = statistics.mean(full_times), statistics.pstdev(full_times)
    actual_points = int(scores.numel())
    row = {
        "Model": "COUTA", "Total Parameters": total_params,
        "Trainable Parameters": trainable, "State Dict (KiB)": state_kib,
        "Latency B=1 Mean (ms)": b1_mean, "Latency B=1 Std (ms)": b1_std,
        "Latency B=128 Mean (ms)": b128_mean, "Latency B=128 Std (ms)": b128_std,
        "Full Test Time Mean (s)": full_mean, "Full Test Time Std (s)": full_std,
        "Actual Processed Points": actual_points,
        "Throughput (points/s)": actual_points / full_mean,
        "GPU Peak (MiB)": peak / 2**20,
        "GPU Incremental (MiB)": (peak - baseline) / 2**20,
    }
    EFF.mkdir(parents=True, exist_ok=True)
    csv_row = dict(row)
    if not CSV_INCLUDE_INCREMENTAL:
        csv_row.pop("GPU Incremental (MiB)")
    with (EFF / "comparison_efficiency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row)); writer.writeheader(); writer.writerow(csv_row)
    save_json(EFF / "comparison_efficiency.json", row)
    audit = {
        "training": False, "label_access": False, "evaluator_called": False,
        "threshold_computed": False, "prediction_generated": False,
        "ray_tune_used": False, "training_ray_used": False,
        "checkpoint_modified": False, "model_source_modified": False,
        "data_loading_excluded": True, "scaler_excluded": True,
        "cpu_window_construction_excluded": True,
        "host_to_device_transfer_excluded": True,
    }
    protocol = {
        "dataset": DATASET_NAME, "model": "COUTA", "seed": 42, "dtype": "float32",
        "seq_len": seq_len, "input_c": expected_input, "batch_size": 64,
        "input_shapes": {"b1": [1,seq_len,expected_input], "b128": [128,seq_len,expected_input],
                         "full_windows": list(windows.shape)},
        "actual_windows": int(windows.shape[0]), "actual_processed_points": actual_points,
        "warmup": warmup, "latency_repeat": latency_repeat, "full_test_repeat": full_repeat,
        "checkpoint": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(BUNDLE),
        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "timing_scope": "GPU-resident windows -> official COUTA net -> dual-representation distance score",
        "gpu_peak_includes": ["model parameters", "GPU input windows", "intermediate activations", "output scores"],
        "file_access": adapter.audit(), "audit": audit,
    }
    save_json(EFF / "protocol_efficiency.json", protocol)
    save_json(EFF / "COUTA" / "efficiency.json", {"metrics": row, "protocol": protocol})
    finish("Total benchmark", total_started)
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
