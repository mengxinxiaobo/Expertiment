#!/usr/bin/env python3
"""Shared inference-only benchmark for a formal TranAD checkpoint.

This program never opens PSM_test_label.npy and contains no threshold,
prediction, evaluator, POT/SPOT, or metric code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.tranad_dataset_adapter import (
    TranADPSMDataAdapter,
    import_official_tranad_class,
    load_tranad_psm_config,
)


OUTPUT_ROOT = ROOT / "results" / "PSM_TRANAD_RESULTS" / "Efficiency"
COMPARISON_CSV = OUTPUT_ROOT / "comparison_efficiency.csv"
COMPARISON_JSON = OUTPUT_ROOT / "comparison_efficiency.json"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol_efficiency.json"
CHECKPOINT_PATH = (
    ROOT
    / "results"
    / "PSM_TRANAD_RESULTS"
    / "checkpoints"
    / "TranAD_PSM_state_dict.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "072d6014ec66feeef46398f91422507d10c10f94728c85feab0430a3fb48525a"
)
WARMUP = 30
LATENCY_REPEAT = 200
FULL_TEST_REPEAT = 20
FULL_TEST_BATCH = 128
EXPECTED_TEST_POINTS = 87841
DATASET_NAME = "PSM"
WINDOW = 10
INPUT_C = 25
EXPECTED_PARAMETERS = 57273


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def stage_start(name: str) -> float:
    print("=" * 72, flush=True)
    print(name, flush=True)
    print(f"Start Time: {now_iso()}", flush=True)
    print("=" * 72, flush=True)
    return time.perf_counter()


def stage_end(name: str, started: float) -> float:
    elapsed = time.perf_counter() - started
    print(f"{name} Finished", flush=True)
    print(f"End Time: {now_iso()}", flush=True)
    print(f"Elapsed Time: {duration(elapsed)}", flush=True)
    return elapsed


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_fresh(overwrite: bool) -> None:
    protected = (COMPARISON_CSV, COMPARISON_JSON, PROTOCOL_PATH)
    existing = [str(path) for path in protected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Efficiency result already exists: {existing}. "
            "Use --overwrite only for an authorized rerun."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_model(
    config: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_hash = sha256(CHECKPOINT_PATH)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "TranAD checkpoint SHA-256 mismatch: "
            f"{checkpoint_hash} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    before = CHECKPOINT_PATH.stat()
    model_class = import_official_tranad_class(
        float(config["training"]["learning_rate"])
    )
    model = model_class(int(config["model"]["input_channels"])).float()
    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    return model, {
        "sha256_before": checkpoint_hash,
        "size_before": int(before.st_size),
        "mtime_ns_before": int(before.st_mtime_ns),
    }


def actual_parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return int(total), int(trainable)


def serialized_state_dict_kib(model: torch.nn.Module) -> float:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="tranad_state_dict_",
            suffix=".pt",
            dir=OUTPUT_ROOT,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        torch.save(model.state_dict(), temporary_path)
        return float(temporary_path.stat().st_size / 1024.0)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@torch.no_grad()
def formal_score_forward(model: torch.nn.Module, windows: torch.Tensor) -> torch.Tensor:
    """Full two-phase TranAD forward and the frozen PSM anomaly score."""

    source = windows.permute(1, 0, 2)
    target = source[-1].unsqueeze(0)
    _first, second = model(source, target)
    score = (second - target).square()[0].mean(dim=1)
    expected = (windows.shape[0],)
    if tuple(score.shape) != expected:
        raise RuntimeError(f"Unexpected TranAD score shape {tuple(score.shape)}")
    return score


def cuda_event_ms(
    operation: Callable[[], torch.Tensor],
    warmup: int,
    repeat: int,
) -> np.ndarray:
    with torch.no_grad():
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        samples = np.empty(repeat, dtype=np.float64)
        for index in range(repeat):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = operation()
            end.record()
            torch.cuda.synchronize()
            samples[index] = float(start.elapsed_time(end))
            del output
    return samples


def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    windows = torch.randn(
        batch_size, WINDOW, INPUT_C, device=device, dtype=torch.float32
    )
    samples = cuda_event_ms(
        lambda: formal_score_forward(model, windows),
        WARMUP,
        LATENCY_REPEAT,
    )
    result = {
        "batch_size": batch_size,
        "input_shape": list(windows.shape),
        "mean_ms": float(samples.mean()),
        "std_ms": float(samples.std(ddof=0)),
        "minimum_ms": float(samples.min()),
        "maximum_ms": float(samples.max()),
        "warmup": WARMUP,
        "repeat": LATENCY_REPEAT,
    }
    del windows
    torch.cuda.synchronize()
    return result


def prepare_full_test_windows() -> tuple[torch.Tensor, dict[str, Any]]:
    """Materialize verified adapter windows on CPU, outside all timing."""

    adapter = TranADPSMDataAdapter("score")
    loader = adapter.loader("test", batch_size=FULL_TEST_BATCH, shuffle=False)
    windows = torch.empty(
        EXPECTED_TEST_POINTS, WINDOW, INPUT_C, dtype=torch.float32, device="cpu"
    )
    cursor = 0
    for batch in loader:
        count = int(batch.shape[0])
        windows[cursor : cursor + count].copy_(batch)
        cursor += count
    adapter.assert_label_free()
    if cursor != EXPECTED_TEST_POINTS:
        raise RuntimeError(
            f"Actual TranAD windows {cursor} != expected {EXPECTED_TEST_POINTS}"
        )
    if tuple(windows.shape) != (EXPECTED_TEST_POINTS, WINDOW, INPUT_C):
        raise RuntimeError(f"Unexpected full-test windows: {tuple(windows.shape)}")
    audit = adapter.audit()
    if any("label" in path.lower() for path in audit["files_accessed"]):
        raise RuntimeError("Efficiency preparation accessed a label file")
    return windows, audit


@torch.no_grad()
def full_test_operation(
    model: torch.nn.Module,
    windows_gpu: torch.Tensor,
) -> torch.Tensor:
    parts = [
        formal_score_forward(model, batch)
        for batch in windows_gpu.split(FULL_TEST_BATCH, dim=0)
    ]
    output = torch.cat(parts, dim=0)
    if output.numel() != windows_gpu.shape[0]:
        raise RuntimeError("Full-test TranAD score does not cover every window")
    return output


def measure_full_test(
    model: torch.nn.Module,
    windows_gpu: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    samples = np.empty(FULL_TEST_REPEAT, dtype=np.float64)
    benchmark_started = time.perf_counter()
    last_output: torch.Tensor | None = None
    with torch.no_grad():
        for repeat_index in range(1, FULL_TEST_REPEAT + 1):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = full_test_operation(model, windows_gpu)
            end.record()
            torch.cuda.synchronize()
            current_seconds = float(start.elapsed_time(end) / 1000.0)
            samples[repeat_index - 1] = current_seconds
            if last_output is not None:
                del last_output
            last_output = output
            elapsed = time.perf_counter() - benchmark_started
            average_wall = elapsed / repeat_index
            remaining = average_wall * (FULL_TEST_REPEAT - repeat_index)
            print(
                f"Repeat: {repeat_index}/{FULL_TEST_REPEAT} | "
                f"Current full-test time: {current_seconds:.6f}s | "
                f"Completed: {repeat_index / FULL_TEST_REPEAT:.0%} | "
                f"Estimated remaining time: {duration(remaining)}",
                flush=True,
            )
    assert last_output is not None
    actual_points = int(last_output.numel())
    mean_seconds = float(samples.mean())
    return {
        "mean_seconds": mean_seconds,
        "std_seconds": float(samples.std(ddof=0)),
        "minimum_seconds": float(samples.min()),
        "maximum_seconds": float(samples.max()),
        "repeat": FULL_TEST_REPEAT,
        "batch_size": FULL_TEST_BATCH,
        "actual_processed_points": actual_points,
        "window_count": int(windows_gpu.shape[0]),
        "score_length": actual_points,
        "throughput_points_per_second": float(actual_points / mean_seconds),
    }, last_output


def measure_gpu_memory(
    model: torch.nn.Module,
    windows_gpu: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device))
    with torch.no_grad():
        output = full_test_operation(model, windows_gpu)
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated(device))
    output_length = int(output.numel())
    del output
    if output_length != EXPECTED_TEST_POINTS:
        raise RuntimeError("Memory pass score length mismatch")
    mib = 1024.0 * 1024.0
    return {
        "baseline_allocated_mib": float(baseline / mib),
        "peak_allocated_mib": float(peak / mib),
        "incremental_allocated_mib": float((peak - baseline) / mib),
        "peak_includes": [
            "model parameters",
            "full preloaded GPU test-window tensor",
            "intermediate activations",
            "anomaly score outputs",
        ],
        "baseline_recorded_after": [
            "model moved to GPU",
            "full test-window tensor moved to GPU",
        ],
    }


def verify_checkpoint_unchanged(before: dict[str, Any]) -> dict[str, Any]:
    after = CHECKPOINT_PATH.stat()
    after_hash = sha256(CHECKPOINT_PATH)
    unchanged = (
        after_hash == before["sha256_before"]
        and int(after.st_size) == before["size_before"]
        and int(after.st_mtime_ns) == before["mtime_ns_before"]
    )
    if not unchanged:
        raise RuntimeError("Formal TranAD checkpoint changed during benchmark")
    return {
        "path": str(CHECKPOINT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "sha256": after_hash,
        "size_bytes": int(after.st_size),
        "modified": False,
    }


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    ensure_fresh(args.overwrite)
    config = load_tranad_psm_config()
    set_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("TranAD efficiency benchmark requires CUDA")
    device = torch.device("cuda:0")

    print(f"{DATASET_NAME} TranAD inference-only efficiency benchmark", flush=True)
    print("training=False label_access=False evaluator_called=False", flush=True)
    print("threshold=False prediction=False POT/SPOT/bf_search=False", flush=True)
    print("data_loading=excluded host_to_device_transfer=excluded", flush=True)

    started = stage_start("Stage 1: Load and verify formal checkpoint")
    model, checkpoint_before = load_model(config, device)
    total_parameters, trainable_parameters = actual_parameter_counts(model)
    if (
        total_parameters != EXPECTED_PARAMETERS
        or trainable_parameters != EXPECTED_PARAMETERS
    ):
        raise RuntimeError(
            f"Unexpected actual parameters: {total_parameters}/{trainable_parameters}"
        )
    stage_end("Stage 1: Checkpoint Verification", started)

    started = stage_start("Stage 2: Prepare official-semantics test windows")
    windows_cpu, data_audit = prepare_full_test_windows()
    windows_gpu = windows_cpu.to(device=device, dtype=torch.float32)
    del windows_cpu
    torch.cuda.synchronize()
    stage_end("Stage 2: Input Preparation (excluded from timing)", started)

    started = stage_start("Stage 3: Serialize state dict and count parameters")
    state_dict_kib = serialized_state_dict_kib(model)
    stage_end("Stage 3: Model Size", started)

    started = stage_start("Stage 4: Latency batch=1")
    latency_b1 = measure_latency(model, device, 1)
    stage_end("Stage 4: Latency batch=1", started)

    started = stage_start("Stage 5: Latency batch=128")
    latency_b128 = measure_latency(model, device, 128)
    stage_end("Stage 5: Latency batch=128", started)

    started = stage_start("Stage 6: Full-test inference, 20 repeats")
    full_test, last_score = measure_full_test(model, windows_gpu)
    stage_end("Stage 6: Full-test Inference", started)
    if full_test["actual_processed_points"] != EXPECTED_TEST_POINTS:
        raise RuntimeError(
            f"Actual processed points do not cover the {DATASET_NAME} test split"
        )
    del last_score

    started = stage_start("Stage 7: GPU memory")
    gpu_memory = measure_gpu_memory(model, windows_gpu, device)
    stage_end("Stage 7: GPU Memory", started)

    checkpoint = verify_checkpoint_unchanged(checkpoint_before)
    total_elapsed = time.perf_counter() - total_started
    environment = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    row = {
        "Model": "TranAD",
        "Total Parameters": total_parameters,
        "Trainable Parameters": trainable_parameters,
        "State Dict (KiB)": state_dict_kib,
        "Latency B=1 Mean (ms)": latency_b1["mean_ms"],
        "Latency B=1 Std (ms)": latency_b1["std_ms"],
        "Latency B=128 Mean (ms)": latency_b128["mean_ms"],
        "Latency B=128 Std (ms)": latency_b128["std_ms"],
        "Full Test Time Mean (s)": full_test["mean_seconds"],
        "Full Test Time Std (s)": full_test["std_seconds"],
        "Actual Processed Points": full_test["actual_processed_points"],
        "Throughput (points/s)": full_test["throughput_points_per_second"],
        "GPU Peak (MiB)": gpu_memory["peak_allocated_mib"],
        "GPU Incremental (MiB)": gpu_memory["incremental_allocated_mib"],
    }
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    details = {
        "dataset": DATASET_NAME,
        "model": "TranAD",
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
            "measurement": "actual model parameter traversal",
        },
        "state_dict": {
            "kib": state_dict_kib,
            "measurement": "torch.save(model.state_dict(), temporary_path)",
            "temporary_file_deleted": True,
        },
        "latency_batch_1": latency_b1,
        "latency_batch_128": latency_b128,
        "full_test": full_test,
        "gpu_memory": gpu_memory,
        "checkpoint": checkpoint,
        "environment": environment,
        "total_benchmark_seconds": total_elapsed,
    }
    write_json(COMPARISON_JSON, details)

    protocol = {
        "dataset": DATASET_NAME,
        "model": "TranAD",
        "seed": 42,
        "window": WINDOW,
        "input_c": INPUT_C,
        "dtype": "float32",
        "model_input_shape": f"[batch,{WINDOW},{INPUT_C}]",
        "model_internal_input_shape": f"[{WINDOW},batch,{INPUT_C}]",
        "forward_scope": "complete two-phase TranAD forward plus formal anomaly score",
        "warmup": WARMUP,
        "latency_repeat": LATENCY_REPEAT,
        "full_test_repeat": FULL_TEST_REPEAT,
        "full_test_batch_size": FULL_TEST_BATCH,
        "actual_processed_points": full_test["actual_processed_points"],
        "window_count": full_test["window_count"],
        "score_length": full_test["score_length"],
        "gpu": environment["gpu"],
        "pytorch_version": environment["torch_version"],
        "cuda_version": environment["cuda_version"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "data_access": data_audit,
        "memory_scope": gpu_memory,
        "audit": {
            "training": False,
            "label_access": False,
            "evaluator_called": False,
            "threshold_computed": False,
            "prediction_generated": False,
            "pot_called": False,
            "spot_called": False,
            "bf_search_called": False,
            "checkpoint_modified": False,
            "model_source_modified": False,
            "data_loading_excluded": True,
            "standard_scaler_excluded": True,
            "cpu_preprocessing_excluded": True,
            "host_to_device_transfer_excluded": True,
        },
    }
    write_json(PROTOCOL_PATH, protocol)

    print("=" * 72, flush=True)
    print(f"TranAD {DATASET_NAME} efficiency benchmark completed", flush=True)
    print(f"Total benchmark time: {duration(total_elapsed)}", flush=True)
    print(
        f"Parameters total={total_parameters} trainable={trainable_parameters} "
        f"state_dict={state_dict_kib:.3f}KiB",
        flush=True,
    )
    print(
        f"Latency B1={latency_b1['mean_ms']:.6f}±{latency_b1['std_ms']:.6f}ms "
        f"B128={latency_b128['mean_ms']:.6f}±{latency_b128['std_ms']:.6f}ms",
        flush=True,
    )
    print(
        f"Full test={full_test['mean_seconds']:.6f}±"
        f"{full_test['std_seconds']:.6f}s "
        f"points={full_test['actual_processed_points']} "
        f"throughput={full_test['throughput_points_per_second']:.2f} points/s",
        flush=True,
    )
    print(
        f"GPU peak={gpu_memory['peak_allocated_mib']:.3f}MiB "
        f"incremental={gpu_memory['incremental_allocated_mib']:.3f}MiB",
        flush=True,
    )
    print(f"comparison_csv={COMPARISON_CSV}", flush=True)
    print(f"protocol={PROTOCOL_PATH}", flush=True)


if __name__ == "__main__":
    main()
