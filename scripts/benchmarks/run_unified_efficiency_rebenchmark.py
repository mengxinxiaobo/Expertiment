#!/usr/bin/env python3
"""Audit-driven, streaming efficiency rebenchmark for the 28 required cases.

This runner never trains, reads labels, evaluates detections, or overwrites legacy
results.  It consumes the protocol audit as its case manifest and writes only to
results/UNIFIED_EFFICIENCY_REBENCHMARK.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIT_CSV = ROOT / "results" / "EFFICIENCY_PROTOCOL_AUDIT" / "efficiency_protocol_audit.csv"
OUT = ROOT / "results" / "UNIFIED_EFFICIENCY_REBENCHMARK"
WARMUP = 30
FULL_REPEAT = 20
CORE_BATCH = 128
TRANAD_BATCH = 128
COUTA_BATCH = 64
SEED = 42
DATASETS = ("PSM", "SKAB", "MSL", "HAI", "PUMP", "SMD")
MODELS = ("ASCA-AD V4", "PPLAD", "LTFAD", "TranAD", "COUTA")
CORE_MODELS = {"ASCA-AD V4", "PPLAD", "LTFAD"}
MODEL_KEYS = {"ASCA-AD V4": "asca", "PPLAD": "pplad", "LTFAD": "ltfad"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_name(model: str) -> str:
    return model.replace("-", "_").replace(" ", "_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stage(name: str) -> float:
    print("=" * 80, flush=True)
    print(f"Stage: {name}", flush=True)
    print(f"Start Time: {now()}", flush=True)
    return time.perf_counter()


def finish(name: str, started: float) -> float:
    elapsed = time.perf_counter() - started
    print(f"End Time: {now()}", flush=True)
    print(f"Elapsed Time: {elapsed:.3f}s", flush=True)
    print(f"Stage Finished: {name}", flush=True)
    return elapsed


def load_audit_rows() -> list[dict[str, str]]:
    if not AUDIT_CSV.is_file():
        raise FileNotFoundError(AUDIT_CSV)
    with AUDIT_CSV.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30 or len({(r["Model"], r["Dataset"]) for r in rows}) != 30:
        raise RuntimeError("Efficiency audit must contain exactly 30 unique cases")
    return rows


def required_rows() -> list[dict[str, str]]:
    rows = [r for r in load_audit_rows() if r["Audit Status"] == "REBENCHMARK_REQUIRED"]
    expected = {
        *((m, d) for m in CORE_MODELS for d in DATASETS),
        *(("TranAD", d) for d in DATASETS),
        *(("COUTA", d) for d in ("PSM", "SKAB", "MSL", "HAI")),
    }
    actual = {(r["Model"], r["Dataset"]) for r in rows}
    if actual != expected or len(rows) != 28:
        raise RuntimeError(f"Audit-driven case set changed: expected=28 actual={len(rows)}")
    return rows


def audit_row(model: str, dataset: str) -> dict[str, str]:
    matches = [r for r in load_audit_rows() if r["Model"] == model and r["Dataset"] == dataset]
    if len(matches) != 1:
        raise RuntimeError(f"Missing/duplicate audit row: {model}/{dataset}")
    return matches[0]


def check_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    return torch.device("cuda:0")


def cleanup_for_peak(device: torch.device) -> int:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gc.collect()
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def checkpoint_guard(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path),
            "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def assert_checkpoint_unchanged(path: Path, before: dict[str, Any]) -> None:
    after = checkpoint_guard(path)
    for key in ("sha256", "size", "mtime_ns"):
        if after[key] != before[key]:
            raise RuntimeError(f"Checkpoint modified during efficiency run: {path} ({key})")


def cuda_score_seconds(score_call: Callable[[torch.Tensor], torch.Tensor], gpu_batch: torch.Tensor) -> tuple[float, int]:
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    output = score_call(gpu_batch)
    end.record()
    end.synchronize()
    seconds = float(begin.elapsed_time(end)) / 1000.0
    output_count = int(output.shape[0]) if output.ndim else 1
    del output
    return seconds, output_count


def core_starts(length: int, window: int) -> list[int]:
    if length < window:
        raise RuntimeError(f"Test length {length} is smaller than native window {window}")
    starts = list(range(0, length - window + 1, window))
    final_start = length - window
    if starts[-1] != final_start:
        starts.append(final_start)
    covered = np.zeros(length, dtype=np.bool_)
    for start in starts:
        covered[start:start + window] = True
    if not bool(covered.all()):
        raise RuntimeError("Final-overlap policy failed to cover the complete test timeline")
    return starts


def core_cpu_batches(test: np.ndarray, starts: list[int], window: int) -> Iterator[tuple[torch.Tensor, int]]:
    for offset in range(0, len(starts), CORE_BATCH):
        current = starts[offset:offset + CORE_BATCH]
        array = np.stack([test[s:s + window] for s in current]).astype(np.float32, copy=False)
        yield torch.from_numpy(np.ascontiguousarray(array)), len(current)


def import_core(dataset: str):
    if dataset == "SKAB":
        protocol = importlib.import_module("scripts.benchmarks.benchmark_skab_efficiency")
    else:
        wrapper = importlib.import_module(f"scripts.benchmarks.benchmark_{dataset.lower()}_efficiency")
        wrapper.configure_protocol()
        protocol = wrapper.protocol
    return protocol


def run_core(model_name: str, dataset: str, row: dict[str, str], device: torch.device) -> dict[str, Any]:
    protocol = import_core(dataset)
    display, window, model, score_call, checkpoint = protocol.load_model(MODEL_KEYS[model_name], device)
    if display != model_name:
        raise RuntimeError(f"Model identity mismatch: {display} != {model_name}")
    model.eval()
    checkpoint = Path(checkpoint)
    guard = checkpoint_guard(checkpoint)
    test = protocol.load_scaled_test()
    if test.dtype != np.float32 or not np.isfinite(test).all():
        raise RuntimeError("Scaled test must be finite float32")
    starts = core_starts(len(test), window)
    expected_batches = math.ceil(len(starts) / CORE_BATCH)

    first_cpu, _ = next(core_cpu_batches(test, starts, window))
    warmup_batch = first_cpu.to(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            score_call(warmup_batch)
    torch.cuda.synchronize(device)
    del warmup_batch, first_cpu
    gc.collect(); torch.cuda.empty_cache()

    full_samples: list[float] = []
    full_started = stage(f"{dataset}/{model_name} Full Test x{FULL_REPEAT}")
    with torch.no_grad():
        for repeat in range(1, FULL_REPEAT + 1):
            gpu_seconds = 0.0
            batch_count = 0
            output_windows = 0
            wall_start = time.perf_counter()
            for cpu_batch, native_count in core_cpu_batches(test, starts, window):
                gpu_batch = cpu_batch.to(device, non_blocking=False)
                seconds, output_count = cuda_score_seconds(score_call, gpu_batch)
                if output_count != native_count:
                    raise RuntimeError(f"Window score batch mismatch: {output_count} != {native_count}")
                gpu_seconds += seconds
                output_windows += output_count
                batch_count += 1
                del gpu_batch, cpu_batch
            if batch_count != expected_batches or output_windows != len(starts):
                raise RuntimeError("Incomplete full-test execution")
            full_samples.append(gpu_seconds)
            elapsed = time.perf_counter() - full_started
            eta = elapsed / repeat * (FULL_REPEAT - repeat)
            print(f"Repeat {repeat}/{FULL_REPEAT} | GPU Time={gpu_seconds:.6f}s | "
                  f"Wall={time.perf_counter()-wall_start:.2f}s | ETA={eta:.1f}s", flush=True)
    finish(f"{dataset}/{model_name} Full Test", full_started)

    baseline = cleanup_for_peak(device)
    processed = 0
    memory_started = stage(f"{dataset}/{model_name} streaming GPU peak")
    with torch.no_grad():
        for index, (cpu_batch, native_count) in enumerate(core_cpu_batches(test, starts, window), 1):
            gpu_batch = cpu_batch.to(device, non_blocking=False)
            output = score_call(gpu_batch)
            if int(output.shape[0]) != native_count:
                raise RuntimeError("Memory pass score count mismatch")
            processed += native_count
            del output, gpu_batch, cpu_batch
            if index % 500 == 0:
                print(f"Memory pass batches={index}/{expected_batches}", flush=True)
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    finish(f"{dataset}/{model_name} streaming GPU peak", memory_started)
    if processed != len(starts):
        raise RuntimeError("Memory pass did not execute every native window")
    assert_checkpoint_unchanged(checkpoint, guard)
    mean = statistics.mean(full_samples)
    return {
        "model": model_name, "dataset": dataset, "metrics_rerun": row["Metrics To Rebenchmark"],
        "full_test_time_mean_s": mean, "full_test_time_std_s": statistics.pstdev(full_samples),
        "throughput_points_per_s": len(test) / mean,
        "gpu_peak_mib": peak / 2**20, "gpu_incremental_mib": (peak - baseline) / 2**20,
        "processed_points": int(len(test)), "inference_windows": len(starts),
        "full_test_batch_size": CORE_BATCH, "full_test_batch_count": expected_batches,
        "tail_policy": "final full window anchored at N-window; overlaps preceding window when needed",
        "checkpoint": guard,
    }


def import_tranad(dataset: str):
    if dataset == "PSM":
        benchmark = importlib.import_module("scripts.benchmarks.benchmark_psm_tranad_efficiency")
    else:
        wrapper = importlib.import_module(f"scripts.benchmarks.benchmark_{dataset.lower()}_tranad_efficiency")
        wrapper.configure()
        benchmark = wrapper.benchmark
    return benchmark


def run_tranad(dataset: str, row: dict[str, str], device: torch.device) -> dict[str, Any]:
    benchmark = import_tranad(dataset)
    config = benchmark.load_tranad_psm_config()
    model, load_audit = benchmark.load_model(config, device)
    model.eval()
    checkpoint = Path(benchmark.CHECKPOINT_PATH)
    guard = checkpoint_guard(checkpoint)
    adapter = benchmark.TranADPSMDataAdapter("score")
    loader = adapter.loader("test", batch_size=TRANAD_BATCH, shuffle=False)
    first_cpu = next(iter(loader))
    first_gpu = first_cpu.to(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            benchmark.formal_score_forward(model, first_gpu)
    torch.cuda.synchronize(device)
    del first_gpu, first_cpu, loader
    baseline = cleanup_for_peak(device)

    adapter = benchmark.TranADPSMDataAdapter("score")
    loader = adapter.loader("test", batch_size=TRANAD_BATCH, shuffle=False)
    processed = 0
    batches = 0
    started = stage(f"{dataset}/TranAD streaming GPU peak")
    with torch.no_grad():
        for batches, cpu_batch in enumerate(loader, 1):
            gpu_batch = cpu_batch.to(device, non_blocking=False)
            output = benchmark.formal_score_forward(model, gpu_batch)
            processed += int(output.numel())
            del output, gpu_batch, cpu_batch
            if batches % 500 == 0:
                print(f"Memory pass batches={batches} points={processed}", flush=True)
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    finish(f"{dataset}/TranAD streaming GPU peak", started)
    adapter.assert_label_free()
    accessed = adapter.audit().get("files_accessed", [])
    if any("label" in str(path).lower() for path in accessed):
        raise RuntimeError("TranAD efficiency accessed a label file")
    if processed != int(benchmark.EXPECTED_TEST_POINTS):
        raise RuntimeError(f"TranAD coverage mismatch: {processed} != {benchmark.EXPECTED_TEST_POINTS}")
    assert_checkpoint_unchanged(checkpoint, guard)
    return {
        "model": "TranAD", "dataset": dataset, "metrics_rerun": row["Metrics To Rebenchmark"],
        "gpu_peak_mib": peak / 2**20, "gpu_incremental_mib": (peak - baseline) / 2**20,
        "processed_points": processed, "inference_windows": processed,
        "full_test_batch_size": TRANAD_BATCH, "full_test_batch_count": batches,
        "checkpoint": guard, "load_audit": load_audit,
    }


def import_couta(dataset: str):
    if dataset == "PSM":
        return importlib.import_module("scripts.benchmarks.benchmark_psm_couta_efficiency")
    if dataset == "PUMP":
        return importlib.import_module("scripts.benchmarks.benchmark_pump_couta_efficiency")
    wrapper = importlib.import_module(f"scripts.benchmarks.benchmark_{dataset.lower()}_couta_efficiency")
    return wrapper.benchmark


def couta_cpu_windows(test_scaled: np.ndarray, seq_len: int) -> torch.Tensor:
    base = torch.from_numpy(np.ascontiguousarray(test_scaled))
    return base.unfold(0, seq_len, 1).permute(0, 2, 1)


def run_couta(dataset: str, row: dict[str, str], device: torch.device) -> dict[str, Any]:
    if dataset not in {"PSM", "SKAB", "MSL", "HAI"}:
        raise RuntimeError(f"Audit forbids COUTA rebenchmark on {dataset}")
    benchmark = import_couta(dataset)
    config = benchmark.load_couta_config()
    checkpoint = Path(benchmark.BUNDLE)
    guard = checkpoint_guard(checkpoint)
    adapter = benchmark.PSMCOUTADataAdapter("score")
    test = adapter.load_test()
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, scaler = benchmark.restore_from_bundle(bundle, str(device))
    net, center = model.net.eval(), model.c
    test_scaled = np.asarray(scaler.transform(test), dtype=np.float32)
    if not np.isfinite(test_scaled).all():
        raise RuntimeError("COUTA scaled test contains NaN/Inf")
    seq_len = int(config["model_config"]["seq_len"])
    windows = couta_cpu_windows(test_scaled, seq_len)
    expected_windows = len(test_scaled) - seq_len + 1
    if int(windows.shape[0]) != expected_windows:
        raise RuntimeError("COUTA window coverage mismatch")
    first_gpu = windows[:COUTA_BATCH].contiguous().to(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            benchmark.score_batch(net, center, first_gpu)
    torch.cuda.synchronize(device)
    del first_gpu
    baseline = cleanup_for_peak(device)

    processed_windows = 0
    batches = 0
    started = stage(f"{dataset}/COUTA streaming GPU peak")
    with torch.no_grad():
        for start in range(0, expected_windows, COUTA_BATCH):
            stop = min(start + COUTA_BATCH, expected_windows)
            gpu_batch = windows[start:stop].contiguous().to(device, non_blocking=False)
            output = benchmark.score_batch(net, center, gpu_batch)
            processed_windows += int(output.numel())
            batches += 1
            del output, gpu_batch
            if batches % 500 == 0:
                print(f"Memory pass batches={batches}/{math.ceil(expected_windows/COUTA_BATCH)}", flush=True)
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    finish(f"{dataset}/COUTA streaming GPU peak", started)
    accessed = adapter.audit().get("files_accessed", [])
    if any("label" in str(path).lower() for path in accessed):
        raise RuntimeError("COUTA efficiency accessed a label file")
    if processed_windows != expected_windows:
        raise RuntimeError("COUTA memory pass incomplete")
    assert_checkpoint_unchanged(checkpoint, guard)
    return {
        "model": "COUTA", "dataset": dataset, "metrics_rerun": row["Metrics To Rebenchmark"],
        "gpu_peak_mib": peak / 2**20, "gpu_incremental_mib": (peak - baseline) / 2**20,
        "processed_points": int(len(test_scaled)), "inference_windows": expected_windows,
        "full_test_batch_size": COUTA_BATCH, "full_test_batch_count": batches,
        "checkpoint": guard,
    }


def environment() -> dict[str, Any]:
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "device": "cuda:0", "dtype": "float32"}


def run_worker(model: str, dataset: str, overwrite: bool) -> None:
    row = audit_row(model, dataset)
    if row["Audit Status"] != "REBENCHMARK_REQUIRED":
        raise RuntimeError(f"Audit does not authorize rebenchmark: {model}/{dataset}")
    output = OUT / dataset / safe_name(model) / "rebenchmark.json"
    if output.exists() and not overwrite:
        print(f"[SKIP] complete artifact exists: {output}", flush=True)
        return
    set_seed()
    device = check_cuda()
    started = stage(f"worker {dataset}/{model}")
    if model in CORE_MODELS:
        metrics = run_core(model, dataset, row, device)
    elif model == "TranAD":
        metrics = run_tranad(dataset, row, device)
    elif model == "COUTA":
        metrics = run_couta(dataset, row, device)
    else:
        raise RuntimeError(model)
    payload = {
        "status": "COMPLETE", "audit_status": row["Audit Status"], "audit_metrics": row["Metrics To Rebenchmark"],
        "metrics": metrics, "protocol": {"seed": SEED, "warmup": WARMUP, "full_test_repeat": FULL_REPEAT,
        "model_eval": True, "torch_no_grad": True, "cuda_event": True,
        "training": False, "label_access": False, "evaluator_called": False,
        "threshold_computed": False, "prediction_generated": False,
        "data_loading_excluded": True, "scaler_excluded": True,
        "cpu_window_construction_excluded": True, "host_to_device_transfer_excluded": True,
        "all_windows_on_gpu": False, "checkpoint_modified": False,
        "legacy_results_modified": False, "environment": environment()},
        "completed_at": now(), "elapsed_wall_s": time.perf_counter() - started,
    }
    save_json(output, payload)
    finish(f"worker {dataset}/{model}", started)
    print(f"artifact={output}", flush=True)


def pick(mapping: dict[str, str], names: Iterable[str], required: bool = True) -> float | None:
    for name in names:
        value = mapping.get(name, "")
        if value not in (None, ""):
            return float(value)
    if required:
        raise KeyError(f"Missing columns: {list(names)}")
    return None


def legacy_row(audit: dict[str, str]) -> dict[str, str]:
    path = ROOT / audit["Result Path"]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    matches = [r for r in rows if r.get("Model") == audit["Model"]]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot identify legacy row {audit['Model']} in {path}")
    row = matches[0]
    # Some historical COUTA CSVs omitted audit-only fields that are present in
    # their sibling JSON. Merge only missing fields; never alter legacy files.
    json_path = path.with_suffix(".json")
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("Model") == audit["Model"]:
            for key, value in payload.items():
                if row.get(key, "") == "":
                    row[key] = str(value)
    return row


def merge_case(audit: dict[str, str]) -> dict[str, Any]:
    old = legacy_row(audit)
    model, dataset = audit["Model"], audit["Dataset"]
    corrected_path = OUT / dataset / safe_name(model) / "rebenchmark.json"
    corrected = json.loads(corrected_path.read_text(encoding="utf-8"))["metrics"] if corrected_path.exists() else None
    if audit["Audit Status"] == "REBENCHMARK_REQUIRED" and corrected is None:
        raise FileNotFoundError(f"Required corrected case is missing: {corrected_path}")
    params = pick(old, ("Trainable Parameters", "Parameters", "Total Parameters", "Params"))
    state = pick(old, ("State Dict (KiB)", "State Dict(KiB)"))
    b1 = pick(old, ("Latency_B1(ms)", "Latency B=1 Mean (ms)", "Latency B=1 (ms)", "Latency batch=1(ms)"))
    b128 = pick(old, ("Latency_B128(ms)", "Latency B=128 Mean (ms)", "Latency B=128 (ms)", "Latency batch=128(ms)"))
    full = pick(old, ("Full_Test_Time(s)", "Full Test Time Mean (s)", "Full Test Time (s)", "Full Test Time(s)"))
    full_std = pick(old, ("Full Test Time Std (s)",), required=False)
    throughput = pick(old, ("Throughput(points/s)", "Throughput (points/s)"))
    peak = pick(old, ("GPU_Peak(MiB)", "GPU Peak (MiB)", "GPU Peak Memory(MiB)"))
    incremental = pick(old, ("GPU_Incremental(MiB)", "GPU Incremental (MiB)", "GPU Incremental Memory(MiB)"), required=False)
    full_source = gpu_source = "legacy retained"
    if corrected and model in CORE_MODELS:
        full = corrected["full_test_time_mean_s"]
        full_std = corrected["full_test_time_std_s"]
        throughput = corrected["throughput_points_per_s"]
        full_source = "unified streaming rebenchmark"
    if corrected:
        peak = corrected["gpu_peak_mib"]
        incremental = corrected["gpu_incremental_mib"]
        gpu_source = "unified streaming rebenchmark"
    return {"Model": model, "Dataset": dataset, "Parameters": int(params), "State Dict (KiB)": state,
            "Latency B=1 (ms)": b1, "Latency B=128 (ms)": b128,
            "Full Test Time Mean (s)": full, "Full Test Time Std (s)": full_std,
            "Throughput (points/s)": throughput, "GPU Peak (MiB)": peak,
            "GPU Incremental (MiB)": incremental, "Full/Throughput Source": full_source,
            "GPU Memory Source": gpu_source, "Legacy Result": audit["Result Path"],
            "Corrected Result": str(corrected_path.relative_to(ROOT)).replace("\\", "/") if corrected else "",
            "Audit Status": audit["Audit Status"]}


def aggregate() -> None:
    rows = [merge_case(row) for row in load_audit_rows()]
    if len(rows) != 30:
        raise RuntimeError("Unified comparison must have 30 rows")
    OUT.mkdir(parents=True, exist_ok=True)
    # GPU Incremental is retained only in the audit artifact, not the paper table.
    fields = [name for name in rows[0] if name != "GPU Incremental (MiB)"]
    with (OUT / "unified_efficiency_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    paper_rows = [{name: row[name] for name in fields} for row in rows]
    save_json(OUT / "unified_efficiency_comparison.json", {"cases": paper_rows})
    save_json(OUT / "gpu_memory_audit.json", {"cases": [
        {"Model": row["Model"], "Dataset": row["Dataset"],
         "GPU Peak (MiB)": row["GPU Peak (MiB)"],
         "GPU Incremental (MiB)": row["GPU Incremental (MiB)"],
         "GPU Memory Source": row["GPU Memory Source"]}
        for row in rows
    ]})
    save_json(OUT / "protocol.json", {"status": "COMPLETE", "generated_at": now(),
        "audit_source": str(AUDIT_CSV.relative_to(ROOT)).replace("\\", "/"),
        "case_count": 30, "rebenchmarked_case_count": 28, "retained_couta_cases": ["PUMP", "SMD"],
        "full_test_repeat": FULL_REPEAT, "warmup": WARMUP,
        "training": False, "label_access": False, "evaluator_called": False,
        "threshold_computed": False, "prediction_generated": False,
        "legacy_results_modified": False, "checkpoint_modified": False,
        "all_windows_on_gpu": False, "environment": environment()})
    print(f"comparison_csv={OUT / 'unified_efficiency_comparison.csv'}", flush=True)


def orchestrate(overwrite: bool) -> None:
    rows = required_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    total = len(rows)
    started = stage(f"Unified efficiency rebenchmark: {total} audit-required cases")
    for index, row in enumerate(rows, 1):
        model, dataset = row["Model"], row["Dataset"]
        print(f"[{index}/{total}] {dataset}/{model} | metrics={row['Metrics To Rebenchmark']}", flush=True)
        command = [sys.executable, str(script), "--worker", "--model", model, "--dataset", dataset]
        if overwrite:
            command.append("--overwrite")
        subprocess.run(command, cwd=ROOT, check=True)
    aggregate()
    finish("Unified efficiency rebenchmark", started)


def validate_only() -> None:
    rows = required_rows()
    missing = []
    for row in rows:
        if not (ROOT / row["Result Path"]).is_file():
            missing.append(row["Result Path"])
    if missing:
        raise FileNotFoundError(f"Missing legacy result files: {missing}")
    print("Validation PASS", flush=True)
    print("required_cases=28", flush=True)
    print("core_full_time_cases=18", flush=True)
    print("tranad_gpu_only_cases=6", flush=True)
    print("couta_gpu_only_cases=4", flush=True)
    print("couta_retained_cases=2", flush=True)
    print("training=false labels=false evaluator=false thresholds=false", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--overwrite", action="store_true", help="overwrite only new unified artifacts")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_only()
    elif args.aggregate_only:
        aggregate()
    elif args.worker:
        if not args.model or not args.dataset:
            raise SystemExit("--worker requires --model and --dataset")
        run_worker(args.model, args.dataset, args.overwrite)
    else:
        orchestrate(args.overwrite)


if __name__ == "__main__":
    main()
