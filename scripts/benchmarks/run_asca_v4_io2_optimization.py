#!/usr/bin/env python3
"""Gated ASCA-AD V4-IO2 correctness, memory and speed audit.

This runner writes only below results/ASCA_INFERENCE_OPTIMIZATION_V2 and never
trains or saves a model checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asca_ad_optimized_v2 import ASCAInferenceOptimizedV2, ASCASolverInferenceOptimizedV2
from scripts.benchmarks.adapters.asca_optimized_v2_adapter import (
    ASCAOptimizedV2ScoreAdapter, ReusableCPUWindowBatcher,
)
from scripts.benchmarks.asca_optimization_common import (
    ANOMALY_RATIOS, BATCH_SIZE, DATASETS, EXPECTED_SHAPES, WINDOW,
    array_equivalence, build_optimized_from_old, checkpoint_compatibility,
    extract_checkpoint_state, formal_starts, full_coverage_starts,
    load_old_model, load_scaled_data, optimized_config_from_old, require_cuda,
    save_json, set_seed, sha256, state_dict_kib, tree_signature,
)
from scripts.benchmarks.run_asca_optimized_detection import evaluate_pair

OUT = ROOT / "results" / "ASCA_INFERENCE_OPTIMIZATION_V2"
SUMMARY = OUT / "summary"
OLD_OUT = ROOT / "results" / "ASCA_INFERENCE_OPTIMIZATION"
CHUNKS = (8, 4, 2, 1)
WARMUP = 30
LATENCY_REPEATS = 200
FULL_REPEATS = 20
ATOL = 1e-7
RTOL = 1e-6


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def protected_state() -> dict[str, Any]:
    files = [
        ROOT / "asca_ad" / "model.py",
        ROOT / "scripts" / "benchmarks" / "adapters" / "asca_adapter.py",
        ROOT / "scripts" / "benchmarks" / "adapters" / "asca_optimized_adapter.py",
    ]
    trees = [
        ROOT / "asca_ad_optimized",
        OLD_OUT,
        ROOT / "results" / "UNIFIED_RAM_BENCHMARK",
        ROOT / "results" / "MEMORY_ROOT_CAUSE_AUDIT",
    ]
    checkpoints = sorted((ROOT / "checkpoints").rglob("*.pt"))
    return {
        "files": {rel(p): sha256(p) if p.is_file() else None for p in files},
        "trees": {rel(p): tree_signature(p) for p in trees},
        "checkpoints": {rel(p): sha256(p) for p in checkpoints},
    }


def git_audit() -> dict[str, Any]:
    commands = {
        "status_short": ["git", "status", "--short"],
        "diff_stat": ["git", "diff", "--stat"],
        "original_v4_diff": ["git", "diff", "--", "asca_ad/model.py"],
        "v4_io_diff": ["git", "diff", "--", "asca_ad_optimized/"],
        "old_asca_adapter_diff": ["git", "diff", "--", "scripts/benchmarks/adapters/asca_adapter.py"],
        "v4_io_adapter_diff": ["git", "diff", "--", "scripts/benchmarks/adapters/asca_optimized_adapter.py"],
    }
    out = {}
    for name, command in commands.items():
        done = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        out[name] = {"returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr}
    return out


def build_v2(old_model, checkpoint: Path, device: torch.device, chunk_k: int):
    config = optimized_config_from_old(old_model)
    model = ASCAInferenceOptimizedV2(**config, chunk_k=chunk_k)
    state, _ = extract_checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"V4-IO2 strict load failed: {incompatible}")
    model = model.to(device).eval()
    solver = ASCASolverInferenceOptimizedV2(model, device, WINDOW, "instance", "official")
    return model, ASCAOptimizedV2ScoreAdapter(solver).eval().window_scores


def reset_shared_protocol_for(dataset: str) -> None:
    """Undo wrapper modules' mutation of the shared SKAB protocol module.

    Dataset-specific efficiency wrappers configure one shared module in-place.
    A six-dataset validation ending in SMD would otherwise make a subsequent
    SKAB load silently use the SMD checkpoint.  Reloading is needed only for
    SKAB; every other wrapper applies its own configuration before returning.
    """
    if dataset == "SKAB":
        module = importlib.import_module("scripts.benchmarks.benchmark_skab_efficiency")
        importlib.reload(module)


def compatibility_v2(dataset: str, device: torch.device, chunk_k: int = 2) -> dict:
    reset_shared_protocol_for(dataset)
    base = checkpoint_compatibility(dataset, device)
    _protocol, original, _score, checkpoint = load_old_model(dataset, device)
    v4io, _ = build_optimized_from_old(original, checkpoint, device)
    v2, _ = build_v2(original, checkpoint, device, chunk_k)
    states = [original.state_dict(), v4io.state_dict(), v2.state_dict()]
    keys = [list(s) for s in states]
    shapes = all(keys[0] == k for k in keys[1:]) and all(
        states[0][k].shape == states[1][k].shape == states[2][k].shape for k in keys[0]
    )
    params = [sum(p.numel() for p in m.parameters()) for m in (original, v4io, v2)]
    sizes = [state_dict_kib(m) for m in (original, v4io, v2)]
    persistent = set(v2.state_dict())
    caches = [name for name, _ in v2.named_buffers() if name.startswith("_")]
    result = {
        "dataset": dataset, "status": "PASS", "checkpoint": rel(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "original": base,
        "strict_load": {"original": True, "v4_io": True, "v4_io2": True},
        "missing_keys": {"original": [], "v4_io": [], "v4_io2": []},
        "unexpected_keys": {"original": [], "v4_io": [], "v4_io2": []},
        "state_dict_keys_identical": keys[0] == keys[1] == keys[2],
        "state_dict_shapes_identical": shapes, "parameter_counts": params,
        "state_dict_kib": sizes, "v4_io2_cache_names": caches,
        "v4_io2_caches_nonpersistent": all(name not in persistent for name in caches),
    }
    if not (shapes and params == [146, 146, 146] and result["v4_io2_caches_nonpersistent"]):
        result["status"] = "FAIL"
    del original, v4io, v2
    gc.collect(); torch.cuda.empty_cache()
    return result


def validate() -> dict:
    set_seed(); device = require_cuda()
    before = protected_state()
    compatibility = []
    for dataset in DATASETS:
        item = compatibility_v2(dataset, device)
        compatibility.append(item)
        print(f"[{dataset}] checkpoint={item['status']} params={item['parameter_counts']}", flush=True)
    after = protected_state()
    status = "PASS" if all(x["status"] == "PASS" for x in compatibility) and before == after else "FAIL"
    payload = {
        "status": status, "checkpoint_compatibility": compatibility,
        "protected_state_unchanged": before == after, "training": False,
        "feature_flags": {"use_total_only": True, "use_cached_indices": True,
            "use_chunked_gather": True, "chunk_k_candidates": list(CHUNKS),
            "reuse_cpu_batch_buffer": True, "use_pinned_staging_buffer": False},
    }
    save_json(OUT / "validation" / "validation.json", payload)
    save_json(OUT / "checkpoint_compatibility" / "checkpoint_compatibility_v2.json", payload)
    save_json(SUMMARY / "checkpoint_compatibility_v2.json", payload)
    print(f"validation={status}", flush=True)
    if status != "PASS":
        raise RuntimeError("V4-IO2 validation failed")
    return payload


def generate_energy(data: np.ndarray, split: str, score_call, device, path: Path) -> dict:
    starts = formal_starts(len(data), WINDOW, split)
    size = len(starts) * WINDOW
    path.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(size,))
    offset = 0; began = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (_starts, cpu) in enumerate(
            ReusableCPUWindowBatcher(data, starts, BATCH_SIZE, WINDOW, pinned=False), 1
        ):
            gpu = cpu.to(device, non_blocking=False)
            score = score_call(gpu)
            values = score.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            target[offset:offset + values.size] = values
            offset += values.size
            del score, gpu, values
            if batch_index % 500 == 0:
                print(f"[{split}] batches={batch_index} values={offset}/{size}", flush=True)
    target.flush()
    if offset != size:
        raise RuntimeError(f"incomplete energy {offset}/{size}")
    return {"windows": len(starts), "energy_length": size, "source_points": len(data),
        "stride": 1 if split == "train" else WINDOW,
        "tail_points_dropped": len(data) - (starts[-1] + WINDOW),
        "elapsed_seconds": time.perf_counter() - began}


def old_reference_paths(dataset: str) -> tuple[Path, Path]:
    base = OLD_OUT / dataset / "score_equivalence"
    train, test = base / "optimized_train_energy.npy", base / "optimized_scores.npy"
    if not train.is_file() or not test.is_file():
        raise FileNotFoundError(f"verified V4-IO score artifacts missing for {dataset}")
    return train, test


def score_and_detection(dataset: str, chunk_k: int, candidate_root: Path) -> tuple[dict, dict]:
    set_seed(); device = require_cuda()
    reset_shared_protocol_for(dataset)
    reference_train, reference_test = old_reference_paths(dataset)
    score_dir = candidate_root / "score_equivalence"
    score_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "SKAB":
        shutil.copyfile(reference_test, candidate_root / "reference_scores.npy")
    _protocol, old, _old_score, checkpoint = load_old_model(dataset, device)
    v2, score_call = build_v2(old, checkpoint, device, chunk_k)
    train, test, _ = load_scaled_data(dataset, include_label=False)
    candidate_train = score_dir / "candidate_train_energy.npy"
    candidate_test = candidate_root / "candidate_scores.npy"
    metadata = {
        "train": generate_energy(train, "train", score_call, device, candidate_train),
        "test": generate_energy(test, "test", score_call, device, candidate_test),
    }
    train_eq = array_equivalence(reference_train, candidate_train, score_dir / "train_score_diff.npy")
    diff_path = candidate_root / "score_diff.npy"
    test_eq = array_equivalence(reference_test, candidate_test, diff_path)
    score = {"status": "PASS" if train_eq["status"] == test_eq["status"] == "PASS" else "FAIL",
        "dataset": dataset, "chunk_k": chunk_k, "train_equivalence": train_eq,
        "test_equivalence": test_eq, "metadata": metadata, "label_access": False,
        "feature_flags": v2.feature_flags()}
    save_json(candidate_root / "score_equivalence.json", score)
    if score["status"] != "PASS":
        detection = {"status": "SKIPPED_SCORE_GATE_FAILED", "dataset": dataset, "chunk_k": chunk_k}
    else:
        detection = evaluate_pair(dataset, reference_train, candidate_train, reference_test,
                                  candidate_test, candidate_root / "detection_artifacts")
        detection["chunk_k"] = chunk_k
    save_json(candidate_root / "detection_equivalence.json", detection)
    coverage = metadata["test"]
    raw_points = EXPECTED_SHAPES[dataset][1][0]
    coverage.update({"dataset": dataset, "raw_test_points": raw_points,
        "evaluated_points": coverage["energy_length"],
        "omitted_tail_points": raw_points - coverage["energy_length"],
        "coverage_ratio": coverage["energy_length"] / raw_points,
        "tail_rule": "formal detection: stride=window, incomplete tail omitted"})
    save_json(candidate_root / "detection_coverage.json", coverage)
    del old, v2, train, test
    gc.collect(); torch.cuda.empty_cache()
    return score, detection


def cuda_latency(call: Callable[[], torch.Tensor]) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(WARMUP): call()
    torch.cuda.synchronize()
    values = []
    with torch.inference_mode():
        for _ in range(LATENCY_REPEATS):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); output = call(); end.record(); end.synchronize()
            values.append(float(start.elapsed_time(end))); del output
    return {"mean_ms": statistics.mean(values), "std_ms": statistics.pstdev(values)}


def efficiency_worker(dataset: str, implementation: str, chunk_k: int, destination: Path) -> None:
    set_seed(); device = require_cuda()
    protocol, old, _old_score, checkpoint = load_old_model(dataset, device)
    if implementation == "reference":
        model, score_call = build_optimized_from_old(old, checkpoint, device)
        display = "ASCA-AD V4-IO"
    else:
        model, score_call = build_v2(old, checkpoint, device, chunk_k)
        display = "ASCA-AD V4-IO2"
    del old; model.eval()
    test = protocol.load_scaled_test()
    starts = full_coverage_starts(len(test), WINDOW)
    batcher = ReusableCPUWindowBatcher(test, starts, BATCH_SIZE, WINDOW, pinned=False)
    _s, first = next(iter(batcher))
    b1 = first[:1].to(device); b128 = first[:min(BATCH_SIZE, len(first))].to(device)
    if len(b128) < BATCH_SIZE: b128 = b1.repeat(BATCH_SIZE, 1, 1)
    print(f"[{dataset}/{implementation}] latency start={now()}", flush=True)
    latency1 = cuda_latency(lambda: score_call(b1)); latency128 = cuda_latency(lambda: score_call(b128))
    del b1, b128, first; gc.collect(); torch.cuda.empty_cache()

    warm = next(iter(ReusableCPUWindowBatcher(test, starts, BATCH_SIZE, WINDOW)))[1].to(device)
    with torch.inference_mode():
        for _ in range(WARMUP): score_call(warm)
    torch.cuda.synchronize(); del warm; gc.collect(); torch.cuda.empty_cache()
    samples = []
    for repeat in range(1, FULL_REPEATS + 1):
        seconds = 0.0; windows = 0
        with torch.inference_mode():
            for _batch_starts, cpu in ReusableCPUWindowBatcher(test, starts, BATCH_SIZE, WINDOW):
                gpu = cpu.to(device, non_blocking=False)
                begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                begin.record(); output = score_call(gpu); end.record(); end.synchronize()
                seconds += float(begin.elapsed_time(end)) / 1000.0
                windows += int(output.shape[0]); del output, gpu
        if windows != len(starts): raise RuntimeError("full coverage window mismatch")
        samples.append(seconds)
        print(f"Repeat {repeat}/{FULL_REPEATS} | GPU Time={seconds:.6f}s", flush=True)

    torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
    baseline = int(torch.cuda.memory_allocated(device)); torch.cuda.reset_peak_memory_stats(device)
    windows = 0
    with torch.inference_mode():
        for _batch_starts, cpu in ReusableCPUWindowBatcher(test, starts, BATCH_SIZE, WINDOW):
            gpu = cpu.to(device); output = score_call(gpu); windows += int(output.shape[0])
            del output, gpu
    torch.cuda.synchronize(); peak = int(torch.cuda.max_memory_allocated(device))
    mean = statistics.mean(samples)
    payload = {"status": "COMPLETE", "dataset": dataset, "implementation": implementation,
        "model": display, "chunk_k": chunk_k if implementation == "candidate" else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "state_dict_kib": state_dict_kib(model), "latency_b1": latency1,
        "latency_b128": latency128, "full_test_time_mean_s": mean,
        "full_test_time_std_s": statistics.pstdev(samples),
        "throughput_points_per_s": len(test) / mean, "gpu_baseline_mib": baseline / 2**20,
        "gpu_peak_mib": peak / 2**20, "gpu_incremental_mib": (peak - baseline) / 2**20,
        "processed_points": len(test), "inference_windows": windows, "coverage_ratio": 1.0,
        "tail_policy": "efficiency: final full window anchored at N-window",
        "warmup": WARMUP, "latency_repeats": LATENCY_REPEATS,
        "full_test_repeats": FULL_REPEATS, "dtype": "float32", "device": str(device),
        "gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "checkpoint": rel(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "training": False, "label_access": False,
        "threshold_computed": False, "evaluator_called": False, "all_windows_on_gpu": False,
        "data_loading_excluded": True, "host_to_device_transfer_excluded": True,
        "reuse_cpu_batch_buffer": True, "use_pinned_staging_buffer": False,
        "peak_process_rss_mib": None, "peak_uss_mib": None}
    save_json(destination, payload)


def monitor_efficiency(dataset: str, implementation: str, chunk_k: int, destination: Path) -> dict:
    import psutil
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-efficiency",
               "--dataset", dataset, "--implementation", implementation,
               "--chunk-k", str(chunk_k), "--output", str(destination)]
    process = subprocess.Popen(command, cwd=ROOT)
    watched = psutil.Process(process.pid); peak_rss = peak_uss = 0
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, watched.memory_info().rss)
            peak_uss = max(peak_uss, watched.memory_full_info().uss)
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        time.sleep(0.01)
    if process.wait(): raise subprocess.CalledProcessError(process.returncode, command)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["peak_process_rss_mib"] = peak_rss / 2**20
    payload["peak_uss_mib"] = peak_uss / 2**20
    save_json(destination, payload)
    return payload


def aggregate_repeats(items: list[dict]) -> dict:
    result = dict(items[0]); result["independent_process_repeats"] = len(items)
    for key in ("gpu_peak_mib", "gpu_incremental_mib", "peak_process_rss_mib", "peak_uss_mib"):
        values = [float(x[key]) for x in items]
        result[key + "_statistics"] = {"mean": statistics.mean(values),
            "std": statistics.pstdev(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}
        result[key] = statistics.mean(values)
    return result


def run_efficiency(dataset: str, implementation: str, chunk_k: int, destination: Path) -> dict:
    repeats = 3 if dataset == "HAI" else 1
    items = []
    for index in range(repeats):
        path = destination if repeats == 1 else destination.with_name(f"{destination.stem}_repeat_{index+1}.json")
        items.append(monitor_efficiency(dataset, implementation, chunk_k, path))
    result = aggregate_repeats(items)
    save_json(destination, result)
    return result


def chunk_search(resume: bool = False, only_chunk: int | None = None) -> dict:
    rows = []
    for chunk in CHUNKS:
        if only_chunk is not None and chunk != only_chunk: continue
        root = OUT / "chunk_search" / f"chunk_{chunk}"; root.mkdir(parents=True, exist_ok=True)
        save_json(root / "configuration.json", {"dataset": "SKAB", "chunk_k": chunk,
            "feature_flags": {"use_total_only": True, "use_cached_indices": True,
            "use_chunked_gather": True, "reuse_cpu_batch_buffer": True,
            "use_pinned_staging_buffer": False}})
        set_seed(); device = require_cuda(); comp = compatibility_v2("SKAB", device, chunk)
        save_json(root / "checkpoint_compatibility.json", comp)
        result_file = root / "detection_equivalence.json"
        if resume and result_file.is_file() and (root / "score_equivalence.json").is_file():
            score = json.loads((root / "score_equivalence.json").read_text(encoding="utf-8"))
            detection = json.loads(result_file.read_text(encoding="utf-8"))
        else:
            score, detection = score_and_detection("SKAB", chunk, root)
        if score["status"] == "PASS":
            reference = run_efficiency("SKAB", "reference", chunk, root / "efficiency_reference.json")
            candidate = run_efficiency("SKAB", "candidate", chunk, root / "efficiency_candidate.json")
            save_json(root / "memory_reference.json", reference)
            save_json(root / "memory_candidate.json", candidate)
        else:
            reference = candidate = {}
        full_slow = ((candidate.get("full_test_time_mean_s", float("inf")) /
                     reference.get("full_test_time_mean_s", 1.0)) - 1) * 100
        throughput_change = ((candidate.get("throughput_points_per_s", 0.0) /
                              reference.get("throughput_points_per_s", 1.0)) - 1) * 100
        peak_reduction = reference.get("gpu_peak_mib", 0.0) - candidate.get("gpu_peak_mib", 0.0)
        eligible = bool(comp["status"] == "PASS" and score["status"] == "PASS" and
            detection.get("status") == "PASS" and full_slow <= 10.0 and
            throughput_change >= -10.0 and peak_reduction > 0.0)
        eq = score.get("test_equivalence", {})
        rows.append({"Chunk K": chunk, "Checkpoint Compatible": comp["status"] == "PASS",
            "Score Max Abs Error": eq.get("max_absolute_error"),
            "Score Allclose": eq.get("allclose_atol_1e-7_rtol_1e-6"),
            "Threshold Difference": detection.get("threshold_absolute_difference"),
            "RAW Identical": detection.get("raw_prediction_equal"),
            "PA Identical": detection.get("pa_prediction_equal"),
            "Detection Identical": detection.get("detection_metrics_equal"),
            "B1 Latency ms": candidate.get("latency_b1", {}).get("mean_ms"),
            "B128 Latency ms": candidate.get("latency_b128", {}).get("mean_ms"),
            "Full Test Time s": candidate.get("full_test_time_mean_s"),
            "Full Test Slowdown %": full_slow, "Throughput points/s": candidate.get("throughput_points_per_s"),
            "Throughput Change %": throughput_change, "GPU Peak MiB": candidate.get("gpu_peak_mib"),
            "GPU Peak Reduction MiB": peak_reduction,
            "GPU Peak Reduction %": peak_reduction / reference.get("gpu_peak_mib", 1) * 100,
            "GPU Incremental MiB": candidate.get("gpu_incremental_mib"),
            "Peak RSS MiB": candidate.get("peak_process_rss_mib"),
            "Peak USS MiB": candidate.get("peak_uss_mib"), "Eligible": eligible,
            "Selected": False, "Status": "ELIGIBLE" if eligible else "REJECTED", "Notes": ""})
    eligible_rows = [r for r in rows if r["Eligible"]]
    selected = min(eligible_rows, key=lambda x: x["GPU Peak MiB"]) if eligible_rows else None
    if selected: selected["Selected"] = True; selected["Status"] = "SELECTED"
    fields = list(rows[0]) if rows else []
    write_csv(SUMMARY / "chunk_search_table.csv", rows, fields)
    write_csv(SUMMARY / "score_equivalence_table.csv", rows,
              ["Chunk K", "Score Max Abs Error", "Score Allclose", "Status"])
    write_csv(SUMMARY / "detection_equivalence_table.csv", rows,
              ["Chunk K", "Threshold Difference", "RAW Identical", "PA Identical", "Detection Identical", "Status"])
    selection = {"status": "SELECTED" if selected else "COMPLETE_NO_ACCEPTABLE_CHUNK",
        "best_chunk": selected["Chunk K"] if selected else None,
        "selection_uses_labels": False, "selection_uses_threshold": False,
        "rule": "lowest GPU Peak among correctness-passing candidates within 10% speed/throughput"}
    save_json(SUMMARY / "selected_chunk.json", selection)
    return selection


def six_dataset(best_chunk: int, resume: bool = False, only_dataset: str | None = None,
                reference_only: bool = False, candidate_only: bool = False) -> None:
    efficiency_rows = []; coverage_rows = []; failed = []
    datasets = (only_dataset,) if only_dataset else DATASETS
    for dataset in datasets:
        root = OUT / dataset
        try:
            if not candidate_only:
                ref = run_efficiency(dataset, "reference", best_chunk, root / "efficiency_reference.json")
            else:
                ref = json.loads((root / "efficiency_reference.json").read_text(encoding="utf-8"))
            if not reference_only:
                score, detection = score_and_detection(dataset, best_chunk, root)
                if score["status"] != "PASS" or detection["status"] != "PASS":
                    raise RuntimeError("correctness equivalence failed")
                cand = run_efficiency(dataset, "candidate", best_chunk, root / "efficiency_candidate.json")
            else:
                cand = json.loads((root / "efficiency_candidate.json").read_text(encoding="utf-8"))
            coverage = json.loads((root / "detection_coverage.json").read_text(encoding="utf-8"))
            coverage_rows.append(coverage)
            metrics = {
                "Parameters": (ref["parameters"], cand["parameters"]),
                "State Dict KiB": (ref["state_dict_kib"], cand["state_dict_kib"]),
                "B=1 Latency ms": (ref["latency_b1"]["mean_ms"], cand["latency_b1"]["mean_ms"]),
                "B=128 Latency ms": (ref["latency_b128"]["mean_ms"], cand["latency_b128"]["mean_ms"]),
                "Full Test Time s": (ref["full_test_time_mean_s"], cand["full_test_time_mean_s"]),
                "Throughput points/s": (ref["throughput_points_per_s"], cand["throughput_points_per_s"]),
                "GPU Peak MiB": (ref["gpu_peak_mib"], cand["gpu_peak_mib"]),
                "GPU Incremental MiB": (ref["gpu_incremental_mib"], cand["gpu_incremental_mib"]),
                "Peak RSS MiB": (ref["peak_process_rss_mib"], cand["peak_process_rss_mib"]),
                "Peak USS MiB": (ref["peak_uss_mib"], cand["peak_uss_mib"]),
            }
            for metric, (a, b) in metrics.items():
                higher = metric.startswith("Throughput")
                efficiency_rows.append({"Dataset": dataset, "Metric": metric, "V4-IO": a,
                    "V4-IO2": b, "Absolute Difference": b-a,
                    "Percentage Change": ((b/a)-1)*100 if a else 0,
                    "Winner": "V4-IO2" if (b > a if higher else b < a) else ("Tie" if b == a else "V4-IO")})
        except Exception as exc:
            failed.append({"Dataset": dataset, "Stage": "six_dataset", "Error": repr(exc)})
            print(f"[{dataset}] FAILED: {exc}", flush=True)
    if efficiency_rows:
        write_csv(SUMMARY / "efficiency_v4io_vs_v4io2.csv", efficiency_rows, list(efficiency_rows[0]))
        gpu = [r for r in efficiency_rows if r["Metric"].startswith("GPU")]
        process = [r for r in efficiency_rows if "RSS" in r["Metric"] or "USS" in r["Metric"]]
        write_csv(SUMMARY / "gpu_memory_v4io_vs_v4io2.csv", gpu, list(efficiency_rows[0]))
        write_csv(SUMMARY / "process_memory_v4io_vs_v4io2.csv", process, list(efficiency_rows[0]))
    if coverage_rows: write_csv(SUMMARY / "detection_coverage_audit.csv", coverage_rows, list(coverage_rows[0]))
    write_csv(SUMMARY / "failed_cases.csv", failed, ["Dataset", "Stage", "Error"])


def finalize(selection: dict, started: float, pre_state: dict) -> None:
    post_state = protected_state(); unchanged = pre_state == post_state
    audit = {"protected_state_unchanged": unchanged, "before": pre_state,
        "after": post_state, "git": git_audit(), "original_v4_modified": not unchanged,
        "v4_io_modified": not unchanged, "old_adapters_modified": not unchanged,
        "checkpoint_modified": not unchanged, "detection_results_modified": not unchanged,
        "historical_efficiency_modified": not unchanged, "old_optimization_results_modified": not unchanged}
    save_json(SUMMARY / "source_integrity_audit.json", audit)
    status = selection["status"] if unchanged else "FAILED_SOURCE_INTEGRITY"
    save_json(SUMMARY / "optimization_v2_status.json", {"status": status,
        "best_chunk": selection.get("best_chunk"), "recommend_v4_io2": status == "SELECTED",
        "retain_v4_io": True, "elapsed_seconds": time.perf_counter()-started})
    protocol = {"model": "ASCA-AD V4-IO2", "training": False, "datasets": list(DATASETS),
        "chunks": list(CHUNKS), "window": WINDOW, "batch_size": BATCH_SIZE,
        "dtype": "float32", "score_mode": "total", "score_atol": ATOL, "score_rtol": RTOL,
        "warmup": WARMUP, "latency_repeats": LATENCY_REPEATS,
        "full_test_repeats": FULL_REPEATS, "labels_used_for_selection": False,
        "threshold_used_for_selection": False, "all_windows_on_gpu": False}
    save_json(SUMMARY / "protocol_v4_io2.json", protocol)
    text = ["# ASCA-AD V4-IO2 execution summary", "", f"Status: {status}",
        f"Selected chunk: {selection.get('best_chunk')}", f"Protected sources unchanged: {unchanged}",
        "", "V4-IO remains the formal implementation unless status is SELECTED."]
    (SUMMARY / "execution_summary.md").write_text("\n".join(text)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-chunk-search", action="store_true")
    mode.add_argument("--run-all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--chunk-k", type=int, choices=CHUNKS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--worker-efficiency", action="store_true")
    parser.add_argument("--implementation", choices=("reference", "candidate"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker_efficiency:
        efficiency_worker(args.dataset, args.implementation, args.chunk_k, args.output); return
    if args.reference_only and args.candidate_only:
        raise SystemExit("reference-only and candidate-only are mutually exclusive")
    OUT.mkdir(parents=True, exist_ok=True); SUMMARY.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); pre_path = OUT / "validation" / "protected_state_before.json"
    if pre_path.is_file(): pre_state = json.loads(pre_path.read_text(encoding="utf-8"))
    else: pre_state = protected_state(); save_json(pre_path, pre_state)
    validation = validate()
    if args.validate_only: return
    selection = chunk_search(args.resume, args.chunk_k)
    if args.run_chunk_search: finalize(selection, started, pre_state); return
    if selection.get("best_chunk") is not None:
        six_dataset(selection["best_chunk"], args.resume, args.dataset,
                    args.reference_only, args.candidate_only)
    finalize(selection, started, pre_state)
    print(f"status={selection['status']} best_chunk={selection.get('best_chunk')}", flush=True)


if __name__ == "__main__":
    main()
