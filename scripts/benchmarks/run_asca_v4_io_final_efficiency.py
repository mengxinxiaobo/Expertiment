#!/usr/bin/env python3
"""Final paper efficiency benchmark for verified ASCA-AD V4-IO only.

No labels, thresholds, predictions, evaluators, training, GPU-memory probes or
process-memory probes are present in this program.  Every dataset worker runs in
an independent process and writes only below ASCA_V4_IO_FINAL_EFFICIENCY.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
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

from scripts.benchmarks.asca_optimization_common import (
    BATCH_SIZE, DATASETS, WINDOW, build_optimized_from_old, cpu_batches,
    full_coverage_starts, load_old_model, require_cuda, save_json, set_seed,
    sha256, tree_signature,
)

OUT = ROOT / "results" / "ASCA_V4_IO_FINAL_EFFICIENCY"
SUMMARY = OUT / "summary"
OLD_OPT = ROOT / "results" / "ASCA_INFERENCE_OPTIMIZATION"
WARMUP = 30
LATENCY_REPEATS = 200
FULL_REPEATS = 20


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def serialized_state(model: torch.nn.Module) -> tuple[float, int, dict[str, list[int]]]:
    state = model.state_dict()
    stream = io.BytesIO(); torch.save(state, stream)
    return len(stream.getvalue()) / 1024.0, len(state), {
        name: list(tensor.shape) for name, tensor in state.items()
    }


def cuda_latency(call: Callable[[], torch.Tensor]) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(WARMUP):
            output = call(); del output
    torch.cuda.synchronize()
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(LATENCY_REPEATS):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(); output = call(); end.record(); end.synchronize()
            samples.append(float(begin.elapsed_time(end))); del output
    return {"mean_ms": statistics.mean(samples), "std_ms": statistics.pstdev(samples)}


def timed_forward(score_call, gpu_batch: torch.Tensor) -> tuple[float, int]:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record(); output = score_call(gpu_batch); end.record(); end.synchronize()
    seconds = float(begin.elapsed_time(end)) / 1000.0
    windows = int(output.shape[0]); del output
    return seconds, windows


def old_efficiency(dataset: str) -> dict[str, Any] | None:
    path = OLD_OPT / dataset / "efficiency" / "efficiency_old.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"source": rel(path), "b1_latency_ms": payload["latency_b1"]["mean_ms"],
        "b128_latency_ms": payload["latency_b128"]["mean_ms"],
        "full_test_time_s": payload["full_test_time_mean_s"],
        "throughput_points_per_s": payload["throughput_points_per_s"]}


def improvements(old: dict[str, Any] | None, current: dict[str, float]) -> dict[str, Any]:
    if old is None:
        return {"available": False}
    return {"available": True, "source": old["source"],
        "b1_latency_reduction_percent": (1-current["b1"]/old["b1_latency_ms"])*100,
        "b128_latency_reduction_percent": (1-current["b128"]/old["b128_latency_ms"])*100,
        "full_test_time_reduction_percent": (1-current["full"]/old["full_test_time_s"])*100,
        "throughput_improvement_percent": (current["throughput"]/old["throughput_points_per_s"]-1)*100}


def worker(dataset: str) -> None:
    began = time.perf_counter(); set_seed(); device = require_cuda()
    protocol, original, _original_score, checkpoint = load_old_model(dataset, device)
    checkpoint_before = sha256(checkpoint)
    model, score_call = build_optimized_from_old(original, checkpoint, device)
    del original
    model.eval()
    parameters = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    state_kib, state_key_count, state_shapes = serialized_state(model)
    if parameters != trainable or parameters != 146:
        raise RuntimeError(f"unexpected V4-IO parameter count: {parameters}/{trainable}")

    # This formal loader performs the already-established train-fit-only scaler
    # transformation and never accesses a label file.
    test = protocol.load_scaled_test()
    if test.dtype != np.float32 or test.ndim != 2 or not np.isfinite(test).all():
        raise RuntimeError("formal scaled test must be finite float32 [N,C]")
    raw_points = int(len(test))
    starts = full_coverage_starts(raw_points, WINDOW)
    coverage = np.zeros(raw_points, dtype=np.bool_)
    for start in starts: coverage[start:start+WINDOW] = True
    processed_points = int(np.count_nonzero(coverage))
    if processed_points != raw_points:
        raise RuntimeError("efficiency inference windows do not cover the full test timeline")

    first_cpu = next(cpu_batches(test, starts, BATCH_SIZE))
    batch1 = first_cpu[:1].to(device, non_blocking=False)
    batch128 = batch1.repeat(BATCH_SIZE, 1, 1)
    print(f"[{dataset}] Stage 1 latency start={now()}", flush=True)
    latency1 = cuda_latency(lambda: score_call(batch1))
    latency128 = cuda_latency(lambda: score_call(batch128))
    del batch1, batch128, first_cpu
    gc.collect()

    warm_cpu = next(cpu_batches(test, starts, BATCH_SIZE))
    warm_gpu = warm_cpu.to(device, non_blocking=False)
    with torch.inference_mode():
        for _ in range(WARMUP):
            output = score_call(warm_gpu); del output
    torch.cuda.synchronize(); del warm_cpu, warm_gpu

    samples: list[float] = []
    print(f"[{dataset}] Stage 2 full-test x{FULL_REPEATS} start={now()}", flush=True)
    with torch.inference_mode():
        for repeat in range(1, FULL_REPEATS + 1):
            gpu_seconds = 0.0; output_windows = 0
            for cpu_batch in cpu_batches(test, starts, BATCH_SIZE):
                gpu_batch = cpu_batch.to(device, non_blocking=False)
                seconds, count = timed_forward(score_call, gpu_batch)
                gpu_seconds += seconds; output_windows += count
                del gpu_batch, cpu_batch
            if output_windows != len(starts):
                raise RuntimeError("full-test window count mismatch")
            samples.append(gpu_seconds)
            eta = statistics.mean(samples) * (FULL_REPEATS-repeat)
            print(f"[{dataset}] Repeat {repeat}/{FULL_REPEATS} "
                  f"GPU={gpu_seconds:.6f}s ETA={eta:.1f}s", flush=True)

    full_mean = statistics.mean(samples); full_std = statistics.pstdev(samples)
    throughput = processed_points / full_mean
    current = {"b1": latency1["mean_ms"], "b128": latency128["mean_ms"],
        "full": full_mean, "throughput": throughput}
    comparison = improvements(old_efficiency(dataset), current)
    checkpoint_after = sha256(checkpoint)
    payload = {"status": "COMPLETE", "dataset": dataset, "model": "ASCA-AD V4-IO",
        "parameters": parameters, "trainable_parameters": trainable,
        "state_dict_kib": state_kib, "state_dict_key_count": state_key_count,
        "state_dict_tensor_shapes": state_shapes, "latency_b1": latency1,
        "latency_b128": latency128, "full_test_time_mean_s": full_mean,
        "full_test_time_std_s": full_std, "throughput_points_per_s": throughput,
        "processed_points": processed_points, "raw_points": raw_points,
        "coverage_ratio": processed_points/raw_points, "inference_windows": len(starts),
        "tail_rule": "final full window anchored at N-window; complete unique-point coverage",
        "window": WINDOW, "batch_size": BATCH_SIZE, "dtype": "float32",
        "warmup": WARMUP, "latency_repeats": LATENCY_REPEATS,
        "full_test_repeats": FULL_REPEATS, "device": str(device),
        "gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "checkpoint": rel(checkpoint),
        "checkpoint_sha256": checkpoint_before,
        "checkpoint_modified": checkpoint_before != checkpoint_after,
        "old_asca_comparison": comparison, "training": False, "label_access": False,
        "evaluator_called": False, "threshold_computed": False,
        "prediction_generated": False, "detection_metrics_computed": False,
        "gpu_memory_measured": False, "process_memory_measured": False,
        "data_loading_excluded": True, "scaler_excluded": True,
        "host_to_device_transfer_excluded": True,
        "timed_scope": "GPU-resident batch -> V4-IO forward -> total score",
        "elapsed_wall_seconds": time.perf_counter()-began}
    if payload["checkpoint_modified"]: payload["status"] = "FAILED"
    save_json(OUT/dataset/"efficiency.json", payload)
    protocol_payload = {key: payload[key] for key in (
        "dataset", "model", "window", "batch_size", "dtype", "warmup",
        "latency_repeats", "full_test_repeats", "raw_points", "processed_points",
        "coverage_ratio", "inference_windows", "tail_rule", "device", "gpu",
        "torch_version", "cuda_version", "checkpoint", "checkpoint_sha256",
        "training", "label_access", "evaluator_called", "threshold_computed",
        "prediction_generated", "detection_metrics_computed", "gpu_memory_measured",
        "process_memory_measured", "data_loading_excluded", "scaler_excluded",
        "host_to_device_transfer_excluded", "timed_scope")}
    save_json(OUT/dataset/"protocol.json", protocol_payload)
    if payload["status"] != "COMPLETE": raise RuntimeError("worker audit failed")
    print(f"[{dataset}] COMPLETE b1={latency1['mean_ms']:.6f}ms "
          f"b128={latency128['mean_ms']:.6f}ms full={full_mean:.6f}±{full_std:.6f}s "
          f"throughput={throughput:.2f} points/s", flush=True)


def protected_snapshot() -> dict[str, Any]:
    files = [ROOT/"asca_ad"/"model.py",
             ROOT/"scripts"/"benchmarks"/"adapters"/"asca_optimized_adapter.py"]
    result_dirs = []
    results = ROOT/"results"
    if results.is_dir():
        result_dirs = [p for p in results.iterdir() if p.is_dir() and p.resolve() != OUT.resolve()]
    return {"files": {rel(p): sha256(p) if p.is_file() else None for p in files},
        "v4_io_source": tree_signature(ROOT/"asca_ad_optimized"),
        "checkpoints": tree_signature(ROOT/"checkpoints"),
        "existing_results": {rel(p): tree_signature(p) for p in sorted(result_dirs)},
        "git_status": subprocess.run(["git", "status", "--short"], cwd=ROOT,
            text=True, capture_output=True, check=False).stdout}


def run_worker(dataset: str) -> None:
    log = OUT/dataset/"run.log"; log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--dataset", dataset]
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); stream.write(line); stream.flush()
    code = process.wait()
    if code: raise subprocess.CalledProcessError(code, command)


def summarize(pre: dict, selected: tuple[str, ...]) -> None:
    rows = []; improvements_rows = []; failed = []
    for dataset in DATASETS:
        path = OUT/dataset/"efficiency.json"
        if not path.is_file():
            if dataset in selected: failed.append({"Dataset": dataset, "Reason": "missing efficiency.json"})
            continue
        p = json.loads(path.read_text(encoding="utf-8"))
        if p.get("status") != "COMPLETE": failed.append({"Dataset": dataset, "Reason": p.get("status")})
        rows.append({"Dataset": dataset, "Model": p["model"], "Parameters": p["parameters"],
            "State Dict KiB": p["state_dict_kib"], "B1 Latency ms": p["latency_b1"]["mean_ms"],
            "B128 Latency ms": p["latency_b128"]["mean_ms"],
            "Full Test Time s": p["full_test_time_mean_s"],
            "Full Test Time Std s": p["full_test_time_std_s"],
            "Throughput points/s": p["throughput_points_per_s"],
            "Processed Points": p["processed_points"], "Raw Points": p["raw_points"],
            "Coverage Ratio": p["coverage_ratio"]})
        imp = p["old_asca_comparison"]
        if imp.get("available"):
            improvements_rows.append({"Dataset": dataset,
                "B1 Latency Reduction %": imp["b1_latency_reduction_percent"],
                "B128 Latency Reduction %": imp["b128_latency_reduction_percent"],
                "Full Test Time Reduction %": imp["full_test_time_reduction_percent"],
                "Throughput Improvement %": imp["throughput_improvement_percent"],
                "Old Result Source": imp["source"]})
    SUMMARY.mkdir(parents=True, exist_ok=True)
    fields = ["Dataset", "Model", "Parameters", "State Dict KiB", "B1 Latency ms",
        "B128 Latency ms", "Full Test Time s", "Full Test Time Std s",
        "Throughput points/s", "Processed Points", "Raw Points", "Coverage Ratio"]
    with (SUMMARY/"final_asca_v4_io_efficiency.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    if improvements_rows:
        with (SUMMARY/"old_asca_improvement.csv").open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(improvements_rows[0])); w.writeheader(); w.writerows(improvements_rows)
    post = protected_snapshot(); unchanged = pre == post
    audit = {"existing_model_modified": not unchanged, "existing_checkpoint_modified": not unchanged,
        "existing_detection_modified": not unchanged, "old_results_modified": not unchanged,
        "protected_snapshot_unchanged": unchanged, "before": pre, "after": post}
    save_json(SUMMARY/"integrity_audit.json", audit)
    completed_names = {r["Dataset"] for r in rows}
    selected_complete = all(name in completed_names for name in selected) and not failed and unchanged
    all_complete = len(rows) == len(DATASETS) and selected_complete
    status = "COMPLETE" if all_complete else ("PARTIAL_COMPLETE" if selected_complete else "INCOMPLETE")
    save_json(SUMMARY/"status.json", {"status": status,
        "completed_datasets": [r["Dataset"] for r in rows], "failed": failed,
        "training": False, "label_access": False, "evaluator_called": False})
    lines = ["# ASCA-AD V4-IO Final Efficiency", "",
        f"Status: {status}", "",
        f"Completed datasets: {', '.join(r['Dataset'] for r in rows)}", "",
        "- Training: false", "- Label access: false", "- Evaluator called: false",
        "- Threshold/prediction/Detection: not computed", "- GPU/RAM memory: not measured",
        f"- Parameters: {rows[0]['Parameters'] if rows else 'N/A'}",
        f"- State Dict: {rows[0]['State Dict KiB'] if rows else 'N/A'} KiB", "",
        "| Dataset | B=1 ms | B=128 ms | Full Test mean±std s | Throughput points/s | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Dataset']} | {row['B1 Latency ms']:.6f} | {row['B128 Latency ms']:.6f} | "
            f"{row['Full Test Time s']:.6f}±{row['Full Test Time Std s']:.6f} | "
            f"{row['Throughput points/s']:.2f} | {row['Coverage Ratio']:.6f} |"
        )
    lines.extend(["", "## Improvement over remeasured Original V4", "",
        "| Dataset | B=1 reduction | B=128 reduction | Full-time reduction | Throughput improvement |",
        "|---|---:|---:|---:|---:|"])
    for item in improvements_rows:
        lines.append(
            f"| {item['Dataset']} | {item['B1 Latency Reduction %']:.2f}% | "
            f"{item['B128 Latency Reduction %']:.2f}% | "
            f"{item['Full Test Time Reduction %']:.2f}% | "
            f"{item['Throughput Improvement %']:.2f}% |"
        )
    lines.extend(["",
        "The measured V4-IO implementation is mathematically equivalent to V4 and is recommended",
        "for the paper efficiency table when this audit status is COMPLETE.", "",
        "Detailed timings are in final_asca_v4_io_efficiency.csv; improvements relative to the",
        "independently remeasured Original V4 are in old_asca_improvement.csv."])
    (SUMMARY/"execution_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if not selected_complete:
        raise RuntimeError(f"final efficiency audit incomplete: {failed}, integrity={unchanged}")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args=parser.parse_args()
    if args.worker:
        if not args.dataset: raise SystemExit("--worker requires --dataset")
        worker(args.dataset); return
    OUT.mkdir(parents=True, exist_ok=True); SUMMARY.mkdir(parents=True, exist_ok=True)
    pre_path=SUMMARY/"protected_snapshot_before.json"
    if pre_path.is_file(): pre=json.loads(pre_path.read_text(encoding="utf-8"))
    else: pre=protected_snapshot(); save_json(pre_path, pre)
    selected=(args.dataset,) if args.dataset else DATASETS
    for dataset in selected:
        result=OUT/dataset/"efficiency.json"
        if args.resume and result.is_file():
            previous=json.loads(result.read_text(encoding="utf-8"))
            if previous.get("status")=="COMPLETE":
                print(f"[{dataset}] COMPLETE artifact exists; skip", flush=True); continue
        run_worker(dataset)
    summarize(pre, selected)
    print(f"COMPLETE datasets={','.join(selected)} summary={SUMMARY/'final_asca_v4_io_efficiency.csv'}", flush=True)


if __name__ == "__main__": main()
