#!/usr/bin/env python3
"""Independent-process Peak RAM benchmark for five models on six datasets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "UNIFIED_RAM_BENCHMARK"
SUMMARY = OUT / "summary"
GPU_TABLE = ROOT / "results" / "UNIFIED_EFFICIENCY_REBENCHMARK" / "unified_efficiency_comparison.csv"
TIME_BIN = Path("/usr/bin/time")
MODELS = ("ASCA-AD V4", "PPLAD", "LTFAD", "TranAD", "COUTA")
TABLE_MODELS = ("PPLAD", "LTFAD", "TranAD", "COUTA", "ASCA-AD V4")
DATASETS = ("SKAB", "MSL", "PSM", "PUMP", "HAI", "SMD")
WARMUP = 3
SEED = 42
THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_name(model: str) -> str:
    return model.replace("-", "_").replace(" ", "_")


def case_dir(dataset: str, model: str) -> Path:
    return OUT / dataset / safe_name(model)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def rss_mib(psutil_module) -> float:
    return float(psutil_module.Process(os.getpid()).memory_info().rss / 2**20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_imports() -> dict[str, Any]:
    import psutil
    start_rss = rss_mib(psutil)
    import numpy as np
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import scripts.benchmarks.run_unified_efficiency_rebenchmark as unified
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unified.set_seed()
    device = unified.check_cuda()
    return {"psutil": psutil, "np": np, "torch": torch, "unified": unified,
            "device": device, "process_start_rss_mib": start_rss,
            "post_import_rss_mib": rss_mib(psutil)}


def checkpoint_guard(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def assert_checkpoint(path: Path, before: dict[str, Any]) -> None:
    after = checkpoint_guard(path)
    if any(after[key] != before[key] for key in ("sha256", "size", "mtime_ns")):
        raise RuntimeError(f"Checkpoint modified: {path}")


def run_core(model_name: str, dataset: str, context: dict[str, Any], validation: bool) -> dict[str, Any]:
    np, torch, unified, device = (context[k] for k in ("np", "torch", "unified", "device"))
    psutil = context["psutil"]
    protocol = unified.import_core(dataset)
    display, window, model, score_call, checkpoint = protocol.load_model(unified.MODEL_KEYS[model_name], device)
    if display != model_name:
        raise RuntimeError(f"Model mismatch: {display} != {model_name}")
    model.eval(); checkpoint = Path(checkpoint); guard = checkpoint_guard(checkpoint)
    post_model = rss_mib(psutil)
    test = protocol.load_scaled_test()
    if test.dtype != np.float32 or not np.isfinite(test).all():
        raise RuntimeError("Scaled test must be finite float32")
    starts = unified.core_starts(len(test), window)
    post_data = rss_mib(psutil)
    first_cpu, _ = next(unified.core_cpu_batches(test, starts, window))
    first_gpu = first_cpu.to(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            out = score_call(first_gpu); del out
    torch.cuda.synchronize(device)
    post_warmup = rss_mib(psutil)
    del first_gpu, first_cpu
    processed_windows = 0
    started = time.perf_counter()
    if validation:
        run_starts = starts[:min(len(starts), unified.CORE_BATCH)]
        expected_points = min(len(test), window * len(run_starts))
    else:
        run_starts = starts
        expected_points = len(test)
    with torch.no_grad():
        for cpu_batch, native_count in unified.core_cpu_batches(test, run_starts, window):
            gpu_batch = cpu_batch.to(device, non_blocking=False)
            output = score_call(gpu_batch)
            if int(output.shape[0]) != native_count:
                raise RuntimeError("Core score count mismatch")
            processed_windows += native_count
            del output, gpu_batch, cpu_batch
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    post_full = rss_mib(psutil)
    assert_checkpoint(checkpoint, guard)
    processed_points = expected_points if validation else len(test)
    return {"checkpoint": guard, "post_model_load_rss_mib": post_model,
            "post_data_prepare_rss_mib": post_data, "post_warmup_rss_mib": post_warmup,
            "post_full_test_rss_mib": post_full, "full_test_wall_time_s": wall,
            "raw_test_points": len(test), "processed_points": processed_points,
            "coverage_ratio": processed_points / len(test), "inference_windows": processed_windows,
            "batch_size": unified.CORE_BATCH,
            "input_residency_scope": "scaled CPU test + current CPU/GPU native-window batch",
            "validation_partial": validation}


def run_tranad(dataset: str, context: dict[str, Any], validation: bool) -> dict[str, Any]:
    torch, unified, device, psutil = (context[k] for k in ("torch", "unified", "device", "psutil"))
    benchmark = unified.import_tranad(dataset)
    config = benchmark.load_tranad_psm_config()
    model, _ = benchmark.load_model(config, device)
    model.eval(); checkpoint = Path(benchmark.CHECKPOINT_PATH); guard = checkpoint_guard(checkpoint)
    post_model = rss_mib(psutil)
    adapter = benchmark.TranADPSMDataAdapter("score")
    loader = adapter.loader("test", batch_size=unified.TRANAD_BATCH, shuffle=False)
    post_data = rss_mib(psutil)
    first_cpu = next(iter(loader)); first_gpu = first_cpu.to(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            out = benchmark.formal_score_forward(model, first_gpu); del out
    torch.cuda.synchronize(device); post_warmup = rss_mib(psutil)
    del first_gpu, first_cpu, loader
    adapter = benchmark.TranADPSMDataAdapter("score")
    loader = adapter.loader("test", batch_size=unified.TRANAD_BATCH, shuffle=False)
    processed = 0; windows = 0; started = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            gpu_batch = cpu_batch.to(device, non_blocking=False)
            output = benchmark.formal_score_forward(model, gpu_batch)
            processed += int(output.numel()); windows += int(cpu_batch.shape[0])
            del output, gpu_batch, cpu_batch
            if validation:
                break
    torch.cuda.synchronize(device); wall = time.perf_counter() - started; post_full = rss_mib(psutil)
    adapter.assert_label_free()
    if any("label" in str(p).lower() for p in adapter.audit().get("files_accessed", [])):
        raise RuntimeError("TranAD label access detected")
    raw = int(benchmark.EXPECTED_TEST_POINTS)
    assert_checkpoint(checkpoint, guard)
    return {"checkpoint": guard, "post_model_load_rss_mib": post_model,
            "post_data_prepare_rss_mib": post_data, "post_warmup_rss_mib": post_warmup,
            "post_full_test_rss_mib": post_full, "full_test_wall_time_s": wall,
            "raw_test_points": raw, "processed_points": processed,
            "coverage_ratio": processed / raw, "inference_windows": windows,
            "batch_size": unified.TRANAD_BATCH,
            "input_residency_scope": "indexed Dataset + DataLoader(num_workers=0) + current CPU/GPU batch",
            "validation_partial": validation}


def couta_binding(dataset: str, unified):
    benchmark = unified.import_couta(dataset)
    if hasattr(benchmark, "load_couta_config"):
        return benchmark, benchmark.load_couta_config(), benchmark.PSMCOUTADataAdapter
    return benchmark, benchmark.load_pump_couta_config(), benchmark.PUMPCOUTADataAdapter


def run_couta(dataset: str, context: dict[str, Any], validation: bool) -> dict[str, Any]:
    np, torch, unified, device, psutil = (context[k] for k in ("np", "torch", "unified", "device", "psutil"))
    benchmark, config, adapter_class = couta_binding(dataset, unified)
    checkpoint = Path(benchmark.BUNDLE); guard = checkpoint_guard(checkpoint)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, scaler = benchmark.restore_from_bundle(bundle, str(device))
    net, center = model.net.eval(), model.c
    post_model = rss_mib(psutil)
    adapter = adapter_class("score"); test = adapter.load_test()
    test_scaled = np.asarray(scaler.transform(test), dtype=np.float32)
    del test
    seq_len = int(config["model_config"]["seq_len"])
    windows = unified.couta_cpu_windows(test_scaled, seq_len)
    expected_windows = len(test_scaled) - seq_len + 1
    if int(windows.shape[0]) != expected_windows:
        raise RuntimeError("COUTA CPU window view mismatch")
    post_data = rss_mib(psutil)
    first_gpu = windows[:unified.COUTA_BATCH].contiguous().to(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            out = benchmark.score_batch(net, center, first_gpu); del out
    torch.cuda.synchronize(device); post_warmup = rss_mib(psutil); del first_gpu
    processed_windows = 0; started = time.perf_counter()
    limit = min(expected_windows, unified.COUTA_BATCH) if validation else expected_windows
    with torch.no_grad():
        for start in range(0, limit, unified.COUTA_BATCH):
            stop = min(start + unified.COUTA_BATCH, limit)
            gpu_batch = windows[start:stop].contiguous().to(device, non_blocking=False)
            output = benchmark.score_batch(net, center, gpu_batch)
            processed_windows += int(output.numel())
            del output, gpu_batch
    torch.cuda.synchronize(device); wall = time.perf_counter() - started; post_full = rss_mib(psutil)
    if any("label" in str(p).lower() for p in adapter.audit().get("files_accessed", [])):
        raise RuntimeError("COUTA label access detected")
    raw = len(test_scaled); processed = min(raw, processed_windows + seq_len - 1) if validation else raw
    assert_checkpoint(checkpoint, guard)
    return {"checkpoint": guard, "post_model_load_rss_mib": post_model,
            "post_data_prepare_rss_mib": post_data, "post_warmup_rss_mib": post_warmup,
            "post_full_test_rss_mib": post_full, "full_test_wall_time_s": wall,
            "raw_test_points": raw, "processed_points": processed,
            "coverage_ratio": processed / raw, "inference_windows": processed_windows,
            "batch_size": unified.COUTA_BATCH,
            "input_residency_scope": "scaled CPU test + zero-copy CPU unfold view + current CPU/GPU batch",
            "validation_partial": validation}


def worker(model: str, dataset: str, output: Path, validation: bool) -> None:
    context = worker_imports(); psutil = context["psutil"]
    if model in {"ASCA-AD V4", "PPLAD", "LTFAD"}:
        result = run_core(model, dataset, context, validation)
    elif model == "TranAD":
        result = run_tranad(dataset, context, validation)
    else:
        result = run_couta(dataset, context, validation)
    result.update({"model": model, "dataset": dataset,
        "process_start_rss_mib": context["process_start_rss_mib"],
        "post_import_rss_mib": context["post_import_rss_mib"],
        "num_workers": 0, "torch_num_threads": 1, "torch_num_interop_threads": 1,
        "training": False, "labels_read": False, "threshold_computed": False,
        "prediction_generated": False, "evaluator_called": False,
        "all_windows_on_gpu": False, "full_test_outputs_accumulated": False,
        "legacy_results_modified": False, "timestamp": now()})
    write_json(output, result)


def parse_peak(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if not match:
        raise RuntimeError(f"Cannot parse Maximum RSS: {path}")
    return int(match.group(1)) / 1024.0


def run_subprocess_case(model: str, dataset: str, directory: Path, validation: bool) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / "worker_payload.json"
    report = directory / "time_verbose.txt"
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    command = [str(TIME_BIN), "-v", "-o", str(report), sys.executable, str(Path(__file__).resolve()),
               "--worker", "--model", model, "--dataset", dataset, "--output", str(payload)]
    if validation:
        command.append("--validation-worker")
    env = os.environ.copy(); env.update(THREAD_ENV); env["PYTHONUNBUFFERED"] = "1"; env["LC_ALL"] = "C"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    if process.returncode != 0:
        raise RuntimeError(f"worker exit={process.returncode}; see {stderr_path}")
    data = json.loads(payload.read_text(encoding="utf-8"))
    peak = parse_peak(report); baseline = float(data["post_model_load_rss_mib"])
    data.update({"status": "COMPLETE", "peak_ram_mib": peak,
        "peak_ram_source": "/usr/bin/time -v Maximum resident set size",
        "baseline_rss_mib": baseline, "incremental_ram_mib": peak - baseline,
        "worker_wall_time_s": time.perf_counter() - started})
    return data


def assertions(data: dict[str, Any], validation: bool) -> list[str]:
    failures = []
    expected = {"training": False, "labels_read": False, "threshold_computed": False,
        "prediction_generated": False, "evaluator_called": False,
        "all_windows_on_gpu": False, "full_test_outputs_accumulated": False,
        "num_workers": 0, "torch_num_threads": 1, "torch_num_interop_threads": 1}
    for key, value in expected.items():
        if data.get(key) != value: failures.append(f"{key}!={value}")
    if not validation and data.get("processed_points") != data.get("raw_test_points"):
        failures.append("processed_points!=raw_test_points")
    if not validation and abs(float(data.get("coverage_ratio", 0)) - 1.0) > 1e-12:
        failures.append("coverage_ratio!=1.0")
    if float(data.get("peak_ram_mib", 0)) <= 0: failures.append("peak_ram_mib<=0")
    if float(data.get("peak_ram_mib", 0)) < float(data.get("baseline_rss_mib", 0)):
        failures.append("peak_ram_mib<baseline_rss_mib")
    return failures


def save_case(data: dict[str, Any], directory: Path) -> None:
    failures = assertions(data, validation=False)
    data["status"] = "FAILED" if failures else "COMPLETE"; data["assertion_failures"] = failures
    write_json(directory / "ram_result.json", data)
    write_csv(directory / "ram_result.csv", [data], list(data))
    protocol = {key: data[key] for key in ("model", "dataset", "status", "checkpoint", "batch_size",
        "num_workers", "torch_num_threads", "torch_num_interop_threads", "training", "labels_read",
        "threshold_computed", "prediction_generated", "evaluator_called", "all_windows_on_gpu",
        "full_test_outputs_accumulated", "input_residency_scope", "peak_ram_source", "timestamp")}
    write_json(directory / "protocol_ram.json", protocol)


def validate() -> bool:
    if not TIME_BIN.is_file(): raise FileNotFoundError(TIME_BIN)
    cases = [(model, "SKAB") for model in MODELS]
    results = []
    for index, (model, dataset) in enumerate(cases, 1):
        directory = OUT / "validation" / safe_name(model)
        print(f"[VALIDATION {index}/5] {model}/{dataset}", flush=True)
        try:
            data = run_subprocess_case(model, dataset, directory, validation=True)
            failures = assertions(data, validation=True)
            data["status"] = "FAILED" if failures else "PASS"; data["assertion_failures"] = failures
        except Exception as exc:
            data = {"model": model, "dataset": dataset, "status": "FAILED", "error": str(exc)}
        write_json(directory / "validation_result.json", data); results.append(data)
        print(f"[VALIDATION] {model}: {data['status']}", flush=True)
    passed = all(item["status"] == "PASS" for item in results)
    write_json(OUT / "validation" / "validation_summary.json",
               {"status": "PASS" if passed else "FAILED", "cases": results, "timestamp": now()})
    return passed


def git_snapshot() -> dict[str, Any]:
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    diff = subprocess.run(["git", "diff", "--stat"], cwd=ROOT, text=True, capture_output=True).stdout
    return {"status_short": status.splitlines(), "diff_stat": diff.splitlines()}


def load_existing_results() -> list[dict[str, str]]:
    with GPU_TABLE.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30: raise RuntimeError("Unified GPU table must contain 30 cases")
    return rows


def summarize(pre_git: dict[str, Any], post_git: dict[str, Any]) -> None:
    case_rows, failed = [], []
    for dataset in DATASETS:
        for model in MODELS:
            path = case_dir(dataset, model) / "ram_result.json"
            if not path.is_file():
                failed.append({"Model": model, "Dataset": dataset, "Reason": "missing ram_result.json"}); continue
            data = json.loads(path.read_text(encoding="utf-8")); case_rows.append(data)
            if data["status"] != "COMPLETE":
                failed.append({"Model": model, "Dataset": dataset, "Reason": ";".join(data.get("assertion_failures", []))})
    audit_fields = ["Model", "Dataset", "Peak RAM MiB", "Baseline RSS MiB", "Incremental RAM MiB",
        "Full Test Wall Time s", "Raw Test Points", "Processed Points", "Coverage Ratio", "Inference Windows",
        "Batch Size", "Num Workers", "Torch Threads", "Interop Threads", "Labels Read", "Threshold Computed",
        "Prediction Generated", "Evaluator Called", "All Windows on GPU", "Full Test Outputs Accumulated",
        "Peak Source", "Status", "Evidence", "Notes"]
    audit = []
    for d in case_rows:
        audit.append({"Model": d["model"], "Dataset": d["dataset"], "Peak RAM MiB": d.get("peak_ram_mib", ""),
          "Baseline RSS MiB": d.get("baseline_rss_mib", ""), "Incremental RAM MiB": d.get("incremental_ram_mib", ""),
          "Full Test Wall Time s": d.get("full_test_wall_time_s", ""), "Raw Test Points": d.get("raw_test_points", ""),
          "Processed Points": d.get("processed_points", ""), "Coverage Ratio": d.get("coverage_ratio", ""),
          "Inference Windows": d.get("inference_windows", ""), "Batch Size": d.get("batch_size", ""),
          "Num Workers": d.get("num_workers", ""), "Torch Threads": d.get("torch_num_threads", ""),
          "Interop Threads": d.get("torch_num_interop_threads", ""), "Labels Read": d.get("labels_read", ""),
          "Threshold Computed": d.get("threshold_computed", ""), "Prediction Generated": d.get("prediction_generated", ""),
          "Evaluator Called": d.get("evaluator_called", ""), "All Windows on GPU": d.get("all_windows_on_gpu", ""),
          "Full Test Outputs Accumulated": d.get("full_test_outputs_accumulated", ""), "Peak Source": d.get("peak_ram_source", ""),
          "Status": d["status"], "Evidence": str(case_dir(d["dataset"], d["model"]) / "time_verbose.txt"),
          "Notes": d.get("input_residency_scope", d.get("error", ""))})
    write_csv(SUMMARY / "ram_audit_table.csv", audit, audit_fields)
    write_csv(SUMMARY / "failed_cases.csv", failed, ["Model", "Dataset", "Reason"])

    gpu = load_existing_results(); ram_map = {(d["dataset"], d["model"]): d for d in case_rows if d["status"] == "COMPLETE"}
    gpu_map = {(r["Dataset"], r["Model"]): r for r in gpu}
    main = []
    metric_keys = [("Parameters", "Parameters"), ("State Dict KiB", "State Dict (KiB)"),
                   ("Full Test Time s", "Full Test Time Mean (s)"),
                   ("Throughput points/s", "Throughput (points/s)"), ("Peak RAM MiB", None)]
    for dataset in DATASETS:
        for metric, gpu_key in metric_keys:
            row = {"Dataset": dataset, "Metric": metric}
            for model in TABLE_MODELS:
                if gpu_key:
                    row[model] = gpu_map[(dataset, model)][gpu_key]
                else:
                    row[model] = ram_map.get((dataset, model), {}).get("peak_ram_mib", "")
            main.append(row)
    main_fields = ["Dataset", "Metric", *TABLE_MODELS]
    write_csv(SUMMARY / "unified_ram_comparison.csv", main, main_fields)
    write_json(SUMMARY / "unified_ram_comparison.json", {"rows": main})

    complete = len(case_rows) - len(failed)
    averages, maxima = {}, {}
    for model in MODELS:
        values = [float(d["peak_ram_mib"]) for d in case_rows if d["model"] == model and d["status"] == "COMPLETE"]
        if values: averages[model] = sum(values) / len(values); maxima[model] = max(values)
    lines = ["# Unified Peak RAM Benchmark", "", f"- Status: {'COMPLETE' if not failed else 'PARTIAL'}",
             f"- Successful cases: {complete}/30", f"- Failed cases: {len(failed)}", "",
             "## Six-dataset Peak RAM summary", "", "| Model | Mean Peak RAM (MiB) | Max Peak RAM (MiB) |",
             "|---|---:|---:|"]
    for model in MODELS:
        lines.append(f"| {model} | {averages.get(model, float('nan')):.3f} | {maxima.get(model, float('nan')):.3f} |")
    lines += ["", "## Interpretation", "",
              "Peak RAM is /usr/bin/time -v Maximum RSS from one fresh process per case.",
              "Full Test Wall Time is audit-only and does not replace the unified CUDA-event Full Test Time.",
              "No labels, thresholds, predictions, evaluators, training, or accumulated outputs are used.",
              "The paper table is eligible only when failed_cases.csv is empty."]
    (SUMMARY / "execution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation = json.loads((OUT / "validation" / "validation_summary.json").read_text(encoding="utf-8"))
    write_json(SUMMARY / "protocol_ram.json", {"status": "COMPLETE" if not failed else "PARTIAL",
      "planned_cases": 30, "successful_cases": complete, "failed_cases": len(failed),
      "validation": validation["status"], "peak_source": "/usr/bin/time -v Maximum resident set size",
      "warmup": WARMUP, "full_test_repeat": 1, "thread_env": THREAD_ENV,
      "num_workers": 0, "independent_process_per_case": True, "training": False,
      "labels_read": False, "threshold_computed": False, "prediction_generated": False,
      "evaluator_called": False, "historical_results_modified": False,
      "unified_gpu_efficiency_modified": False, "git_before": pre_git, "git_after": post_git,
      "generated_at": now()})


def run_all(resume: bool, retry_failed: bool, only_model: str | None, only_dataset: str | None) -> None:
    pre_git = git_snapshot(); failures = []
    cases = [(d, m) for d in DATASETS for m in MODELS if (not only_model or m == only_model) and (not only_dataset or d == only_dataset)]
    for index, (dataset, model) in enumerate(cases, 1):
        directory = case_dir(dataset, model); result_path = directory / "ram_result.json"
        if result_path.exists() and resume:
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if old.get("status") == "COMPLETE" or (old.get("status") == "FAILED" and not retry_failed):
                print(f"[{index}/{len(cases)}] SKIP {dataset}/{model}: {old.get('status')}", flush=True); continue
        print(f"[{index}/{len(cases)}] RUN {dataset}/{model} Start={now()}", flush=True)
        try:
            data = run_subprocess_case(model, dataset, directory, validation=False)
            save_case(data, directory)
            print(f"[{dataset}/{model}] {data['peak_ram_mib']:.3f} MiB | status={data['status']}", flush=True)
        except Exception as exc:
            failures.append((dataset, model, str(exc)))
            error = {"model": model, "dataset": dataset, "status": "FAILED", "error": str(exc),
                     "traceback": traceback.format_exc(), "timestamp": now()}
            write_json(result_path, error)
            print(f"[{dataset}/{model}] FAILED: {exc}", flush=True)
    if only_model or only_dataset:
        print("Subset run complete; summary requires all 30 cases.", flush=True); return
    summarize(pre_git, git_snapshot())
    print(f"All cases finished; immediate_failures={len(failures)} summary={SUMMARY}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if not args.model or not args.dataset or not args.output:
            raise SystemExit("worker requires --model --dataset --output")
        worker(args.model, args.dataset, args.output, args.validation_worker); return
    OUT.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        if not validate(): raise SystemExit("Validation FAILED")
        print("Validation PASS", flush=True); return
    if not validate(): raise SystemExit("Validation FAILED; formal cases not started")
    run_all(resume=args.resume, retry_failed=args.retry_failed,
            only_model=args.model, only_dataset=args.dataset)


if __name__ == "__main__":
    main()
