#!/usr/bin/env python3
"""Gated orchestration and summary generation for ASCA-AD V4-IO Phase 1."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from scripts.benchmarks.asca_optimization_common import (
    DATASETS, OUT, ROOT, git_snapshot, protected_snapshot, save_json,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def call(script: str, *arguments: str, log_dir: Path | None = None) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "benchmarks" / script), *arguments]
    if log_dir is None:
        subprocess.run(command, cwd=ROOT, check=True)
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = log_dir / "stdout.log", log_dir / "stderr.log"
    stderr_path.touch(exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def passed(path: Path) -> bool:
    return path.is_file() and load_json(path).get("status") in {"PASS", "COMPLETE"}


def validate(dataset: str) -> None:
    call("validate_asca_inference_optimization.py", "--dataset", dataset, log_dir=OUT / dataset)


def detection(dataset: str, resume: bool) -> None:
    path = OUT / dataset / "detection_equivalence" / "detection_equivalence.json"
    if resume and passed(path):
        print(f"[{dataset}] detection already PASS; skip", flush=True)
        return
    arguments = ["--dataset", dataset]
    if resume:
        arguments.append("--resume")
    call("run_asca_optimized_detection.py", *arguments, log_dir=OUT / dataset)


def efficiency(dataset: str, resume: bool, old_only: bool, optimized_only: bool) -> None:
    path = OUT / dataset / "efficiency" / "efficiency_comparison.json"
    if resume and not old_only and not optimized_only and passed(path):
        print(f"[{dataset}] efficiency already COMPLETE; skip", flush=True)
        return
    arguments = ["--dataset", dataset]
    if old_only:
        arguments.append("--old-only")
    if optimized_only:
        arguments.append("--optimized-only")
    call("run_asca_optimized_efficiency.py", *arguments, log_dir=OUT / dataset)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def summarize(source_before: dict[str, Any], git_before: dict[str, Any]) -> None:
    summary = OUT / "summary"
    score_rows, detection_rows, efficiency_rows, speed_rows, memory_rows, failures = [], [], [], [], [], []
    compatibility_rows = []
    for dataset in DATASETS:
        score_path = OUT / dataset / "score_equivalence" / "score_equivalence.json"
        detection_path = OUT / dataset / "detection_equivalence" / "detection_equivalence.json"
        efficiency_path = OUT / dataset / "efficiency" / "efficiency_comparison.json"
        if score_path.is_file():
            score = load_json(score_path)
            test = score["test_equivalence"]
            train = score["train_equivalence"]
            score_rows.append({
                "Dataset": dataset, "Status": score["status"],
                "Train Max Abs Error": train.get("max_absolute_error"),
                "Test Max Abs Error": test.get("max_absolute_error"),
                "Train Allclose": train.get("allclose_atol_1e-7_rtol_1e-6"),
                "Test Allclose": test.get("allclose_atol_1e-7_rtol_1e-6"),
                "Test Array Equal": test.get("array_equal"),
            })
            compatibility_rows.append(score["checkpoint"])
            if score["status"] != "PASS": failures.append({"Dataset": dataset, "Stage": "score", "Reason": score["status"]})
        if detection_path.is_file():
            item = load_json(detection_path)
            detection_rows.append({
                "Dataset": dataset, "Status": item["status"],
                "Threshold Difference": item["threshold_absolute_difference"],
                "RAW Equal": item["raw_prediction_equal"], "PA Equal": item["pa_prediction_equal"],
                "Metrics Equal": item["detection_metrics_equal"], "Evaluated Points": item["evaluated_points"],
            })
            if item["status"] != "PASS": failures.append({"Dataset": dataset, "Stage": "detection", "Reason": item["status"]})
        if efficiency_path.is_file():
            item = load_json(efficiency_path); old, new = item["old"], item["optimized"]
            row = {
                "Dataset": dataset,
                "Old Params": old["parameters"], "Optimized Params": new["parameters"],
                "Old State KiB": old["state_dict_kib"], "Optimized State KiB": new["state_dict_kib"],
                "Old B1 ms": old["latency_b1"]["mean_ms"], "Optimized B1 ms": new["latency_b1"]["mean_ms"],
                "Old B128 ms": old["latency_b128"]["mean_ms"], "Optimized B128 ms": new["latency_b128"]["mean_ms"],
                "Old Full s": old["full_test_time_mean_s"], "Optimized Full s": new["full_test_time_mean_s"],
                "Old Throughput": old["throughput_points_per_s"], "Optimized Throughput": new["throughput_points_per_s"],
                "Old GPU Peak MiB": old["gpu_peak_mib"], "Optimized GPU Peak MiB": new["gpu_peak_mib"],
                "Old GPU Incremental MiB": old["gpu_incremental_mib"], "Optimized GPU Incremental MiB": new["gpu_incremental_mib"],
                "Old Peak RSS MiB": old["peak_process_rss_mib"], "Optimized Peak RSS MiB": new["peak_process_rss_mib"],
                "Old Peak USS MiB": old["peak_uss_mib"], "Optimized Peak USS MiB": new["peak_uss_mib"],
            }
            efficiency_rows.append(row)
            diffs = item["differences"]
            speed_rows.append({
                "Dataset": dataset,
                "B1 Speedup": diffs["latency_b1_ms"]["speedup_old_over_optimized"],
                "B128 Speedup": diffs["latency_b128_ms"]["speedup_old_over_optimized"],
                "Full Test Speedup": diffs["full_test_time_s"]["speedup_old_over_optimized"],
                "Throughput Improvement %": diffs["throughput_points_per_s"]["percentage_improvement"],
            })
            memory_rows.append({
                "Dataset": dataset,
                "GPU Peak Reduction %": diffs["gpu_peak_mib"]["percentage_reduction"],
                "GPU Incremental Reduction %": diffs["gpu_incremental_mib"]["percentage_reduction"],
                "Peak RSS Reduction %": diffs["peak_rss_mib"]["percentage_reduction"],
                "Peak USS Reduction %": diffs["peak_uss_mib"]["percentage_reduction"],
            })

    source_after = protected_snapshot()
    git_after = git_snapshot()
    before_trees = source_before.get("trees", {})
    after_trees = source_after.get("trees", {})
    checkpoint_modified = any(
        load_json(OUT / row["dataset"] / "detection_equivalence" / "detection_equivalence.json").get("checkpoint_modified", True)
        for row in compatibility_rows
        if (OUT / row["dataset"] / "detection_equivalence" / "detection_equivalence.json").is_file()
    ) if compatibility_rows else False
    integrity = {
        "existing_model_modified": source_before["files"].get("asca_ad/model.py") != source_after["files"].get("asca_ad/model.py"),
        "existing_adapter_modified": source_before["files"].get("scripts/benchmarks/adapters/asca_adapter.py") != source_after["files"].get("scripts/benchmarks/adapters/asca_adapter.py"),
        "historical_results_modified": source_before["all_historical_result_trees"] != source_after["all_historical_result_trees"],
        "checkpoint_modified": checkpoint_modified,
        "detection_results_modified": source_before["all_historical_result_trees"] != source_after["all_historical_result_trees"],
        "historical_efficiency_modified": before_trees.get("results/UNIFIED_EFFICIENCY_COMPARISON") != after_trees.get("results/UNIFIED_EFFICIENCY_COMPARISON"),
        "ram_results_modified": before_trees.get("results/UNIFIED_RAM_BENCHMARK") != after_trees.get("results/UNIFIED_RAM_BENCHMARK"),
        "memory_audit_modified": before_trees.get("results/MEMORY_ROOT_CAUSE_AUDIT") != after_trees.get("results/MEMORY_ROOT_CAUSE_AUDIT"),
        "before": source_before, "after": source_after,
        "git_before": git_before, "git_after": git_after,
    }
    integrity["status"] = "PASS" if not any(
        integrity[key] for key in (
            "existing_model_modified", "existing_adapter_modified", "historical_results_modified",
            "checkpoint_modified", "historical_efficiency_modified", "ram_results_modified", "memory_audit_modified",
        )
    ) else "FAIL"
    if integrity["status"] != "PASS": failures.append({"Dataset": "ALL", "Stage": "source_integrity", "Reason": "protected artifact changed"})

    write_csv(summary / "score_equivalence_table.csv", score_rows,
              ["Dataset", "Status", "Train Max Abs Error", "Test Max Abs Error", "Train Allclose", "Test Allclose", "Test Array Equal"])
    write_csv(summary / "detection_equivalence_table.csv", detection_rows,
              ["Dataset", "Status", "Threshold Difference", "RAW Equal", "PA Equal", "Metrics Equal", "Evaluated Points"])
    if efficiency_rows:
        write_csv(summary / "efficiency_old_vs_optimized.csv", efficiency_rows, list(efficiency_rows[0]))
        write_csv(summary / "efficiency_speedup_table.csv", speed_rows, list(speed_rows[0]))
        write_csv(summary / "memory_reduction_table.csv", memory_rows, list(memory_rows[0]))
    write_csv(summary / "failed_cases.csv", failures, ["Dataset", "Stage", "Reason"])
    save_json(summary / "checkpoint_compatibility.json", {"datasets": compatibility_rows})
    save_json(summary / "source_integrity_audit.json", integrity)
    save_json(summary / "protocol_optimization.json", {
        "phase": 1, "model": "ASCA-AD V4-IO", "score_mode": "total",
        "window": 100, "batch_size": 128, "dtype": "float32",
        "training": False, "label_parameter_selection": False,
        "optimized_inference_mode": True, "state_dict_changed": False,
    })
    equivalence_complete = len(score_rows) == len(detection_rows) == len(DATASETS) and not failures
    speedups = [row["Full Test Speedup"] for row in speed_rows]
    has_speedup = bool(speedups) and sum(speedups) / len(speedups) > 1.0
    status = "COMPLETE" if equivalence_complete and len(efficiency_rows) == len(DATASETS) and has_speedup else (
        "COMPLETE_NO_SPEEDUP" if equivalence_complete and len(efficiency_rows) == len(DATASETS) else
        "FAILED" if failures else "PARTIAL"
    )
    save_json(summary / "optimization_status.json", {"status": status, "failed_cases": failures})
    score_digest = "; ".join(
        f"{row['Dataset']}={row['Test Max Abs Error']} (allclose={row['Test Allclose']})"
        for row in score_rows
    ) or "not run"
    detection_digest = "; ".join(
        f"{row['Dataset']}: threshold_diff={row['Threshold Difference']}, "
        f"RAW={row['RAW Equal']}, PA={row['PA Equal']}, metrics={row['Metrics Equal']}"
        for row in detection_rows
    ) or "not run"
    speed_digest = "; ".join(
        f"{row['Dataset']}: B1={row['B1 Speedup']}x, B128={row['B128 Speedup']}x, "
        f"Full={row['Full Test Speedup']}x, throughput={row['Throughput Improvement %']}%"
        for row in speed_rows
    ) or "not run"
    memory_digest = "; ".join(
        f"{row['Dataset']}: GPU peak={row['GPU Peak Reduction %']}%, "
        f"GPU incremental={row['GPU Incremental Reduction %']}%, "
        f"RSS={row['Peak RSS Reduction %']}%, USS={row['Peak USS Reduction %']}%"
        for row in memory_rows
    ) or "not run"
    report = f"""# ASCA-AD V4-IO Phase-1 execution summary

1. New files: independent model/solver under `asca_ad_optimized/`, a new adapter, and new validation/Detection/Efficiency/orchestration scripts.
2. Existing formal files modified: model={integrity['existing_model_modified']}, adapter={integrity['existing_adapter_modified']}, historical results={integrity['historical_results_modified']}.
3. Checkpoint strict compatibility: see `checkpoint_compatibility.json` ({len(compatibility_rows)} datasets recorded).
4. Parameters remain 146: {all(row.get('old_parameters') == row.get('optimized_parameters') == 146 for row in compatibility_rows) if compatibility_rows else 'not run'}.
5. State Dict keys/shapes identical: {all(row.get('state_dict_keys_equal') and row.get('state_dict_shapes_equal') for row in compatibility_rows) if compatibility_rows else 'not run'}.
6. Total formula unchanged: `local_fit + global_fit`, then the original total min/max scaling and softmax.
7. Removed inference work: Normal CDF/area errors, gap, combined, local/global probabilities, full output details, gap/combined normalization.
8. Skipped eval softmax: the unused selector-probability softmax inside both local and global Top-k branches.
9. Only total normalization: yes.
10. `torch.inference_mode()`: yes, optimized adapter and optimized benchmark path.
11. Fixed lag/index caching: yes, non-persistent normalized-lag and left/right index buffers.
12. Six-dataset point-score equivalence: {score_digest}.
13. Threshold differences: {detection_digest}.
14. RAW equality: reported per dataset above and in `detection_equivalence_table.csv`.
15. PA/Detection metric equality: reported per dataset above and in `detection_equivalence_table.csv`.
16. B=1 speed change: {speed_digest}.
17. B=128 speed change: see the same per-dataset speed digest and `efficiency_speedup_table.csv`.
18. Full-test speed change: see the same per-dataset speed digest.
19. Throughput change: see the same per-dataset speed digest.
20. GPU Peak change: {memory_digest}.
21. GPU Incremental change: see the same per-dataset memory digest.
22. Peak RSS/USS change: see the same per-dataset memory digest.
23. Largest individual optimization contribution: not claimed; Phase 1 measures a safe bundle and does not perform feature ablation.
24. Negative optimization: retained transparently in the speed and memory tables; none is discarded.
25. Replace formal inference: recommended only when status is COMPLETE and gains are stable; never replace training code automatically.
26. Continue Phase 2 chunked gather: only if Phase 1 is equivalent and a material gather working-set cost remains.

Final status: **{status}**. Failed cases: {failures or 'none'}.
"""
    (summary / "execution_summary.md").write_text(report, encoding="utf-8")
    print(f"summary_status={status} summary={summary / 'execution_summary.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-skab", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--old-only", action="store_true")
    parser.add_argument("--optimized-only", action="store_true")
    args = parser.parse_args()
    if args.old_only and args.optimized_only:
        raise SystemExit("--old-only and --optimized-only are mutually exclusive")
    source_before = protected_snapshot()
    git_before = git_snapshot()
    if args.validate_only:
        for dataset in tuple(args.dataset or ("SKAB",)):
            validate(dataset)
        return
    selected = tuple(args.dataset or (("SKAB",) if args.run_skab else DATASETS))
    if args.run_all:
        selected = DATASETS

    # Mandatory gate: SKAB checkpoint, score, Detection and efficiency first.
    if "SKAB" in selected or len(selected) > 1:
        validate("SKAB")
        detection("SKAB", args.resume)
        efficiency("SKAB", args.resume, args.old_only, args.optimized_only)
        gate_paths = (
            OUT / "SKAB" / "score_equivalence" / "score_equivalence.json",
            OUT / "SKAB" / "detection_equivalence" / "detection_equivalence.json",
        )
        if not all(passed(path) for path in gate_paths):
            raise SystemExit("SKAB gate FAILED; six-dataset continuation forbidden")
    for dataset in selected:
        if dataset == "SKAB":
            continue
        validate(dataset)
        detection(dataset, args.resume)
        efficiency(dataset, args.resume, args.old_only, args.optimized_only)
    summarize(source_before, git_before)


if __name__ == "__main__":
    main()
