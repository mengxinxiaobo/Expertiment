#!/usr/bin/env python3
"""Old-vs-V4-IO score and detection equivalence audit.

Labels are loaded only after both label-free score stages have completed.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from scripts.benchmarks.asca_optimization_common import (
    ANOMALY_RATIOS, BATCH_SIZE, DATASETS, OUT, ROOT, WINDOW, array_equivalence,
    build_optimized_from_old, checkpoint_compatibility, cpu_batches, formal_starts,
    load_old_model, load_scaled_data, point_adjust, protected_snapshot, require_cuda,
    save_json, set_seed,
)


def metrics(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def generate_energy(
    data: np.ndarray,
    split: str,
    score_call: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    output: Path,
    optimized: bool,
) -> dict[str, int | float]:
    starts = formal_starts(len(data), WINDOW, split)
    size = len(starts) * WINDOW
    output.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(size,))
    offset = 0
    began = time.perf_counter()
    context = torch.inference_mode if optimized else torch.no_grad
    with context():
        for batch_index, cpu_batch in enumerate(cpu_batches(data, starts), 1):
            gpu_batch = cpu_batch.to(device, non_blocking=False)
            score = score_call(gpu_batch)
            values = score.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            target[offset : offset + values.size] = values
            offset += values.size
            del score, gpu_batch, cpu_batch
            if batch_index % 500 == 0:
                print(f"[{split}][{'optimized' if optimized else 'old'}] batches={batch_index} values={offset}/{size}", flush=True)
    target.flush()
    if offset != size:
        raise RuntimeError(f"incomplete {split} energy: {offset} != {size}")
    return {
        "windows": len(starts), "energy_length": size, "processed_points": size,
        "source_points": len(data), "coverage_ratio": size / max(len(data), 1),
        "stride": 1 if split == "train" else WINDOW,
        "tail_points_dropped": len(data) - (starts[-1] + WINDOW),
        "elapsed_seconds": time.perf_counter() - began,
    }


def evaluate_pair(
    dataset: str, old_train: Path, new_train: Path, old_test: Path, new_test: Path,
    output_dir: Path,
) -> dict:
    # Label access begins here and nowhere in either model/score stage.
    from scripts.benchmarks.asca_optimization_common import dataset_paths
    _train_path, _test_path, label_path = dataset_paths(dataset)
    labels = np.asarray(np.load(label_path, allow_pickle=False)).reshape(-1).astype(np.int64)
    old_train_values = np.load(old_train, mmap_mode="r")
    new_train_values = np.load(new_train, mmap_mode="r")
    old_test_values = np.load(old_test, mmap_mode="r")
    new_test_values = np.load(new_test, mmap_mode="r")
    evaluated = len(old_test_values)
    if evaluated > labels.size or not set(np.unique(labels)).issubset({0, 1}):
        raise RuntimeError(f"invalid {dataset} labels for final evaluation")
    labels = labels[:evaluated]
    percentile = 100.0 - ANOMALY_RATIOS[dataset]
    old_threshold = float(np.percentile(np.concatenate([old_train_values, old_test_values]), percentile))
    new_threshold = float(np.percentile(np.concatenate([new_train_values, new_test_values]), percentile))
    old_raw = (old_test_values > old_threshold).astype(np.int64)
    new_raw = (new_test_values > new_threshold).astype(np.int64)
    old_pa = point_adjust(old_raw, labels)
    new_pa = point_adjust(new_raw, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "old_raw_prediction.npy", old_raw, allow_pickle=False)
    np.save(output_dir / "optimized_raw_prediction.npy", new_raw, allow_pickle=False)
    np.save(output_dir / "old_pa_prediction.npy", old_pa, allow_pickle=False)
    np.save(output_dir / "optimized_pa_prediction.npy", new_pa, allow_pickle=False)
    threshold_diff = abs(old_threshold - new_threshold)
    old_raw_metrics, new_raw_metrics = metrics(old_raw, labels), metrics(new_raw, labels)
    old_pa_metrics, new_pa_metrics = metrics(old_pa, labels), metrics(new_pa, labels)
    metrics_equal = old_raw_metrics == new_raw_metrics and old_pa_metrics == new_pa_metrics
    status = "PASS" if (
        threshold_diff <= 1e-10
        and np.array_equal(old_raw, new_raw)
        and np.array_equal(old_pa, new_pa)
        and metrics_equal
    ) else "FAIL"
    return {
        "status": status, "dataset": dataset,
        "anomaly_ratio": ANOMALY_RATIOS[dataset], "percentile": percentile,
        "old_threshold": old_threshold, "optimized_threshold": new_threshold,
        "threshold_absolute_difference": threshold_diff,
        "raw_prediction_equal": bool(np.array_equal(old_raw, new_raw)),
        "pa_prediction_equal": bool(np.array_equal(old_pa, new_pa)),
        "detection_metrics_equal": metrics_equal,
        "old_raw": old_raw_metrics, "optimized_raw": new_raw_metrics,
        "old_pa": old_pa_metrics, "optimized_pa": new_pa_metrics,
        "evaluated_points": evaluated, "label_access_stage": "final_evaluator_only",
    }


def run(dataset: str, resume: bool = False) -> dict:
    dataset_out = OUT / dataset
    score_out = dataset_out / "score_equivalence"
    detection_out = dataset_out / "detection_equivalence"
    result_path = detection_out / "detection_equivalence.json"
    if resume and result_path.is_file():
        import json
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS":
            print(f"[{dataset}] detection equivalence already PASS; skip", flush=True)
            return previous

    before = protected_snapshot()
    set_seed()
    device = require_cuda()
    compatibility = checkpoint_compatibility(dataset, device)
    if compatibility["status"] != "PASS":
        raise RuntimeError(f"{dataset} checkpoint compatibility failed")
    _protocol, old_model, old_score_call, checkpoint = load_old_model(dataset, device)
    optimized_model, optimized_score_call = build_optimized_from_old(old_model, checkpoint, device)
    train, test, _labels = load_scaled_data(dataset, include_label=False)
    paths = {
        "old_train": score_out / "old_train_energy.npy",
        "new_train": score_out / "optimized_train_energy.npy",
        "old_test": score_out / "old_scores.npy",
        "new_test": score_out / "optimized_scores.npy",
    }
    metadata = {
        "old_train": generate_energy(train, "train", old_score_call, device, paths["old_train"], False),
        "old_test": generate_energy(test, "test", old_score_call, device, paths["old_test"], False),
        "optimized_train": generate_energy(train, "train", optimized_score_call, device, paths["new_train"], True),
        "optimized_test": generate_energy(test, "test", optimized_score_call, device, paths["new_test"], True),
    }
    train_eq = array_equivalence(paths["old_train"], paths["new_train"], score_out / "train_score_diff.npy")
    test_eq = array_equivalence(paths["old_test"], paths["new_test"], score_out / "score_diff.npy")
    score_status = "PASS" if train_eq["status"] == test_eq["status"] == "PASS" else "FAIL"
    score_result = {
        "status": score_status, "dataset": dataset, "checkpoint": compatibility,
        "protocol": {"window": WINDOW, "batch_size": BATCH_SIZE, "dtype": "float32",
                     "train_stride": 1, "test_stride": WINDOW,
                     "tail_handling": "formal loader: drop incomplete final window"},
        "metadata": metadata, "train_equivalence": train_eq, "test_equivalence": test_eq,
        "training": False, "label_access": False,
    }
    save_json(score_out / "score_equivalence.json", score_result)
    if score_status != "PASS":
        raise RuntimeError(f"{dataset} score equivalence FAILED; detection/all-dataset continuation forbidden")
    detection = evaluate_pair(
        dataset, paths["old_train"], paths["new_train"], paths["old_test"], paths["new_test"],
        detection_out,
    )
    from scripts.benchmarks.asca_optimization_common import sha256
    checkpoint_hash_after = sha256(checkpoint)
    after = protected_snapshot()
    detection["source_integrity_unchanged"] = before == after
    detection["checkpoint_sha256"] = compatibility["checkpoint_sha256"]
    detection["checkpoint_modified"] = checkpoint_hash_after != compatibility["checkpoint_sha256"]
    if not detection["source_integrity_unchanged"] or detection["checkpoint_modified"]:
        detection["status"] = "FAIL"
    save_json(result_path, detection)
    if detection["status"] != "PASS":
        raise RuntimeError(f"{dataset} detection equivalence FAILED")
    print(
        f"[{dataset}] score=PASS threshold_diff={detection['threshold_absolute_difference']:.3e} "
        f"RAW_equal={detection['raw_prediction_equal']} PA_equal={detection['pa_prediction_equal']}",
        flush=True,
    )
    del old_model, optimized_model, train, test
    gc.collect(); torch.cuda.empty_cache()
    return detection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.dataset, args.resume)


if __name__ == "__main__":
    main()
