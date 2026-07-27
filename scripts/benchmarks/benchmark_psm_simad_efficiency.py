#!/usr/bin/env python3
"""Inference-only efficiency benchmark for the formal PSM SimAD checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmarks.benchmark_skab_efficiency as protocol  # noqa: E402
from scripts.benchmarks.adapters.simad_dataset_adapter import (  # noqa: E402
    SimADWindowScoreAdapter,
)
from scripts.benchmarks.run_psm_simad_detection import build_model  # noqa: E402


OUTPUT_ROOT = ROOT / "results" / "PSM_SIMAD_RESULTS" / "Efficiency"
CHECKPOINT = (
    ROOT / "results" / "PSM_SIMAD_RESULTS" / "Detection" /
    "SimAD" / "SimAD_state_dict.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_scaled_psm_test() -> np.ndarray:
    """Load feature arrays only; PSM_test_label.npy is never opened."""
    dataset = ROOT / "dataset" / "PSM"
    train = np.asarray(
        np.load(dataset / "PSM_train.npy", allow_pickle=False),
        dtype=np.float32,
    )
    test = np.asarray(
        np.load(dataset / "PSM_test.npy", allow_pickle=False),
        dtype=np.float32,
    )
    if train.shape != (132481, 25) or test.shape != (87841, 25):
        raise RuntimeError(f"Unexpected PSM shapes: {train.shape}, {test.shape}")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError("PSM train/test contains NaN or Inf")
    scaler = StandardScaler()
    scaler.fit(train)
    return scaler.transform(test).astype(np.float32, copy=False)


def load_simad_model(_model_name: str, device: torch.device):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    scorer = SimADWindowScoreAdapter(model).to(device).eval()
    return "SimAD", 2048, model, scorer.window_scores, CHECKPOINT


def configure_protocol() -> None:
    protocol.ROOT = ROOT
    protocol.OUTPUT_ROOT = OUTPUT_ROOT
    protocol.load_scaled_test = load_scaled_psm_test
    protocol.load_model = load_simad_model


def run_benchmark(overwrite: bool) -> None:
    comparison_path = OUTPUT_ROOT / "comparison_efficiency.json"
    if comparison_path.exists() and not overwrite:
        raise FileExistsError(
            f"PSM SimAD efficiency result already exists: {comparison_path}. "
            "Use --overwrite only for an explicitly authorized rerun."
        )
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Run detection training first; checkpoint not found: {CHECKPOINT}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_hash_before = sha256(CHECKPOINT)
    protocol.run_worker("simad")
    checkpoint_hash_after = sha256(CHECKPOINT)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError("SimAD source checkpoint changed during benchmark")

    result_path = OUTPUT_ROOT / "SimAD" / "efficiency.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["checkpoint_sha256_before"] = checkpoint_hash_before
    result["checkpoint_sha256_after"] = checkpoint_hash_after
    result["protocol"].update({
        "training": False,
        "label_access": False,
        "evaluator_called": False,
        "checkpoint_modified": False,
        "model_source_modified": False,
        "data_loading_excluded": True,
        "host_to_device_transfer_excluded": True,
        "threshold_computed": False,
        "prediction_computed": False,
        "pa_computed": False,
    })
    write_json(result_path, result)

    fields = [
        "Model",
        "Parameters",
        "State Dict (KiB)",
        "Latency B=1 (ms)",
        "Latency B=128 (ms)",
        "Full Test Time (s)",
        "Throughput (points/s)",
        "GPU Peak (MiB)",
        "GPU Incremental (MiB)",
    ]
    with (OUTPUT_ROOT / "comparison_efficiency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "Model": result["model"],
            "Parameters": result["trainable_parameters"],
            "State Dict (KiB)": result["state_dict"]["kib"],
            "Latency B=1 (ms)": result["latency_batch_1_ms"]["mean"],
            "Latency B=128 (ms)": result["latency_batch_128_ms"]["mean"],
            "Full Test Time (s)": result["full_test"]["mean"],
            "Throughput (points/s)":
                result["full_test"]["throughput_points_per_second"],
            "GPU Peak (MiB)": result["gpu_memory"]["peak_allocated_mib"],
            "GPU Incremental (MiB)":
                result["gpu_memory"]["incremental_allocated_mib"],
        })

    audit = {
        "torch_no_grad": True,
        "model_eval": True,
        "dtype": "float32",
        "cuda_event": True,
        "warmup": protocol.WARMUP,
        "latency_repeat": protocol.REPEATS,
        "full_test_repeat": protocol.FULL_TEST_REPEATS,
        "training": False,
        "label_access": False,
        "evaluator_called": False,
        "checkpoint_modified": False,
        "model_source_modified": False,
        "data_loading_excluded": True,
        "host_to_device_transfer_excluded": True,
        "threshold_computed": False,
        "prediction_computed": False,
        "pa_computed": False,
    }
    write_json(comparison_path, {
        "dataset": "PSM",
        "metric_scope": "SimAD inference efficiency only",
        "protocol": audit,
        "models": [result],
    })
    write_json(OUTPUT_ROOT / "protocol_efficiency.json", {
        "dataset": "PSM",
        "model": "SimAD",
        "gpu": result["environment"]["gpu"],
        "device": result["device"],
        "torch_version": result["torch_version"],
        "cuda_version": result["cuda_version"],
        "dtype": "float32",
        "window_size": result["window_size"],
        "input_dimension": 25,
        "input_shape": result["input_shape"],
        "processed_points": result["full_test"]["points"],
        **audit,
    })


def main() -> None:
    args = parse_args()
    configure_protocol()
    run_benchmark(args.overwrite)


if __name__ == "__main__":
    main()
