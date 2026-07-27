#!/usr/bin/env python3
"""Inference-only SMD efficiency benchmark for the three formal models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmarks.benchmark_skab_efficiency as protocol  # noqa: E402
from scripts.benchmarks.adapters.ltfad_adapter import LTFADScoreAdapter  # noqa: E402
from scripts.benchmarks.adapters.pplad_adapter import PPLADScoreAdapter  # noqa: E402


OUTPUT_ROOT = ROOT / "results" / "SMD_PAPER_RESULTS" / "Efficiency"
ASCA_CHECKPOINT = (
    ROOT / "checkpoints" / "SMD" /
    "SMD_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
    "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
)
PPLAD_CHECKPOINT = (
    ROOT / "results" / "SMD_PAPER_RESULTS" / "Detection" /
    "PPLAD" / "PPLAD_state_dict.pt"
)
LTFAD_CHECKPOINT = (
    ROOT / "results" / "SMD_PAPER_RESULTS" / "Detection" /
    "LTFAD" / "LTFAD_state_dict.pt"
)
CHECKPOINTS = {
    "asca": ASCA_CHECKPOINT,
    "pplad": PPLAD_CHECKPOINT,
    "ltfad": LTFAD_CHECKPOINT,
}
MODEL_DIRS = {
    "asca": "ASCA-AD_V4",
    "pplad": "PPLAD",
    "ltfad": "LTFAD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("asca", "pplad", "ltfad"))
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


def load_scaled_smd_test() -> np.ndarray:
    """Load train/test features only; SMD_test_label.npy is never opened."""
    dataset = ROOT / "dataset" / "SMD"
    train = np.asarray(
        np.load(dataset / "SMD_train.npy", allow_pickle=False),
        dtype=np.float32,
    )
    test = np.asarray(
        np.load(dataset / "SMD_test.npy", allow_pickle=False),
        dtype=np.float32,
    )
    if train.shape != (708405, 38) or test.shape != (708420, 38):
        raise RuntimeError(f"Unexpected SMD shapes: {train.shape}, {test.shape}")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError("SMD train/test contains NaN or Inf")
    scaler = StandardScaler()
    scaler.fit(train)
    return scaler.transform(test).astype(np.float32, copy=False)


def build_official_solver(official_solver, config: dict, device: torch.device):
    """Build inference state without constructing loaders or training."""
    solver = object.__new__(official_solver.Solver)
    solver.__dict__.update(config)
    solver.device = device
    solver.build_model()
    solver.model = solver.model.to(device)
    solver.criterion = nn.MSELoss()
    solver.criterion_keep = nn.MSELoss(reduction="none")
    return solver


def load_smd_model(model_name: str, device: torch.device):
    if model_name == "asca":
        import scripts.benchmarks.generate_asca_v4_skab_score as generator

        model = generator.AdaptiveSparseAnchorCompetitiveModelV4(
            **generator.MODEL_CONFIG
        )
        payload = torch.load(
            ASCA_CHECKPOINT, map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["model"], strict=True)
        model = model.to(device).eval()
        solver = generator.build_inference_solver(model, device)
        adapter = generator.ASCAV4ScoreAdapter(solver).eval()
        return "ASCA-AD V4", 100, model, adapter.window_scores, ASCA_CHECKPOINT

    if model_name == "pplad":
        baseline_root = ROOT / "BaselineModels" / "PPLAD-main"
        sys.path.insert(0, str(baseline_root))
        import solver as official_solver  # type: ignore

        payload = torch.load(
            PPLAD_CHECKPOINT, map_location="cpu", weights_only=False
        )
        solver = build_official_solver(official_solver, payload["config"], device)
        solver.model.load_state_dict(payload["model"], strict=True)
        solver.model.eval()
        adapter = PPLADScoreAdapter(solver, official_solver).eval()
        return "PPLAD", 105, solver.model, adapter.window_scores, PPLAD_CHECKPOINT

    baseline_root = ROOT / "BaselineModels" / "LTFAD-main"
    sys.path.insert(0, str(baseline_root))
    import solver as official_solver  # type: ignore

    payload = torch.load(
        LTFAD_CHECKPOINT, map_location="cpu", weights_only=False
    )
    solver = build_official_solver(official_solver, payload["config"], device)
    solver.model.load_state_dict(payload["model"], strict=True)
    solver.model.eval()
    adapter = LTFADScoreAdapter(solver, official_solver).eval()
    return "LTFAD", 90, solver.model, adapter.window_scores, LTFAD_CHECKPOINT


def configure_protocol() -> None:
    protocol.ROOT = ROOT
    protocol.OUTPUT_ROOT = OUTPUT_ROOT
    protocol.load_scaled_test = load_scaled_smd_test
    protocol.load_model = load_smd_model


def run_worker(worker: str) -> None:
    checkpoint = CHECKPOINTS[worker]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash_before = sha256(checkpoint)
    protocol.run_worker(worker)
    checkpoint_hash_after = sha256(checkpoint)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError(f"{worker} checkpoint changed during benchmark")

    result_path = OUTPUT_ROOT / MODEL_DIRS[worker] / "efficiency.json"
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


def aggregate(overwrite: bool = False) -> None:
    comparison_path = OUTPUT_ROOT / "comparison_efficiency.json"
    if comparison_path.exists() and not overwrite:
        raise FileExistsError(
            f"SMD efficiency result already exists: {comparison_path}. "
            "Use OVERWRITE=1 only for an explicitly authorized rerun."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for worker in ("asca", "pplad", "ltfad"):
        subprocess.run(
            [sys.executable, str(script), "--worker", worker],
            cwd=ROOT,
            check=True,
        )

    result_paths = [
        OUTPUT_ROOT / MODEL_DIRS[name] / "efficiency.json"
        for name in ("asca", "pplad", "ltfad")
    ]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
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
        for item in rows:
            writer.writerow({
                "Model": item["model"],
                "Parameters": item["trainable_parameters"],
                "State Dict (KiB)": item["state_dict"]["kib"],
                "Latency B=1 (ms)": item["latency_batch_1_ms"]["mean"],
                "Latency B=128 (ms)": item["latency_batch_128_ms"]["mean"],
                "Full Test Time (s)": item["full_test"]["mean"],
                "Throughput (points/s)":
                    item["full_test"]["throughput_points_per_second"],
                "GPU Peak (MiB)": item["gpu_memory"]["peak_allocated_mib"],
                "GPU Incremental (MiB)":
                    item["gpu_memory"]["incremental_allocated_mib"],
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
        "dataset": "SMD",
        "metric_scope": "inference efficiency only",
        "protocol": audit,
        "models": rows,
    })
    first = rows[0]
    write_json(OUTPUT_ROOT / "protocol_efficiency.json", {
        "dataset": "SMD",
        "gpu": first["environment"]["gpu"],
        "device": first["device"],
        "torch_version": first["torch_version"],
        "cuda_version": first["cuda_version"],
        "dtype": "float32",
        "input_dimension": 38,
        "window_sizes": {
            item["model"]: item["window_size"] for item in rows
        },
        "input_shapes": {
            item["model"]: item["input_shape"] for item in rows
        },
        "processed_points": {
            item["model"]: item["full_test"]["points"] for item in rows
        },
        **audit,
    })


def main() -> None:
    args = parse_args()
    configure_protocol()
    if args.worker:
        run_worker(args.worker)
    else:
        aggregate(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
