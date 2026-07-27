#!/usr/bin/env python3
"""Inference-only PSM efficiency benchmark for the three formal models."""

from __future__ import annotations

import argparse
import csv
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


OUTPUT_ROOT = ROOT / "results" / "PSM_PAPER_RESULTS" / "Efficiency"
ASCA_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "PSM"
    / "PSM_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
    "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
)
PPLAD_CHECKPOINT = (
    ROOT
    / "results"
    / "PSM_PAPER_RESULTS"
    / "Detection"
    / "PPLAD"
    / "PPLAD_official_state_dict.pt"
)
LTFAD_CHECKPOINT = (
    ROOT
    / "results"
    / "PSM_PAPER_RESULTS"
    / "Detection"
    / "LTFAD"
    / "LTFAD_official_state_dict.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("asca", "pplad", "ltfad"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scaled_psm_test() -> np.ndarray:
    dataset = ROOT / "dataset" / "PSM"
    train = np.asarray(
        np.load(dataset / "PSM_train.npy", allow_pickle=False), dtype=np.float32
    )
    test = np.asarray(
        np.load(dataset / "PSM_test.npy", allow_pickle=False), dtype=np.float32
    )
    if train.shape != (132481, 25) or test.shape != (87841, 25):
        raise RuntimeError(f"Unexpected PSM shapes: train={train.shape}, test={test.shape}")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError("PSM train/test contains NaN or Inf")
    scaler = StandardScaler()
    scaler.fit(train)
    return scaler.transform(test).astype(np.float32, copy=False)


def build_official_solver(official_solver, config: dict, device: torch.device):
    solver = object.__new__(official_solver.Solver)
    solver.__dict__.update(config)
    solver.device = device
    solver.build_model()
    solver.model = solver.model.to(device)
    solver.criterion = nn.MSELoss()
    solver.criterion_keep = nn.MSELoss(reduction="none")
    return solver


def load_psm_model(model_name: str, device: torch.device):
    if model_name == "asca":
        import scripts.benchmarks.generate_asca_v4_skab_score as generator

        model = generator.AdaptiveSparseAnchorCompetitiveModelV4(
            **generator.MODEL_CONFIG
        )
        payload = torch.load(ASCA_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model = model.to(device).eval()
        solver = generator.build_inference_solver(model, device)
        adapter = generator.ASCAV4ScoreAdapter(solver).eval()
        return "ASCA-AD V4", 100, model, adapter.window_scores, ASCA_CHECKPOINT

    if model_name == "pplad":
        baseline_root = ROOT / "BaselineModels" / "PPLAD-main"
        sys.path.insert(0, str(baseline_root))
        import solver as official_solver  # type: ignore

        payload = torch.load(PPLAD_CHECKPOINT, map_location="cpu", weights_only=False)
        solver = build_official_solver(official_solver, payload["config"], device)
        solver.model.load_state_dict(payload["model"], strict=True)
        solver.model.eval()
        adapter = PPLADScoreAdapter(solver, official_solver).eval()
        return "PPLAD", 60, solver.model, adapter.window_scores, PPLAD_CHECKPOINT

    baseline_root = ROOT / "BaselineModels" / "LTFAD-main"
    sys.path.insert(0, str(baseline_root))
    import solver as official_solver  # type: ignore

    payload = torch.load(LTFAD_CHECKPOINT, map_location="cpu", weights_only=False)
    solver = build_official_solver(official_solver, payload["config"], device)
    solver.model.load_state_dict(payload["model"], strict=True)
    solver.model.eval()
    adapter = LTFADScoreAdapter(solver, official_solver).eval()
    return "LTFAD", 90, solver.model, adapter.window_scores, LTFAD_CHECKPOINT


def configure_protocol() -> None:
    protocol.ROOT = ROOT
    protocol.OUTPUT_ROOT = OUTPUT_ROOT
    protocol.load_scaled_test = load_scaled_psm_test
    protocol.load_model = load_psm_model


def aggregate(overwrite: bool = False) -> None:
    comparison_path = OUTPUT_ROOT / "comparison_efficiency.json"
    if comparison_path.exists() and not overwrite:
        raise FileExistsError(
            f"Formal PSM efficiency result already exists: {comparison_path}. "
            "Use --overwrite only for an explicitly authorized rerun."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for worker in ("asca", "pplad", "ltfad"):
        subprocess.run(
            [sys.executable, str(script), "--worker", worker], check=True
        )

    paths = (
        OUTPUT_ROOT / "ASCA-AD_V4" / "efficiency.json",
        OUTPUT_ROOT / "PPLAD" / "efficiency.json",
        OUTPUT_ROOT / "LTFAD" / "efficiency.json",
    )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    fields = [
        "Model",
        "Parameters",
        "State Dict(KiB)",
        "Latency_B1(ms)",
        "Latency_B128(ms)",
        "Full_Test_Time(s)",
        "Throughput(points/s)",
        "GPU_Peak(MiB)",
        "GPU_Incremental(MiB)",
    ]
    with (OUTPUT_ROOT / "comparison_efficiency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "Model": item["model"],
                    "Parameters": item["parameters"],
                    "State Dict(KiB)": item["state_dict"]["kib"],
                    "Latency_B1(ms)": item["latency_batch_1_ms"]["mean"],
                    "Latency_B128(ms)": item["latency_batch_128_ms"]["mean"],
                    "Full_Test_Time(s)": item["full_test"]["mean"],
                    "Throughput(points/s)": item["full_test"][
                        "throughput_points_per_second"
                    ],
                    "GPU_Peak(MiB)": item["gpu_memory"]["peak_allocated_mib"],
                    "GPU_Incremental(MiB)": item["gpu_memory"][
                        "incremental_allocated_mib"
                    ],
                }
            )

    comparison = {
        "dataset": "PSM",
        "detection_protocol_anomaly_ratio": 0.8,
        "threshold_independent": True,
        "metric_scope": "inference efficiency only",
        "protocol": {
            "torch_no_grad": True,
            "model_eval": True,
            "cuda_event": True,
            "warmup": protocol.WARMUP,
            "latency_repeat": protocol.REPEATS,
            "full_test_repeat": protocol.FULL_TEST_REPEATS,
            "uses_label": False,
            "calls_evaluator": False,
            "includes_data_loading": False,
            "includes_host_to_device_transfer": False,
        },
        "models": rows,
    }
    protocol.write_json(comparison_path, comparison)

    first = rows[0]
    protocol.write_json(
        OUTPUT_ROOT / "protocol_efficiency.json",
        {
            "dataset": "PSM",
            "gpu": first["environment"]["gpu"],
            "device": first["device"],
            "torch_version": first["torch_version"],
            "cuda_version": first["cuda_version"],
            "input_dimension": 25,
            "window_sizes": {
                item["model"]: item["window_size"] for item in rows
            },
            "input_shapes": {
                item["model"]: item["input_shape"] for item in rows
            },
            "warmup": protocol.WARMUP,
            "repeat": protocol.REPEATS,
            "full_test_repeat": protocol.FULL_TEST_REPEATS,
            "includes_data_loading": False,
            "includes_host_to_device_transfer": False,
            "uses_label": False,
            "calls_evaluator": False,
            "training_performed": False,
        },
    )


def main() -> None:
    args = parse_args()
    configure_protocol()
    if args.worker:
        protocol.run_worker(args.worker)
    else:
        aggregate(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
