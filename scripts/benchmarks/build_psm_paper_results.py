#!/usr/bin/env python3
"""Frozen PSM detection experiment for ASCA-AD V4, PPLAD, and LTFAD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENTRY_SCRIPT = Path(__file__).resolve()

OUTPUT_ROOT = ROOT / "results" / "PSM_PAPER_RESULTS"
DETECTION_ROOT = OUTPUT_ROOT / "Detection"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
DATASET_NAME = "PSM"
PPLAD_CONFIG_SOURCE = "BaselineModels/PPLAD-main/scripts/PSM.sh"
LTFAD_CONFIG_SOURCE = (
    "pre-registered fixed PSM configuration; official configuration unavailable"
)
ASCA_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "PSM"
    / "PSM_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
    "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
)

SEED = 42
ANOMALY_RATIO = 0.8
PERCENTILE = 100.0 - ANOMALY_RATIO
EXPECTED_SHAPES = {
    "train": (132481, 25),
    "test": (87841, 25),
    "label": (87841,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("asca", "pplad", "ltfad"))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def metrics(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def validate_dataset() -> None:
    dataset = ROOT / "dataset" / DATASET_NAME
    paths = {
        "train": dataset / f"{DATASET_NAME}_train.npy",
        "test": dataset / f"{DATASET_NAME}_test.npy",
        "label": dataset / f"{DATASET_NAME}_test_label.npy",
    }
    for split, path in paths.items():
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(values.shape) != EXPECTED_SHAPES[split]:
            raise RuntimeError(
                f"Unexpected {DATASET_NAME} {split} shape {values.shape}; "
                f"expected {EXPECTED_SHAPES[split]}"
            )


def asca_config() -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "data_path": DATASET_NAME,
        "input_c": 25,
        "output_c": 25,
        "win_size": 100,
        "batch_size": 128,
        "num_epochs": 10,
        "lr": 1e-3,
        "anormly_ratio": ANOMALY_RATIO,
        "index": 137,
        "mode": "train",
        "seed": SEED,
        "local_candidate_lags": [1, 2, 3, 4, 5, 6, 7, 8],
        "global_candidate_lags": [12, 16, 20, 24, 28, 32, 40, 48],
        "local_topk": 2,
        "global_topk": 4,
        "selector_hidden": 8,
        "fitter_hidden": 8,
        "selector_temperature": 0.5,
        "similarity_tau": 1.0,
        "sigma_min": 0.03,
        "sigma_max": 1.5,
        "area_weight": 0.1,
        "selector_balance_weight": 0.05,
        "gap_weight": 1.0,
        "relation_input": "instance",
        "score_modes": ["total"],
        "primary_score": "total",
        "score_normalization": "official",
        "threshold_source": "original",
        "quantile_method": "exact",
        "quantile_buffer": 50000,
        "local_size": 1,
        "global_size": [20],
        "d_model": 8,
        "loss_fuc": "MSE",
        "r": 0.5,
        "similar": "MSE",
        "rec_timeseries": True,
        "model_save_path": str(ASCA_CHECKPOINT.parent),
        "result_path": str(DETECTION_ROOT / "ASCA"),
        "use_gpu": torch.cuda.is_available(),
        "use_multi_gpu": False,
        "gpu": 0,
        "devices": "0",
    }


def run_asca() -> None:
    validate_dataset()
    set_seed()
    from asca_ad.model import AdaptiveSparseAnchorSolverV4

    config = asca_config()
    runner = AdaptiveSparseAnchorSolverV4(config)
    checkpoint_payload = torch.load(
        ASCA_CHECKPOINT, map_location="cpu", weights_only=False
    )
    runner.model.load_state_dict(checkpoint_payload["model"], strict=True)
    runner.model.eval()

    captured: dict[str, Any] = {}
    original_thresholds = runner._stream_thresholds
    test_globals = inspect.unwrap(runner.test).__globals__
    original_combiner = test_globals["combine_all_evaluation_scores"]

    def capture_thresholds(loaders):
        values = original_thresholds(loaders)
        captured["threshold"] = float(values["total"])
        captured["threshold_samples"] = int(
            runner._threshold_sample_counts["total"]
        )
        return values

    def capture_combiner(pred, gt, energy):
        captured["raw_prediction"] = np.asarray(pred).astype(np.int64).reshape(-1).copy()
        captured["labels"] = np.asarray(gt).astype(np.int64).reshape(-1).copy()
        return original_combiner(pred, gt, energy)

    runner._stream_thresholds = capture_thresholds
    test_globals["combine_all_evaluation_scores"] = capture_combiner
    try:
        pa_values = runner.test()
    finally:
        test_globals["combine_all_evaluation_scores"] = original_combiner

    model_dir = DETECTION_ROOT / "ASCA"
    prefix = model_dir / f"{DATASET_NAME}_adaptive_anchor_v4_total"
    pa_prediction = np.loadtxt(str(prefix) + "_pred_pa.txt", dtype=np.int64)
    labels = np.loadtxt(str(prefix) + "_label.txt", dtype=np.int64)
    raw_prediction = captured["raw_prediction"]
    if not np.array_equal(labels.reshape(-1), captured["labels"]):
        raise RuntimeError("ASCA captured labels do not match saved labels")
    raw = metrics(raw_prediction, labels)
    pa = metrics(pa_prediction, labels)
    returned_pa = dict(
        zip(("accuracy", "precision", "recall", "f1"), map(float, pa_values))
    )
    if not np.allclose(list(pa.values()), list(returned_pa.values())):
        raise RuntimeError("ASCA PA metrics do not match the saved PA prediction")

    np.save(model_dir / "pred_raw.npy", raw_prediction, allow_pickle=False)
    np.save(model_dir / "pred_pa.npy", pa_prediction, allow_pickle=False)
    result = {
        "model": "ASCA-AD V4",
        "parameters": int(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)),
        "seed": SEED,
        "checkpoint": str(ASCA_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(ASCA_CHECKPOINT),
        "configuration": config,
        "training_performed": False,
        "score_mode": "total",
        "threshold": captured["threshold"],
        "threshold_samples": captured["threshold_samples"],
        "threshold_rule": f"percentile(concat(train_energy,test_energy),{PERCENTILE:g})",
        "threshold_search": False,
        "evaluated_points": int(labels.size),
        "raw": raw,
        "pa": pa,
        "pa_implementation": "asca_ad.model PPLAD-compatible point adjustment",
    }
    write_json(model_dir / "metrics.json", result)


def baseline_config(model: str) -> dict[str, Any]:
    name = "PPLAD" if model == "pplad" else "LTFAD"
    common = {
        "index": 137,
        "dataset": DATASET_NAME,
        "data_path": DATASET_NAME,
        "input_c": 25,
        "output_c": 25,
        "d_model": 128,
        "lr": 1e-4,
        "loss_fuc": "MSE",
        "anormly_ratio": ANOMALY_RATIO,
        "model_save_path": str(DETECTION_ROOT / name),
    }
    if model == "pplad":
        return {
            **common,
            "win_size": 60,
            "batch_size": 256,
            "num_epochs": 1,
            "local_size": [1],
            "global_size": [20],
            "r": 0.5,
            "similar": "MSE",
        }
    return {
        **common,
        "win_size": 90,
        "batch_size": 128,
        "num_epochs": 2,
        "local_size": [5],
        "global_size": [13],
        "r": 0.1,
    }


def run_baseline(model: str) -> None:
    validate_dataset()
    set_seed()
    baseline_root = ROOT / "BaselineModels" / (
        "PPLAD-main" if model == "pplad" else "LTFAD-main"
    )
    sys.path.insert(0, str(baseline_root))
    os.chdir(ROOT)
    official_solver = importlib.import_module("solver")

    config = baseline_config(model)
    runner = official_solver.Solver(config)
    captured: dict[str, Any] = {}
    original_combiner = official_solver.combine_all_evaluation_scores
    original_percentile = official_solver.np.percentile

    def capture_combiner(pred, gt, energy):
        raw_prediction = np.asarray(pred).astype(np.int64).reshape(-1).copy()
        labels = np.asarray(gt).astype(np.int64).reshape(-1).copy()
        captured["raw_prediction"] = raw_prediction
        captured["labels"] = labels
        captured["raw"] = metrics(raw_prediction, labels)
        values = original_combiner(pred, gt, energy)
        captured["pa_prediction"] = np.asarray(pred).astype(np.int64).reshape(-1).copy()
        captured["extended"] = {key: float(value) for key, value in values.items()}
        return values

    def capture_percentile(values, percentile, *args, **kwargs):
        result = original_percentile(values, percentile, *args, **kwargs)
        if "threshold" not in captured and np.isclose(float(percentile), PERCENTILE):
            captured["threshold"] = float(result)
            captured["threshold_samples"] = int(np.asarray(values).size)
        return result

    official_solver.combine_all_evaluation_scores = capture_combiner
    official_solver.np.percentile = capture_percentile
    pa_holder: dict[str, Any] = {}
    try:
        if model == "pplad":
            runner.train()
            pa_values = runner.test()
        else:
            original_test = runner.test

            def capture_test():
                values = original_test()
                pa_holder["values"] = values
                return values

            runner.test = capture_test
            runner.run()
            pa_values = pa_holder["values"]
    finally:
        official_solver.combine_all_evaluation_scores = original_combiner
        official_solver.np.percentile = original_percentile

    name = "PPLAD" if model == "pplad" else "LTFAD"
    model_dir = DETECTION_ROOT / name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / f"{name}_official_state_dict.pt"
    torch.save({"model": runner.model.state_dict(), "config": config}, checkpoint)
    np.save(model_dir / "pred_raw.npy", captured["raw_prediction"], allow_pickle=False)
    np.save(model_dir / "pred_pa.npy", captured["pa_prediction"], allow_pickle=False)
    pa = dict(zip(("accuracy", "precision", "recall", "f1"), map(float, pa_values)))
    result = {
        "model": name,
        "parameters": int(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)),
        "seed": SEED,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "configuration": config,
        "configuration_source": (
            PPLAD_CONFIG_SOURCE if model == "pplad" else LTFAD_CONFIG_SOURCE
        ),
        "training_performed": True,
        "threshold": captured["threshold"],
        "threshold_samples": captured["threshold_samples"],
        "threshold_rule": f"percentile(concat(train_energy,test_energy),{PERCENTILE:g})",
        "threshold_search": False,
        "evaluated_points": int(captured["labels"].size),
        "raw": captured["raw"],
        "pa": pa,
        "official_extended_metrics": captured["extended"],
        "pa_implementation": f"BaselineModels/{name}-main/solver.py official test path",
    }
    write_json(model_dir / "metrics.json", result)


def aggregate(run_workers: bool = True, overwrite: bool = False) -> None:
    comparison_path = DETECTION_ROOT / "comparison_detection.json"
    if run_workers and comparison_path.exists() and not overwrite:
        raise FileExistsError(
            f"Formal {DATASET_NAME} result already exists: {comparison_path}. "
            "Use --overwrite only for an explicitly authorized rerun."
        )
    DETECTION_ROOT.mkdir(parents=True, exist_ok=True)
    script = ENTRY_SCRIPT
    if run_workers:
        for worker in ("asca", "pplad", "ltfad"):
            command = [sys.executable, str(script), "--worker", worker]
            if overwrite:
                command.append("--overwrite")
            subprocess.run(command, check=True)

    paths = [
        DETECTION_ROOT / "ASCA" / "metrics.json",
        DETECTION_ROOT / "PPLAD" / "metrics.json",
        DETECTION_ROOT / "LTFAD" / "metrics.json",
    ]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    fields = [
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "PA-Accuracy", "PA-Precision", "PA-Recall", "PA-F1", "Threshold",
    ]
    with (DETECTION_ROOT / "comparison_detection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "Model": item["model"],
                    "Accuracy": item["raw"]["accuracy"],
                    "Precision": item["raw"]["precision"],
                    "Recall": item["raw"]["recall"],
                    "F1": item["raw"]["f1"],
                    "PA-Accuracy": item["pa"]["accuracy"],
                    "PA-Precision": item["pa"]["precision"],
                    "PA-Recall": item["pa"]["recall"],
                    "PA-F1": item["pa"]["f1"],
                    "Threshold": item["threshold"],
                }
            )

    comparison = {
        "dataset": DATASET_NAME,
        "anomaly_ratio": ANOMALY_RATIO,
        "threshold_rule": (
            f"per-model percentile(concat(train_energy,test_energy),{PERCENTILE:g})"
        ),
        "score_or_ratio_search": False,
        "test_labels_used_for_parameter_selection": False,
        "results": rows,
    }
    write_json(comparison_path, comparison)

    dataset_root = ROOT / "dataset" / DATASET_NAME
    protocol = {
        "dataset": {
            "name": DATASET_NAME,
            "train": {
                "path": f"dataset/{DATASET_NAME}/{DATASET_NAME}_train.npy",
                "shape": list(EXPECTED_SHAPES["train"]),
                "sha256": sha256(dataset_root / f"{DATASET_NAME}_train.npy"),
            },
            "test": {
                "path": f"dataset/{DATASET_NAME}/{DATASET_NAME}_test.npy",
                "shape": list(EXPECTED_SHAPES["test"]),
                "sha256": sha256(dataset_root / f"{DATASET_NAME}_test.npy"),
            },
            "label": {
                "path": f"dataset/{DATASET_NAME}/{DATASET_NAME}_test_label.npy",
                "shape": list(EXPECTED_SHAPES["label"]),
                "sha256": sha256(
                    dataset_root / f"{DATASET_NAME}_test_label.npy"
                ),
            },
        },
        "seed": SEED,
        "anomaly_ratio": ANOMALY_RATIO,
        "percentile": PERCENTILE,
        "threshold_rule": "each model independently uses train_energy + test_energy",
        "threshold_search": False,
        "test_label_parameter_selection": False,
        "pa_implementation": "official PPLAD-compatible segment point adjustment",
        "models": {
            item["model"]: {
                "configuration": item["configuration"],
                "checkpoint": item["checkpoint"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "score_mode": item.get("score_mode"),
            }
            for item in rows
        },
    }
    write_json(PROTOCOL_PATH, protocol)


def main() -> None:
    args = parse_args()
    if args.worker == "asca":
        run_asca()
    elif args.worker in {"pplad", "ltfad"}:
        run_baseline(args.worker)
    else:
        aggregate(run_workers=not args.aggregate_only, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
