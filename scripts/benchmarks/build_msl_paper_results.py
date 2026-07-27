#!/usr/bin/env python3
"""MSL paper detection results using existing model-specific test paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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
OUTPUT_ROOT = ROOT / "results" / "MSL_PAPER_RESULTS"
DETECTION_ROOT = OUTPUT_ROOT / "Detection"
ASCA_CHECKPOINT = (
    ROOT / "checkpoints" / "MSL" /
    "MSL_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
    "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
)
SEED = 42
ANOMALY_RATIO = 0.83
PERCENTILE = 100.0 - ANOMALY_RATIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("asca", "pplad", "ltfad"))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="reuse existing PPLAD/LTFAD checkpoints and skip training",
    )
    parser.add_argument(
        "--output-name",
        default="Detection",
        help="subdirectory below results/MSL_PAPER_RESULTS",
    )
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def raw_metrics(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def repair_asca_raw_metrics_from_saved_score() -> None:
    """Rebuild RAW predictions before PA mutates the official prediction array."""
    model_dir = DETECTION_ROOT / "ASCA"
    metrics_path = model_dir / "metrics.json"
    prefix = model_dir / "MSL_adaptive_anchor_v4_total"
    score_path = Path(str(prefix) + "_score.txt")
    label_path = Path(str(prefix) + "_label.txt")
    if not metrics_path.is_file() or not score_path.is_file() or not label_path.is_file():
        return

    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_energy = np.loadtxt(score_path, dtype=np.float64).reshape(-1)
    labels = np.loadtxt(label_path, dtype=np.int64).reshape(-1)
    if test_energy.size != labels.size:
        raise RuntimeError(
            f"ASCA saved score/label length mismatch: {test_energy.size} != {labels.size}"
        )
    raw_prediction = (test_energy > float(result["threshold"])).astype(np.int64)
    result["raw"] = raw_metrics(raw_prediction, labels)
    result["raw_prediction_source"] = "saved_test_energy > saved_threshold (before PA)"
    np.save(model_dir / "pred_raw.npy", raw_prediction, allow_pickle=False)
    np.savetxt(str(prefix) + "_pred_raw.txt", raw_prediction, fmt="%d")
    write_json(metrics_path, result)


def run_asca() -> None:
    runner_module = load_module(
        "msl_paper_runner",
        ROOT / "scripts" / "dataset_runners" / "run_msl_official_vs_v4_best_v3.py",
    )
    set_seed()
    channels = runner_module.verify_data(ROOT)
    _solver, solver_v4 = runner_module.bootstrap_project(ROOT)
    config = runner_module.v4_config(ROOT, channels, SEED, 10)
    config.update(
        {
            "win_size": 100,
            "batch_size": 128,
            "num_epochs": 10,
            "lr": 1e-3,
            "anormly_ratio": ANOMALY_RATIO,
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
            "gap_weight": 1.0,
            "relation_input": "instance",
            "score_modes": ["total"],
            "primary_score": "total",
            "score_normalization": "official",
            "threshold_source": "original",
            "quantile_method": "exact",
            "model_save_path": str(ASCA_CHECKPOINT.parent),
            "result_path": str(DETECTION_ROOT / "ASCA"),
        }
    )
    model_dir = DETECTION_ROOT / "ASCA"
    model_dir.mkdir(parents=True, exist_ok=True)
    runner = solver_v4(config)
    payload = torch.load(ASCA_CHECKPOINT, map_location="cpu", weights_only=False)
    runner.model.load_state_dict(payload["model"], strict=True)
    captured: dict[str, Any] = {}
    original_thresholds = runner._stream_thresholds

    def capture_thresholds(loaders):
        values = original_thresholds(loaders)
        captured["threshold"] = float(values["total"])
        captured["threshold_samples"] = int(runner._threshold_sample_counts["total"])
        return values

    runner._stream_thresholds = capture_thresholds
    pa_values = runner.test()
    prefix = model_dir / "MSL_adaptive_anchor_v4_total"
    test_energy = np.loadtxt(str(prefix) + "_score.txt", dtype=np.float64)
    pa_prediction = np.loadtxt(str(prefix) + "_pred_pa.txt", dtype=np.int64)
    labels = np.loadtxt(str(prefix) + "_label.txt", dtype=np.int64)
    raw_prediction = (test_energy > captured["threshold"]).astype(np.int64)
    np.save(model_dir / "pred_raw.npy", raw_prediction, allow_pickle=False)
    np.savetxt(str(prefix) + "_pred_raw.txt", raw_prediction, fmt="%d")
    raw = raw_metrics(raw_prediction, labels)
    pa = raw_metrics(pa_prediction, labels)
    if not np.allclose(list(pa.values()), [pa_values[0], pa_values[1], pa_values[2], pa_values[3]]):
        raise RuntimeError("ASCA returned PA metrics do not match saved official predictions")
    result = {
        "model": "ASCA-AD V4",
        "parameters": int(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)),
        "seed": SEED,
        "checkpoint": str(ASCA_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(ASCA_CHECKPOINT),
        "configuration": config,
        "score_mode": "total",
        "threshold": captured["threshold"],
        "threshold_samples": captured["threshold_samples"],
        "threshold_rule": f"percentile(concat(train_energy,test_energy),{PERCENTILE:g})",
        "threshold_search": False,
        "evaluated_points": int(labels.size),
        "raw": raw,
        "pa": pa,
        "raw_prediction_source": "saved_test_energy > threshold (before PA)",
    }
    write_json(model_dir / "metrics.json", result)


def baseline_config(model: str) -> dict[str, Any]:
    common = {
        "index": 137,
        "dataset": "MSL",
        "data_path": "MSL",
        "input_c": 55,
        "output_c": 55,
        "d_model": 128,
        "lr": 1e-4,
        "loss_fuc": "MSE",
        "anormly_ratio": ANOMALY_RATIO,
        "model_save_path": str(DETECTION_ROOT / model.upper()),
    }
    if model == "pplad":
        return {
            **common,
            "win_size": 90,
            "batch_size": 256,
            "num_epochs": 3,
            "local_size": [7],
            "global_size": [30],
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


def run_baseline(model: str, evaluate_existing: bool = False) -> None:
    baseline_root = ROOT / "BaselineModels" / ("PPLAD-main" if model == "pplad" else "LTFAD-main")
    sys.path.insert(0, str(baseline_root))
    os.chdir(ROOT)
    set_seed()
    import solver as official_solver  # type: ignore

    config = baseline_config(model)
    runner = official_solver.Solver(config)
    name = "PPLAD" if model == "pplad" else "LTFAD"
    model_dir = DETECTION_ROOT / name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / f"{name}_official_state_dict.pt"
    if evaluate_existing:
        source_checkpoint = (
            OUTPUT_ROOT / "Detection" / name / f"{name}_official_state_dict.pt"
        )
        if not source_checkpoint.is_file():
            raise FileNotFoundError(
                f"Existing {name} checkpoint not found: {source_checkpoint}"
            )
        payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        runner.model.load_state_dict(payload["model"], strict=True)
        checkpoint = source_checkpoint
    captured: dict[str, Any] = {}
    original_combiner = official_solver.combine_all_evaluation_scores
    original_percentile = official_solver.np.percentile

    def capture_combiner(pred, gt, energy):
        raw_prediction = np.asarray(pred).astype(np.int64).reshape(-1).copy()
        labels = np.asarray(gt).astype(np.int64).reshape(-1).copy()
        captured["raw_prediction"] = raw_prediction
        captured["labels"] = labels
        captured["raw"] = raw_metrics(raw_prediction, labels)
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
    if evaluate_existing:
        pa_values = runner.test()
    elif model == "pplad":
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
    official_solver.np.percentile = original_percentile

    if not evaluate_existing:
        torch.save({"model": runner.model.state_dict(), "config": config}, checkpoint)
    np.save(model_dir / "pred_raw.npy", captured["raw_prediction"], allow_pickle=False)
    np.save(model_dir / "pred_pa.npy", captured["pa_prediction"], allow_pickle=False)
    pa = {
        key: float(value)
        for key, value in zip(("accuracy", "precision", "recall", "f1"), pa_values)
    }
    result = {
        "model": name,
        "parameters": int(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)),
        "seed": SEED,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "configuration": config,
        "configuration_source": (
            "BaselineModels/PPLAD-main/scripts/MSL.sh (official MSL anomaly_ratio=0.83)"
            if model == "pplad"
            else "pre-registered fixed non-official MSL model configuration; "
            "evaluation anomaly_ratio shared at 0.83"
        ),
        "evaluation_mode": "existing_checkpoint" if evaluate_existing else "train_then_evaluate",
        "threshold": captured["threshold"],
        "threshold_samples": captured["threshold_samples"],
        "threshold_rule": f"percentile(concat(train_energy,test_energy),{PERCENTILE:g})",
        "threshold_search": False,
        "evaluated_points": int(captured["labels"].size),
        "raw": captured["raw"],
        "pa": pa,
        "official_extended_metrics": captured["extended"],
    }
    write_json(model_dir / "metrics.json", result)


def aggregate(run_workers: bool = True, evaluate_existing: bool = False) -> None:
    DETECTION_ROOT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    if run_workers:
        for worker in ("asca", "pplad", "ltfad"):
            command = [
                sys.executable,
                str(script),
                "--worker",
                worker,
                "--output-name",
                DETECTION_ROOT.name,
            ]
            if evaluate_existing:
                command.append("--evaluate-existing")
            subprocess.run(command, check=True)
    repair_asca_raw_metrics_from_saved_score()
    paths = (
        DETECTION_ROOT / "ASCA" / "metrics.json",
        DETECTION_ROOT / "PPLAD" / "metrics.json",
        DETECTION_ROOT / "LTFAD" / "metrics.json",
    )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    fields = [
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "PA-Accuracy", "PA-Precision", "PA-Recall", "PA-F1", "Threshold",
    ]
    with (DETECTION_ROOT / "comparison_detection.csv").open("w", newline="", encoding="utf-8") as stream:
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
    write_json(
        DETECTION_ROOT / "comparison_detection.json",
        {
            "dataset": "MSL",
            "anomaly_ratio": ANOMALY_RATIO,
            "threshold_rule": (
                f"per-model percentile(concat(train_energy,test_energy),{PERCENTILE:g})"
            ),
            "parameter_search": False,
            "results": rows,
        },
    )


def main() -> None:
    global DETECTION_ROOT
    args = parse_args()
    output_name = Path(args.output_name)
    if output_name.name != args.output_name or args.output_name in {"", ".", ".."}:
        raise ValueError("--output-name must be a single directory name")
    DETECTION_ROOT = OUTPUT_ROOT / output_name
    if args.worker == "asca":
        run_asca()
    elif args.worker in {"pplad", "ltfad"}:
        run_baseline(args.worker, evaluate_existing=args.evaluate_existing)
    else:
        aggregate(
            run_workers=not args.aggregate_only,
            evaluate_existing=args.evaluate_existing,
        )


if __name__ == "__main__":
    main()
