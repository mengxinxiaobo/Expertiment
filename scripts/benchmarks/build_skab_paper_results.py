#!/usr/bin/env python3
"""Build the SKAB paper table from each model's own evaluation path.

This script deliberately does not create a unified score timeline.  ASCA uses
the project's fixed-combined evaluator, while PPLAD and LTFAD call their
unmodified official Solver.test() implementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "results" / "SKAB_PAPER_RESULTS"
ASCA_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "SKAB"
    / "SKAB_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
    "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
)
PPLAD_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "SKAB_OFFICIAL_DEFAULT_VS_V4_BEST"
    / "ORIGINAL"
    / "SKAB_original_official_default_state.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("asca", "pplad", "ltfad"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


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


def run_asca() -> dict[str, Any]:
    if not ASCA_CHECKPOINT.is_file():
        raise FileNotFoundError(ASCA_CHECKPOINT)
    engine = load_module(
        "skab_paper_fixed_combined_engine", ROOT / "scripts" / "run_all_fixed_combined.py"
    )
    runner_module = load_module(
        "skab_paper_asca_runner",
        ROOT / "scripts" / "dataset_runners" / "run_skab_official_default_vs_v4_best.py",
    )
    set_seed(42)
    channels = runner_module.verify_data(ROOT)
    _solver, solver_v4 = runner_module.bootstrap_project(ROOT)
    config = runner_module.v4_config(ROOT, channels, 42, 10)
    runner = solver_v4(config)
    engine.load_checkpoint(runner, ASCA_CHECKPOINT)

    train_scores, _ = runner_module.collect_v4_scores(
        runner, runner.train_loader, include_labels=False
    )
    test_scores, labels = runner_module.collect_v4_scores(
        runner, runner.thre_loader, include_labels=True
    )
    if labels is None:
        raise RuntimeError("ASCA official evaluation returned no labels")
    train_energy = np.asarray(train_scores["combined"]).reshape(-1)
    test_energy = np.asarray(test_scores["combined"]).reshape(-1)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    threshold = float(
        np.percentile(np.concatenate([train_energy, test_energy]), 99.5)
    )
    arrays = engine.local_threshold_metrics(
        test_energy, labels, np.asarray([threshold], dtype=np.float64)
    )
    metric = {key: engine.scalar(arrays, key) for key in arrays}
    result = {
        "model": "ASCA-AD V4",
        "parameters": 146,
        "seed": 42,
        "checkpoint": str(ASCA_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(ASCA_CHECKPOINT),
        "configuration": {
            "window_size": 100,
            "score_mode": "combined",
            "anomaly_ratio": 0.5,
        },
        "evaluation": {
            "implementation": "asca_ad/evaluator.py -> scripts/run_all_fixed_combined.py",
            "timeline": "official non-overlapping thre_loader; no overlap aggregation",
            "evaluated_points": int(test_energy.size),
            "threshold": threshold,
            "threshold_search": False,
        },
        "raw": {
            "accuracy": float(metric["raw_accuracy"]),
            "precision": float(metric["raw_precision"]),
            "recall": float(metric["raw_recall"]),
            "f1": float(metric["raw_f1"]),
        },
        "pa": {
            "accuracy": float(metric["pa_accuracy"]),
            "precision": float(metric["pa_precision"]),
            "recall": float(metric["pa_recall"]),
            "f1": float(metric["pa_f1"]),
        },
        "official_extended_metrics": {},
        "environment": environment(),
    }
    write_json(OUTPUT_ROOT / "ASCA-AD_V4" / "metrics.json", result)
    return result


def baseline_config(model: str) -> dict[str, Any]:
    common = {
        "index": 137,
        "dataset": "SKAB",
        "data_path": "SKAB",
        "input_c": 8,
        "output_c": 8,
        "batch_size": 128,
        "lr": 1e-4,
        "d_model": 128,
        "loss_fuc": "MSE",
        "model_save_path": str(OUTPUT_ROOT / model.upper()),
    }
    if model == "pplad":
        return {
            **common,
            "win_size": 60,
            "num_epochs": 3,
            "local_size": [3],
            "global_size": [20],
            "r": 0.5,
            "similar": "MSE",
            "anormly_ratio": 0.5,
        }
    return {
        **common,
        "win_size": 90,
        "num_epochs": 2,
        "local_size": [5],
        "global_size": [13],
        "r": 0.1,
        "anormly_ratio": 0.3,
    }


def run_official_baseline(model: str) -> dict[str, Any]:
    baseline_root = ROOT / "BaselineModels" / ("PPLAD-main" if model == "pplad" else "LTFAD-main")
    sys.path.insert(0, str(baseline_root))
    os.chdir(ROOT)
    set_seed(42)
    import solver as official_solver  # type: ignore

    config = baseline_config(model)
    runner = official_solver.Solver(config)
    captured: dict[str, Any] = {}
    original_combiner = official_solver.combine_all_evaluation_scores

    def capture_official_metrics(pred, gt, energy):
        prediction = np.asarray(pred).astype(np.int64).reshape(-1).copy()
        labels = np.asarray(gt).astype(np.int64).reshape(-1).copy()
        captured["raw"] = raw_metrics(prediction, labels)
        captured["evaluated_points"] = int(labels.size)
        official_values = original_combiner(pred, gt, energy)
        captured["official_extended_metrics"] = {
            key: float(value) for key, value in official_values.items()
        }
        return official_values

    official_solver.combine_all_evaluation_scores = capture_official_metrics
    if model == "pplad":
        if not PPLAD_CHECKPOINT.is_file():
            raise FileNotFoundError(PPLAD_CHECKPOINT)
        payload = torch.load(PPLAD_CHECKPOINT, map_location=runner.device)
        runner.model.load_state_dict(payload["model"], strict=True)
        checkpoint = PPLAD_CHECKPOINT
        pa_values = runner.test()
        checkpoint_origin = "reused existing official trained checkpoint"
    else:
        pa_holder: dict[str, Any] = {}
        original_test = runner.test

        def capture_test_return():
            values = original_test()
            pa_holder["values"] = values
            return values

        runner.test = capture_test_return
        runner.run()
        pa_values = pa_holder["values"]
        checkpoint = OUTPUT_ROOT / "LTFAD" / "LTFAD_official_state_dict.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": runner.model.state_dict(), "config": config}, checkpoint)
        checkpoint_origin = "trained by unmodified official Solver.run()"

    if "raw" not in captured:
        raise RuntimeError(f"{model} official test did not call its metrics function")
    pa = {
        key: float(value)
        for key, value in zip(("accuracy", "precision", "recall", "f1"), pa_values)
    }
    name = "PPLAD" if model == "pplad" else "LTFAD"
    result = {
        "model": name,
        "parameters": int(sum(p.numel() for p in runner.model.parameters())),
        "seed": 42,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_origin": checkpoint_origin,
        "configuration": config,
        "evaluation": {
            "implementation": str((baseline_root / "solver.py").relative_to(ROOT)) + "::Solver.test",
            "implementation_sha256": sha256(baseline_root / "solver.py"),
            "timeline": "official thre_loader; no overlap aggregation",
            "evaluated_points": captured["evaluated_points"],
            "threshold_search": False,
        },
        "raw": captured["raw"],
        "pa": pa,
        "official_extended_metrics": captured["official_extended_metrics"],
        "environment": environment(),
    }
    write_json(OUTPUT_ROOT / name / "metrics.json", result)
    return result


def build_comparison() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for worker in ("asca", "pplad", "ltfad"):
        subprocess.run([sys.executable, str(script), "--worker", worker], check=True)

    paths = (
        OUTPUT_ROOT / "ASCA-AD_V4" / "metrics.json",
        OUTPUT_ROOT / "PPLAD" / "metrics.json",
        OUTPUT_ROOT / "LTFAD" / "metrics.json",
    )
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    csv_path = OUTPUT_ROOT / "comparison.csv"
    fields = [
        "Model", "Params", "Accuracy", "Precision", "Recall", "F1", "PA-F1"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "Model": item["model"],
                    "Params": item["parameters"],
                    "Accuracy": item["raw"]["accuracy"],
                    "Precision": item["raw"]["precision"],
                    "Recall": item["raw"]["recall"],
                    "F1": item["raw"]["f1"],
                    "PA-F1": item["pa"]["f1"],
                }
            )
    protocol = {
        "dataset": "SKAB",
        "protocol": "model-specific official evaluation paths",
        "overlap_mean": False,
        "threshold_search": False,
        "models": results,
    }
    write_json(OUTPUT_ROOT / "protocol.json", protocol)
    print(f"comparison={csv_path}")


def main() -> None:
    args = parse_args()
    if args.worker == "asca":
        run_asca()
    elif args.worker in {"pplad", "ltfad"}:
        run_official_baseline(args.worker)
    else:
        build_comparison()


if __name__ == "__main__":
    main()
