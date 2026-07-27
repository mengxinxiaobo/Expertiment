#!/usr/bin/env python3
"""Fixed-protocol SimAD detection experiment on PSM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.simad_dataset_adapter import (  # noqa: E402
    CONFIG_PATH,
    SimADPSMDataAdapter,
    SimADWindowScoreAdapter,
    load_frozen_config,
)
from scripts.benchmarks.build_pump_paper_results import point_adjust  # noqa: E402


SIMAD_ROOT = ROOT / "BaselineModels" / "SimAD-main"
OUTPUT_ROOT = ROOT / "results" / "PSM_SIMAD_RESULTS"
DETECTION_ROOT = OUTPUT_ROOT / "Detection"
SCORES_ROOT = OUTPUT_ROOT / "scores"
CHECKPOINT = DETECTION_ROOT / "SimAD" / "SimAD_state_dict.pt"
METRICS_PATH = DETECTION_ROOT / "SimAD" / "metrics.json"
SEED = 42
ANOMALY_RATIO = 0.8
PERCENTILE = 99.2


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_simad():
    if str(SIMAD_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMAD_ROOT))
    from models_v2.model2_lazy import ContAD_wo_ci
    from trainer_v2.trainer2 import ContAD_Trainer

    return ContAD_wo_ci, ContAD_Trainer


def build_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    ContAD_wo_ci, _trainer = import_simad()
    model_config = config["model"]
    model = ContAD_wo_ci(
        c_dim=25,
        seq_len=int(model_config["win_size"]),
        patch_size=int(model_config["patch_size"]),
        d_model=int(model_config["d_model"]),
        n_layers=int(model_config["n_layers"]),
        d_feed_foward_scale=int(model_config["ffn_scale"]),
        n_head=int(model_config["n_head"]),
        dropout=float(model_config["dropout"]),
        with_inter=int(model_config["with_inter"]),
        with_intra=int(model_config["with_intra"]),
        proj_dim=int(model_config["proj_dim"]),
        query_len=int(model_config["q_len"]),
    )
    return model.to(device)


def trainer_config(config: dict[str, Any]) -> Namespace:
    model = config["model"]
    training = config["training"]
    return Namespace(
        data_pth=str(ROOT / "dataset"),
        data_name="PSM",
        index=1,
        model_id="fixed_psm_seed42",
        patch_size=int(model["patch_size"]),
        d_model=int(model["d_model"]),
        ffn_scale=int(model["ffn_scale"]),
        n_head=int(model["n_head"]),
        n_layers=int(model["n_layers"]),
        q_len=int(model["q_len"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        accumulate_grad_steps=int(training["accumulate_grad_steps"]),
        lr=float(training["lr"]),
        momentum=0.9,
        print_freq=50,
        test_freq=1000,
        use_amp=int(bool(training["use_amp"])),
        multiprocessing=0,
        world_size=-1,
        rank=-1,
        dist_url="tcp://127.0.0.1:23437",
        dist_backend="nccl",
        workers=0,
        resume=0,
        logs_pth=str(OUTPUT_ROOT / "official_training_logs"),
        save_pth="checkpoints",
        seed=SEED,
        optimizer=str(training["optimizer"]),
        distributed=False,
        gpu=0,
        ar=0.1,
        win_size=int(model["win_size"]),
        step=int(model["step"]),
        noise_level=float(training["noise_level"]),
        warmup_steps=int(training["warmup_steps"]),
        warmup_max_ratio=float(training["warmup_max_ratio"]),
        nc=25,
    )


def train_model(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> tuple[float, list[str]]:
    _model_class, trainer_class = import_simad()
    adapter = SimADPSMDataAdapter(phase="train")
    args = trainer_config(config)
    trainer = trainer_class(args, model, adapter.official_loader)

    # Official SimAD train() calls test() periodically and its test() performs
    # label-driven best-F1 threshold search. The fixed protocol disables only
    # that evaluation hook; the official model and train() implementation stay
    # unchanged.
    trainer.test = lambda *_args, **_kwargs: None
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    adapter.assert_training_label_free()
    return elapsed, adapter.files_accessed


@torch.no_grad()
def generate_scores(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    adapter = SimADPSMDataAdapter(phase="score")
    scorer = SimADWindowScoreAdapter(model).to(device).eval()
    stride = int(config["scoring"]["window_stride"])
    score_arrays: dict[str, np.ndarray] = {}
    score_metadata: dict[str, Any] = {}
    for split in ("train", "test"):
        loader = adapter.loader(
            split,
            stride=stride,
            shuffle=False,
            batch_size=int(config["training"]["batch_size"]),
        )
        chunks: list[np.ndarray] = []
        input_shape = None
        score_shape = None
        for windows, _placeholder in loader:
            windows = windows.to(device=device, dtype=torch.float32)
            scores = scorer.window_scores(windows)
            if not torch.isfinite(scores).all():
                raise RuntimeError(f"Non-finite SimAD {split} score")
            input_shape = input_shape or list(windows.shape)
            score_shape = score_shape or list(scores.shape)
            chunks.append(scores.reshape(-1).cpu().numpy().astype(np.float32))
        energy = np.concatenate(chunks)
        expected_points = len(loader.dataset) * int(config["model"]["win_size"])
        if energy.shape != (expected_points,):
            raise RuntimeError(
                f"Unexpected {split} energy shape {energy.shape}; "
                f"expected {(expected_points,)}"
            )
        score_arrays[split] = energy
        score_metadata[split] = {
            "input_shape": input_shape,
            "window_score_shape": score_shape,
            "windows": len(loader.dataset),
            "energy_length": int(energy.size),
            "stride": stride,
        }
        print(
            f"[SimAD][{split}] input={tuple(input_shape)}, "
            f"window_score={tuple(score_shape)}, windows={len(loader.dataset)}, "
            f"energy={energy.size}",
            flush=True,
        )
    adapter.assert_model_phase_label_free()
    score_metadata["files_accessed"] = adapter.files_accessed
    score_metadata["test_label_access"] = False
    return score_arrays["train"], score_arrays["test"], score_metadata


def binary_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate(
    train_score: np.ndarray,
    test_score: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    # This is the only function allowed to open PSM_test_label.npy.
    label_path = ROOT / "dataset" / "PSM" / "PSM_test_label.npy"
    labels = np.asarray(
        np.load(label_path, allow_pickle=False), dtype=np.int64
    ).reshape(-1)
    if labels.shape != (87841,):
        raise RuntimeError(f"Unexpected PSM label shape: {labels.shape}")
    labels = labels[: test_score.size]
    threshold = float(
        np.percentile(np.concatenate([train_score, test_score]), PERCENTILE)
    )
    prediction = (test_score > threshold).astype(np.int64)
    raw = binary_metrics(labels, prediction)
    pa_prediction = prediction.copy()
    point_adjust(pa_prediction, labels)
    pa = binary_metrics(labels, pa_prediction)
    return {
        "model": "SimAD",
        "anomaly_ratio": ANOMALY_RATIO,
        "percentile": PERCENTILE,
        "threshold": threshold,
        "threshold_samples": int(train_score.size + test_score.size),
        "evaluated_points": int(labels.size),
        "raw": raw,
        "pa": pa,
    }, prediction, pa_prediction


def write_detection_outputs(
    result: dict[str, Any],
    config: dict[str, Any],
    train_score: np.ndarray,
    test_score: np.ndarray,
    raw_prediction: np.ndarray,
    pa_prediction: np.ndarray,
    training_seconds: float,
    training_files: list[str],
    score_metadata: dict[str, Any],
) -> None:
    DETECTION_ROOT.mkdir(parents=True, exist_ok=True)
    SCORES_ROOT.mkdir(parents=True, exist_ok=True)
    model_dir = DETECTION_ROOT / "SimAD"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(SCORES_ROOT / "train_score.npy", train_score, allow_pickle=False)
    np.save(SCORES_ROOT / "test_score.npy", test_score, allow_pickle=False)
    np.save(model_dir / "pred_raw.npy", raw_prediction, allow_pickle=False)
    np.save(model_dir / "pred_pa.npy", pa_prediction, allow_pickle=False)

    result.update({
        "parameters": config["runtime_parameters"],
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "training_seconds": training_seconds,
        "configuration": config["model"],
        "training_configuration": config["training"],
        "score_configuration": config["scoring"],
        "score_metadata": score_metadata,
        "audit": {
            "training_label_access": False,
            "training_test_feature_access": False,
            "training_test_label_access": False,
            "test_label_access": False,
            "evaluator_label_access": True,
            "score_search": False,
            "ratio_search": False,
            "threshold_search": False,
            "parameter_search": False,
            "test_label_parameter_selection": False,
            "simad_model_source_modified": False,
            "training_files_accessed": training_files,
        },
    })
    write_json(METRICS_PATH, result)
    write_json(DETECTION_ROOT / "comparison_detection.json", {
        "dataset": "PSM",
        "results": [result],
    })
    fields = [
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "PA-Accuracy", "PA-Precision", "PA-Recall", "PA-F1", "Threshold",
    ]
    with (DETECTION_ROOT / "comparison_detection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "Model": "SimAD",
            "Accuracy": result["raw"]["accuracy"],
            "Precision": result["raw"]["precision"],
            "Recall": result["raw"]["recall"],
            "F1": result["raw"]["f1"],
            "PA-Accuracy": result["pa"]["accuracy"],
            "PA-Precision": result["pa"]["precision"],
            "PA-Recall": result["pa"]["recall"],
            "PA-F1": result["pa"]["f1"],
            "Threshold": result["threshold"],
        })

    data_root = ROOT / "dataset" / "PSM"
    write_json(OUTPUT_ROOT / "protocol.json", {
        "dataset": "PSM",
        "seed": SEED,
        "dtype": "float32",
        "data": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for name, path in {
                "train": data_root / "PSM_train.npy",
                "test": data_root / "PSM_test.npy",
                "label": data_root / "PSM_test_label.npy",
            }.items()
        },
        "frozen_config": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "model_source": "BaselineModels/SimAD-main (unmodified)",
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "training": config["training"],
        "model": config["model"],
        "scoring": config["scoring"],
        "evaluation": config["evaluation"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None
            ),
        },
        "audit": result["audit"],
    })


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official SimAD PSM experiment")
    if METRICS_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"SimAD PSM result already exists: {METRICS_PATH}. "
            "Use --overwrite only for an explicitly authorized rerun."
        )
    set_seed()
    config = load_frozen_config()
    device = torch.device("cuda:0")
    model = build_model(config, device)
    config["runtime_parameters"] = int(
        sum(parameter.numel() for parameter in model.parameters())
    )

    training_seconds, training_files = train_model(model, config)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "seed": SEED,
        },
        CHECKPOINT,
    )
    model.eval()
    train_score, test_score, score_metadata = generate_scores(
        model, config, device
    )
    result, raw_prediction, pa_prediction = evaluate(train_score, test_score)
    write_detection_outputs(
        result,
        config,
        train_score,
        test_score,
        raw_prediction,
        pa_prediction,
        training_seconds,
        training_files,
        score_metadata,
    )
    print(
        f"SimAD: threshold={result['threshold']:.12g}, "
        f"RAW_F1={result['raw']['f1']:.6f}, "
        f"PA_F1={result['pa']['f1']:.6f}",
        flush=True,
    )
    print(f"metrics={METRICS_PATH}", flush=True)


if __name__ == "__main__":
    main()
