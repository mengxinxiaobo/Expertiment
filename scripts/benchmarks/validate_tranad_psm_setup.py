#!/usr/bin/env python3
"""Stage 2 validation for the label-isolated TranAD PSM adapter.

Allowed operations: import, model construction for parameter inspection, data
loading, train-only scaling, and input-shape inspection.

Forbidden and not performed: training, model forward, score generation,
thresholding, label loading, or metric calculation.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.tranad_dataset_adapter import (
    TranADPSMDataAdapter,
    import_official_tranad_class,
    load_tranad_psm_config,
)


OUTPUT = ROOT / "results" / "TRANAD_CHECK" / "adapter_validation.json"


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> None:
    started_at = iso_now()
    timer = time.perf_counter()
    print("=" * 64)
    print("Stage 2: Validate TranAD PSM Adapter")
    print(f"Start Time: {started_at}")
    print("training=False forward=False score_generation=False")
    print("test_label_access=False")
    print("=" * 64)

    config = load_tranad_psm_config()
    model_class = import_official_tranad_class(
        float(config["training"]["learning_rate"])
    )
    model = model_class(int(config["model"]["input_channels"])).float()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if parameters != 57273 or trainable != 57273:
        raise RuntimeError(
            f"Unexpected TranAD PSM parameter count: {parameters}/{trainable}"
        )

    train_adapter = TranADPSMDataAdapter("train")
    train_batch = next(iter(train_adapter.loader("train", shuffle=False)))
    train_adapter.assert_training_isolated()
    train_adapter.assert_label_free()

    score_adapter = TranADPSMDataAdapter("score")
    test_batch = next(iter(score_adapter.loader("test", shuffle=False)))
    score_adapter.assert_label_free()

    expected = (128, 10, 25)
    if tuple(train_batch.shape) != expected:
        raise RuntimeError(
            f"Unexpected TranAD train batch {tuple(train_batch.shape)} != {expected}"
        )
    if tuple(test_batch.shape) != expected:
        raise RuntimeError(
            f"Unexpected TranAD test batch {tuple(test_batch.shape)} != {expected}"
        )
    if train_batch.dtype != torch.float32 or test_batch.dtype != torch.float32:
        raise TypeError("TranAD adapter batches must be torch.float32")

    train_audit = train_adapter.audit()
    score_audit = score_adapter.audit()
    all_accessed = train_audit["files_accessed"] + score_audit["files_accessed"]
    if any("label" in path.lower() for path in all_accessed):
        raise RuntimeError("Validation access log contains a test label path")
    if score_audit["scaler_fit_files"] != ["dataset/PSM/PSM_train.npy"]:
        raise RuntimeError("Score adapter scaler was not fitted on train only")

    elapsed = time.perf_counter() - timer
    finished_at = iso_now()
    payload = {
        "stage": 2,
        "dataset": "PSM",
        "status": "PASS",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed,
        "import_status": {
            "adapter": "PASS",
            "official_tranad_class": "PASS",
            "official_main_imported": False,
            "official_evaluator_imported": False,
        },
        "loader_status": "PASS",
        "batch_shape": {
            "train": list(train_batch.shape),
            "test": list(test_batch.shape),
        },
        "dtype": {
            "train": str(train_batch.dtype),
            "test": str(test_batch.dtype),
        },
        "parameters": {
            "total": parameters,
            "trainable": trainable,
        },
        "file_access_record": {
            "train_phase": train_audit,
            "score_phase": score_audit,
        },
        "test_label_access": False,
        "training_test_feature_access": False,
        "standard_scaler_fit": "train_only",
        "training": False,
        "forward": False,
        "score_generated": False,
        "metrics_generated": False,
        "pot_spot_called": False,
        "bf_search_called": False,
        "f1_threshold_search_called": False,
        "model_source_modified": False,
        "frozen_protocol": {
            "seed": config["seed"],
            "window": config["model"]["window"],
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "optimizer": config["training"]["optimizer"],
            "learning_rate": config["training"]["learning_rate"],
            "anomaly_ratio": config["evaluation"]["anomaly_ratio"],
            "percentile": config["evaluation"]["percentile"],
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("Adapter import: PASS")
    print("Official TranAD class import: PASS")
    print(f"Train batch: {tuple(train_batch.shape)} {train_batch.dtype}")
    print(f"Test batch:  {tuple(test_batch.shape)} {test_batch.dtype}")
    print(f"Train files: {train_audit['files_accessed']}")
    print(f"Score files: {score_audit['files_accessed']}")
    print("StandardScaler fit: train_only")
    print("test_label_access=False")
    print("training=False forward=False score_generated=False metrics_generated=False")
    print(f"End Time: {finished_at}")
    print(f"Elapsed Time: {elapsed:.3f}s")
    print(f"report={OUTPUT}")


if __name__ == "__main__":
    main()
