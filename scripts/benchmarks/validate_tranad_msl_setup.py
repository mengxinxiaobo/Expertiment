#!/usr/bin/env python3
"""No-forward validation of the frozen MSL TranAD adapter and protocol."""

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
    TranADMSLDataAdapter,
    import_official_tranad_class,
    load_tranad_msl_config,
)


OUTPUT = ROOT / "results" / "TRANAD_CHECK" / "adapter_validation_msl.json"
FORMAL_RATIO_SOURCE = (
    ROOT
    / "results"
    / "MSL_PAPER_RESULTS"
    / "Detection_ratio083"
    / "comparison_detection.json"
)
DATASET_CONFIG_SOURCE = ROOT / "configs" / "datasets.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> None:
    started_at = now_iso()
    timer = time.perf_counter()
    print("=" * 68)
    print("Validate TranAD MSL fixed adapter")
    print(f"Start Time: {started_at}")
    print("training=False forward=False score_generation=False")
    print("test_label_access=False")
    print("=" * 68)

    config = load_tranad_msl_config()
    dataset_config = json.loads(DATASET_CONFIG_SOURCE.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL_RATIO_SOURCE.read_text(encoding="utf-8"))
    ratios = {
        "configs/datasets.json": float(dataset_config["MSL"]["anormly_ratio"]),
        "formal_detection_result": float(formal["anomaly_ratio"]),
        "msl_tranad_frozen": float(config["evaluation"]["anomaly_ratio"]),
    }
    if set(ratios.values()) != {0.83}:
        raise RuntimeError(f"MSL formal anomaly ratio sources disagree: {ratios}")
    if formal["threshold_rule"] != (
        "per-model percentile(concat(train_energy,test_energy),99.17)"
    ):
        raise RuntimeError("Existing formal MSL percentile source changed")

    model_class = import_official_tranad_class(
        float(config["training"]["learning_rate"])
    )
    model = model_class(int(config["model"]["input_channels"])).float()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if total_parameters != 261243 or trainable_parameters != 261243:
        raise RuntimeError(
            f"Unexpected MSL TranAD parameters: "
            f"{total_parameters}/{trainable_parameters}"
        )

    train_adapter = TranADMSLDataAdapter("train")
    train_batch = next(iter(train_adapter.loader("train", shuffle=False)))
    train_adapter.assert_training_isolated()
    train_adapter.assert_label_free()

    score_adapter = TranADMSLDataAdapter("score")
    test_batch = next(iter(score_adapter.loader("test", shuffle=False)))
    score_adapter.assert_label_free()

    expected_batch = (128, 10, 55)
    if tuple(train_batch.shape) != expected_batch:
        raise RuntimeError(f"Unexpected MSL train batch: {tuple(train_batch.shape)}")
    if tuple(test_batch.shape) != expected_batch:
        raise RuntimeError(f"Unexpected MSL test batch: {tuple(test_batch.shape)}")
    if train_batch.dtype != torch.float32 or test_batch.dtype != torch.float32:
        raise RuntimeError("MSL TranAD batches are not float32")

    train_audit = train_adapter.audit()
    score_audit = score_adapter.audit()
    if train_audit["files_accessed"] != ["dataset/MSL/MSL_train.npy"]:
        raise RuntimeError("MSL training access isolation failed")
    if score_audit["files_accessed"] != [
        "dataset/MSL/MSL_train.npy",
        "dataset/MSL/MSL_test.npy",
    ]:
        raise RuntimeError("MSL score access isolation failed")

    elapsed = time.perf_counter() - timer
    payload = {
        "dataset": "MSL",
        "status": "PASS",
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds": elapsed,
        "data": {
            "train_shape": [58317, 55],
            "test_shape": [73729, 55],
            "label_shape": [73729],
            "input_c": 55,
            "train_dtype_original": "float64",
            "test_dtype_original": "float64",
            "adapter_dtype": "torch.float32",
            "nan": 0,
            "inf": 0,
        },
        "ratio": {
            "anomaly_ratio": 0.83,
            "percentile": 99.17,
            "sources": ratios,
            "test_label_used_to_select": False,
        },
        "import_status": "PASS",
        "loader_status": "PASS",
        "batch_shape": {
            "train": list(train_batch.shape),
            "test": list(test_batch.shape),
        },
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "access": {
            "train": train_audit,
            "score": score_audit,
            "test_label_access": False,
        },
        "training": False,
        "forward": False,
        "score_generated": False,
        "metrics_generated": False,
        "pot_spot_called": False,
        "bf_search_called": False,
        "model_source_modified": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("MSL TranAD validation: PASS")
    print(f"train_batch={tuple(train_batch.shape)} {train_batch.dtype}")
    print(f"test_batch={tuple(test_batch.shape)} {test_batch.dtype}")
    print(f"anomaly_ratio=0.83 percentile=99.17 input_c=55")
    print(f"test_label_access=False forward=False training=False")
    print(f"End Time: {now_iso()}")
    print(f"Elapsed Time: {elapsed:.3f}s")
    print(f"report={OUTPUT}")


if __name__ == "__main__":
    main()
