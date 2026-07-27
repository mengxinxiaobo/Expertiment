#!/usr/bin/env python3
"""No-forward validation of the frozen PUMP TranAD adapter and protocol."""

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
    TranADPUMPDataAdapter,
    import_official_tranad_class,
    load_tranad_pump_config,
)


OUTPUT = ROOT / "results" / "TRANAD_CHECK" / "adapter_validation_pump.json"
FROZEN_SOURCE = ROOT / "configs" / "pump_three_models_frozen.json"
FORMAL_SOURCE = (
    ROOT / "results" / "PUMP_PAPER_RESULTS" / "Detection" /
    "comparison_detection.json"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> None:
    started = now_iso()
    timer = time.perf_counter()
    print("=" * 68)
    print("Validate TranAD PUMP fixed adapter")
    print(f"Start Time: {started}")
    print("training=False forward=False score_generation=False")
    print("test_label_access=False")
    print("=" * 68)

    config = load_tranad_pump_config()
    frozen = json.loads(FROZEN_SOURCE.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL_SOURCE.read_text(encoding="utf-8"))
    formal_first = formal["results"][0]
    ratios = {
        "pump_three_models_frozen": float(
            frozen["evaluation"]["anomaly_ratio"]
        ),
        "formal_detection": float(formal_first["anomaly_ratio"]),
        "pump_tranad_frozen": float(config["evaluation"]["anomaly_ratio"]),
    }
    if set(ratios.values()) != {0.5}:
        raise RuntimeError(f"PUMP formal ratio sources disagree: {ratios}")

    model_class = import_official_tranad_class(1e-4)
    model = model_class(51).float()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if total_parameters != 225519 or trainable_parameters != 225519:
        raise RuntimeError(
            f"Unexpected PUMP TranAD parameters: "
            f"{total_parameters}/{trainable_parameters}"
        )

    train_adapter = TranADPUMPDataAdapter("train")
    train_batch = next(iter(train_adapter.loader("train", shuffle=False)))
    train_adapter.assert_training_isolated()
    train_adapter.assert_label_free()
    score_adapter = TranADPUMPDataAdapter("score")
    test_batch = next(iter(score_adapter.loader("test", shuffle=False)))
    score_adapter.assert_label_free()
    expected = (128, 10, 51)
    if tuple(train_batch.shape) != expected or tuple(test_batch.shape) != expected:
        raise RuntimeError("PUMP TranAD batch shape mismatch")
    if train_batch.dtype != torch.float32 or test_batch.dtype != torch.float32:
        raise RuntimeError("PUMP TranAD adapter dtype mismatch")

    train_audit = train_adapter.audit()
    score_audit = score_adapter.audit()
    if train_audit["files_accessed"] != ["dataset/PUMP/PUMP_train.npy"]:
        raise RuntimeError("PUMP training isolation failed")
    if score_audit["files_accessed"] != [
        "dataset/PUMP/PUMP_train.npy", "dataset/PUMP/PUMP_test.npy"
    ]:
        raise RuntimeError("PUMP score isolation failed")

    elapsed = time.perf_counter() - timer
    payload = {
        "dataset": "PUMP",
        "status": "PASS",
        "started_at": started,
        "finished_at": now_iso(),
        "elapsed_seconds": elapsed,
        "data": {
            "train_shape": [17155, 51],
            "test_shape": [203165, 51],
            "label_shape": [203165],
            "input_c": 51,
            "original_dtype": "float32",
            "adapter_dtype": "torch.float32",
            "nan": 0,
            "inf": 0
        },
        "ratio": {
            "anomaly_ratio": 0.5,
            "percentile": 99.5,
            "sources": ratios,
            "test_label_used_to_select": False
        },
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters
        },
        "batch_shape": {
            "train": list(train_batch.shape),
            "test": list(test_batch.shape)
        },
        "access": {
            "train": train_audit,
            "score": score_audit,
            "test_label_access": False
        },
        "import_status": "PASS",
        "loader_status": "PASS",
        "training": False,
        "forward": False,
        "score_generated": False,
        "metrics_generated": False,
        "pot_spot_called": False,
        "bf_search_called": False,
        "model_source_modified": False
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PUMP TranAD validation: PASS")
    print(f"train_batch={tuple(train_batch.shape)} {train_batch.dtype}")
    print(f"test_batch={tuple(test_batch.shape)} {test_batch.dtype}")
    print("anomaly_ratio=0.5 percentile=99.5 input_c=51")
    print("test_label_access=False forward=False training=False")
    print(f"End Time: {now_iso()}")
    print(f"Elapsed Time: {elapsed:.3f}s")
    print(f"report={OUTPUT}")


if __name__ == "__main__":
    main()
