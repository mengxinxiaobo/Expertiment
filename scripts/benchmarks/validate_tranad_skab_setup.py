#!/usr/bin/env python3
"""No-forward validation of the frozen SKAB TranAD adapter and protocol."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.tranad_dataset_adapter import (
    TranADSKABDataAdapter,
    import_official_tranad_class,
    load_tranad_skab_config,
)


OUTPUT = ROOT / "results" / "TRANAD_CHECK" / "adapter_validation_skab.json"
PROTOCOL_SOURCE = ROOT / "results" / "SKAB_BENCHMARK" / "protocol.json"
THREE_MODEL_SOURCE = (
    ROOT / "results" / "SKAB_BENCHMARK" / "three_model_protocol.json"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def inspect_array(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    values = np.load(path, allow_pickle=False)
    unique = None
    if "label" in path.name:
        keys, counts = np.unique(values, return_counts=True)
        unique = {str(key.item()): int(count) for key, count in zip(keys, counts)}
    return values, {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "bytes": path.stat().st_size,
        "nan": int(np.isnan(values).sum()),
        "inf": int(np.isinf(values).sum()),
        "distribution": unique,
    }


def main() -> None:
    started = now_iso()
    timer = time.perf_counter()
    print("=" * 68)
    print("Validate TranAD SKAB fixed adapter")
    print(f"Start Time: {started}")
    print("training=False forward=False score_generation=False")
    print("test_label_access=False")
    print("=" * 68)

    config = load_tranad_skab_config()
    protocol = json.loads(PROTOCOL_SOURCE.read_text(encoding="utf-8"))
    three_model = json.loads(THREE_MODEL_SOURCE.read_text(encoding="utf-8"))
    ratios = {
        "skab_protocol": float(protocol["evaluation"]["anomaly_ratio"]),
        "three_model_protocol": float(
            three_model["evaluation"]["anomaly_ratio"]
        ),
        "skab_tranad_frozen": float(config["evaluation"]["anomaly_ratio"]),
    }
    percentiles = {
        "skab_protocol": float(protocol["evaluation"]["percentile"]),
        "three_model_protocol": float(three_model["evaluation"]["percentile"]),
        "skab_tranad_frozen": float(config["evaluation"]["percentile"]),
    }
    if set(ratios.values()) != {0.5} or set(percentiles.values()) != {99.5}:
        raise RuntimeError(
            f"SKAB formal protocol sources disagree: {ratios}/{percentiles}"
        )

    data_dir = ROOT / "dataset" / "SKAB"
    train_raw, train_info = inspect_array(data_dir / "SKAB_train.npy")
    test_raw, test_info = inspect_array(data_dir / "SKAB_test.npy")
    labels, label_info = inspect_array(data_dir / "SKAB_test_label.npy")
    if train_raw.shape != (12450, 8) or test_raw.shape != (5710, 8):
        raise RuntimeError("Unexpected SKAB feature shapes")
    if labels.reshape(-1).shape != (5710,):
        raise RuntimeError("SKAB label length does not match test")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("SKAB labels are not binary")

    model_class = import_official_tranad_class(1e-4)
    model = model_class(8).float()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters != 7208 or trainable_parameters != 7208:
        raise RuntimeError(
            f"Unexpected SKAB TranAD parameters: "
            f"{total_parameters}/{trainable_parameters}"
        )

    train_adapter = TranADSKABDataAdapter("train")
    train_batch = next(iter(train_adapter.loader("train", shuffle=False)))
    train_adapter.assert_training_isolated()
    train_adapter.assert_label_free()
    score_adapter = TranADSKABDataAdapter("score")
    test_batch = next(iter(score_adapter.loader("test", shuffle=False)))
    score_adapter.assert_label_free()
    expected = (128, 10, 8)
    if tuple(train_batch.shape) != expected or tuple(test_batch.shape) != expected:
        raise RuntimeError("SKAB TranAD batch shape mismatch")
    if train_batch.dtype != torch.float32 or test_batch.dtype != torch.float32:
        raise RuntimeError("SKAB TranAD adapter dtype mismatch")

    train_audit = train_adapter.audit()
    score_audit = score_adapter.audit()
    if train_audit["files_accessed"] != ["dataset/SKAB/SKAB_train.npy"]:
        raise RuntimeError("SKAB training isolation failed")
    if score_audit["files_accessed"] != [
        "dataset/SKAB/SKAB_train.npy",
        "dataset/SKAB/SKAB_test.npy",
    ]:
        raise RuntimeError("SKAB score isolation failed")

    elapsed = time.perf_counter() - timer
    payload = {
        "dataset": "SKAB",
        "status": "PASS",
        "started_at": started,
        "finished_at": now_iso(),
        "elapsed_seconds": elapsed,
        "data": {"train": train_info, "test": test_info, "label": label_info},
        "ratio": {
            "anomaly_ratio": 0.5,
            "percentile": 99.5,
            "ratio_sources": ratios,
            "percentile_sources": percentiles,
            "test_label_used_to_select": False,
        },
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "batch_shape": {
            "train": list(train_batch.shape),
            "test": list(test_batch.shape),
        },
        "access": {
            "train": train_audit,
            "score": score_audit,
            "test_label_access_by_adapter": False,
        },
        "import_status": "PASS",
        "loader_status": "PASS",
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
    print("SKAB TranAD validation: PASS")
    print(f"train_batch={tuple(train_batch.shape)} {train_batch.dtype}")
    print(f"test_batch={tuple(test_batch.shape)} {test_batch.dtype}")
    print("anomaly_ratio=0.5 percentile=99.5 input_c=8")
    print("test_label_access=False forward=False training=False")
    print(f"End Time: {now_iso()}")
    print(f"Elapsed Time: {elapsed:.3f}s")
    print(f"report={OUTPUT}")


if __name__ == "__main__":
    main()
