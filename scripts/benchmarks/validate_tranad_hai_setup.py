#!/usr/bin/env python3
"""No-forward validation of the frozen HAI TranAD adapter and protocol."""

from __future__ import annotations

import gc
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
    TranADHAIDataAdapter,
    import_official_tranad_class,
    load_tranad_hai_config,
)


OUTPUT = ROOT / "results" / "TRANAD_CHECK" / "adapter_validation_hai.json"
PROTOCOL_SOURCE = ROOT / "results" / "HAI_PAPER_RESULTS" / "protocol.json"
DETECTION_SOURCE = (
    ROOT / "results" / "HAI_PAPER_RESULTS" / "Detection" /
    "comparison_detection.json"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def inspect_array(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    nan_count = 0
    inf_count = 0
    for start in range(0, len(values), 100000):
        chunk = np.asarray(values[start : start + 100000])
        nan_count += int(np.isnan(chunk).sum())
        inf_count += int(np.isinf(chunk).sum())
    unique = None
    if "label" in path.name:
        keys, counts = np.unique(values, return_counts=True)
        unique = {str(key.item()): int(count) for key, count in zip(keys, counts)}
    return values, {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "bytes": path.stat().st_size,
        "nan": nan_count,
        "inf": inf_count,
        "distribution": unique,
    }


def main() -> None:
    started = now_iso()
    timer = time.perf_counter()
    print("=" * 68)
    print("Validate TranAD HAI fixed adapter")
    print(f"Start Time: {started}")
    print("training=False forward=False score_generation=False")
    print("test_label_access=False")
    print("=" * 68)

    config = load_tranad_hai_config()
    protocol = json.loads(PROTOCOL_SOURCE.read_text(encoding="utf-8"))
    detection = json.loads(DETECTION_SOURCE.read_text(encoding="utf-8"))
    detection_first = detection["results"][0]
    ratios = {
        "formal_protocol": float(protocol["evaluation"]["anomaly_ratio"]),
        "formal_detection": float(detection_first["anomaly_ratio"]),
        "hai_tranad_frozen": float(config["evaluation"]["anomaly_ratio"]),
    }
    percentiles = {
        "formal_protocol": float(protocol["evaluation"]["percentile"]),
        "formal_detection": float(detection_first["percentile"]),
        "hai_tranad_frozen": float(config["evaluation"]["percentile"]),
    }
    if set(ratios.values()) != {0.98} or set(percentiles.values()) != {99.02}:
        raise RuntimeError(
            f"HAI formal protocol sources disagree: {ratios}/{percentiles}"
        )

    data_dir = ROOT / "dataset" / "HAI"
    train_raw, train_info = inspect_array(data_dir / "HAI_train.npy")
    test_raw, test_info = inspect_array(data_dir / "HAI_test.npy")
    labels, label_info = inspect_array(data_dir / "HAI_test_label.npy")
    if train_raw.shape != (896400, 86) or test_raw.shape != (284400, 86):
        raise RuntimeError("Unexpected HAI feature shapes")
    if labels.reshape(-1).shape != (284400,):
        raise RuntimeError("HAI label length does not match test")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("HAI labels are not binary")
    del train_raw, test_raw, labels

    model_class = import_official_tranad_class(1e-4)
    model = model_class(86).float()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters != 627074 or trainable_parameters != 627074:
        raise RuntimeError(
            f"Unexpected HAI TranAD parameters: "
            f"{total_parameters}/{trainable_parameters}"
        )

    train_adapter = TranADHAIDataAdapter("train")
    train_batch = next(iter(train_adapter.loader("train", shuffle=False)))
    train_adapter.assert_training_isolated()
    train_adapter.assert_label_free()
    train_audit = train_adapter.audit()
    train_shape = list(train_batch.shape)
    del train_batch, train_adapter
    gc.collect()

    score_adapter = TranADHAIDataAdapter("score")
    test_batch = next(iter(score_adapter.loader("test", shuffle=False)))
    score_adapter.assert_label_free()
    score_audit = score_adapter.audit()
    test_shape = list(test_batch.shape)
    expected = [128, 10, 86]
    if train_shape != expected or test_shape != expected:
        raise RuntimeError("HAI TranAD batch shape mismatch")
    if test_batch.dtype != torch.float32:
        raise RuntimeError("HAI TranAD adapter dtype mismatch")

    if train_audit["files_accessed"] != ["dataset/HAI/HAI_train.npy"]:
        raise RuntimeError("HAI training isolation failed")
    if score_audit["files_accessed"] != [
        "dataset/HAI/HAI_train.npy",
        "dataset/HAI/HAI_test.npy",
    ]:
        raise RuntimeError("HAI score isolation failed")

    elapsed = time.perf_counter() - timer
    payload = {
        "dataset": "HAI",
        "status": "PASS",
        "started_at": started,
        "finished_at": now_iso(),
        "elapsed_seconds": elapsed,
        "data": {"train": train_info, "test": test_info, "label": label_info},
        "ratio": {
            "anomaly_ratio": 0.98,
            "percentile": 99.02,
            "ratio_sources": ratios,
            "percentile_sources": percentiles,
            "test_label_used_to_select": False,
        },
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "batch_shape": {"train": train_shape, "test": test_shape},
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
    print("HAI TranAD validation: PASS")
    print(f"train_batch={tuple(train_shape)} torch.float32")
    print(f"test_batch={tuple(test_shape)} {test_batch.dtype}")
    print("anomaly_ratio=0.98 percentile=99.02 input_c=86")
    print("test_label_access=False forward=False training=False")
    print(f"End Time: {now_iso()}")
    print(f"Elapsed Time: {elapsed:.3f}s")
    print(f"report={OUTPUT}")


if __name__ == "__main__":
    main()
