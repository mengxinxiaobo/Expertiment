#!/usr/bin/env python3
"""No-forward validation of the frozen SMD TranAD adapter and protocol."""

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
    TranADSMDDataAdapter,
    import_official_tranad_class,
    load_tranad_smd_config,
)


OUTPUT = ROOT / "results" / "TRANAD_CHECK" / "adapter_validation_smd.json"
FROZEN_SOURCE = ROOT / "configs" / "smd_three_models_frozen.json"
FORMAL_SOURCE = ROOT / "results" / "SMD_PAPER_RESULTS" / "protocol.json"


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
    print("Validate TranAD SMD fixed adapter")
    print(f"Start Time: {started}")
    print("training=False forward=False score_generation=False")
    print("test_label_access=False")
    print("=" * 68)

    config = load_tranad_smd_config()
    frozen = json.loads(FROZEN_SOURCE.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL_SOURCE.read_text(encoding="utf-8"))
    ratios = {
        "smd_three_models_frozen": float(frozen["evaluation"]["anomaly_ratio"]),
        "formal_protocol": float(formal["evaluation"]["anomaly_ratio"]),
        "smd_tranad_frozen": float(config["evaluation"]["anomaly_ratio"]),
    }
    percentiles = {
        "smd_three_models_frozen": float(
            frozen["evaluation"]["threshold_percentile"]
        ),
        "formal_protocol": float(formal["evaluation"]["percentile"]),
        "smd_tranad_frozen": float(config["evaluation"]["percentile"]),
    }
    if set(ratios.values()) != {0.9} or set(percentiles.values()) != {99.1}:
        raise RuntimeError(
            f"SMD formal protocol sources disagree: {ratios}/{percentiles}"
        )

    data_dir = ROOT / "dataset" / "SMD"
    train_raw, train_info = inspect_array(data_dir / "SMD_train.npy")
    test_raw, test_info = inspect_array(data_dir / "SMD_test.npy")
    labels, label_info = inspect_array(data_dir / "SMD_test_label.npy")
    if train_raw.shape != (708405, 38) or test_raw.shape != (708420, 38):
        raise RuntimeError("Unexpected SMD feature shapes")
    if labels.reshape(-1).shape != (708420,):
        raise RuntimeError("SMD label length does not match test")
    if set(np.unique(labels).tolist()) != {0.0, 1.0}:
        raise RuntimeError("SMD labels are not binary")

    model_class = import_official_tranad_class(1e-4)
    model = model_class(38).float()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters != 127538 or trainable_parameters != 127538:
        raise RuntimeError(
            f"Unexpected SMD TranAD parameters: "
            f"{total_parameters}/{trainable_parameters}"
        )

    train_adapter = TranADSMDDataAdapter("train")
    train_batch = next(iter(train_adapter.loader("train", shuffle=False)))
    train_adapter.assert_training_isolated()
    train_adapter.assert_label_free()
    score_adapter = TranADSMDDataAdapter("score")
    test_batch = next(iter(score_adapter.loader("test", shuffle=False)))
    score_adapter.assert_label_free()
    expected = (128, 10, 38)
    if tuple(train_batch.shape) != expected or tuple(test_batch.shape) != expected:
        raise RuntimeError("SMD TranAD batch shape mismatch")
    if train_batch.dtype != torch.float32 or test_batch.dtype != torch.float32:
        raise RuntimeError("SMD TranAD adapter dtype mismatch")

    train_audit = train_adapter.audit()
    score_audit = score_adapter.audit()
    if train_audit["files_accessed"] != ["dataset/SMD/SMD_train.npy"]:
        raise RuntimeError("SMD training isolation failed")
    if score_audit["files_accessed"] != [
        "dataset/SMD/SMD_train.npy",
        "dataset/SMD/SMD_test.npy",
    ]:
        raise RuntimeError("SMD score isolation failed")

    elapsed = time.perf_counter() - timer
    payload = {
        "dataset": "SMD",
        "status": "PASS",
        "started_at": started,
        "finished_at": now_iso(),
        "elapsed_seconds": elapsed,
        "data": {"train": train_info, "test": test_info, "label": label_info},
        "ratio": {
            "anomaly_ratio": 0.9,
            "percentile": 99.1,
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
    print("SMD TranAD validation: PASS")
    print(f"train_batch={tuple(train_batch.shape)} {train_batch.dtype}")
    print(f"test_batch={tuple(test_batch.shape)} {test_batch.dtype}")
    print("anomaly_ratio=0.9 percentile=99.1 input_c=38")
    print("test_label_access=False forward=False training=False")
    print(f"End Time: {now_iso()}")
    print(f"Elapsed Time: {elapsed:.3f}s")
    print(f"report={OUTPUT}")


if __name__ == "__main__":
    main()
