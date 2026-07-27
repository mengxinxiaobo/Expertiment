#!/usr/bin/env python3
"""Small, non-formal official-COUTA validation on a train-only PSM subset."""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.couta_dataset_adapter import (
    CONFIG_PATH,
    PSMCOUTADataAdapter,
    RAY_AUDIT,
    construct_official_couta,
    import_official_couta,
    load_couta_config,
    make_bundle,
    restore_from_bundle,
    sha256,
)


OUTPUT_ROOT = ROOT / "results" / "PSM_COUTA_RESULTS" / "Validation"
REPORT_PATH = OUTPUT_ROOT / "validation.json"
BUNDLE_PATH = OUTPUT_ROOT / "validation_bundle.pt"
STATE_PATH = OUTPUT_ROOT / "validation_state_dict.pt"
VALIDATION_POINTS = 4096


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> None:
    started_at = now_iso()
    started = time.perf_counter()
    print("=" * 72, flush=True)
    print("COUTA PSM formal-detection setup validation", flush=True)
    print(f"Start Time: {started_at}", flush=True)
    print("formal_training=False formal_scores=False formal_evaluation=False", flush=True)
    print("=" * 72, flush=True)

    config = load_couta_config()
    if RAY_AUDIT["ray_installed"]:
        raise RuntimeError("Validation requires ray_installed=false")
    COUTA, _Network = import_official_couta()
    import_status = COUTA.__module__ == "deepod.models.time_series.couta"
    if not import_status or not RAY_AUDIT["ray_stub_used"]:
        raise RuntimeError("Official COUTA import or Ray stub failed")

    from ray import tune
    ray_blocked = False
    try:
        tune.choice([1, 2])
    except RuntimeError:
        ray_blocked = True
    if not ray_blocked or not RAY_AUDIT["ray_tune_attempt_blocked"]:
        raise RuntimeError("Ray stub is not fail-closed")

    adapter = PSMCOUTADataAdapter("validation")
    train = adapter.load_train(limit=VALIDATION_POINTS)
    scaler, train_scaled = adapter.fit_train_scaler(train)
    adapter.assert_training_isolated()
    if train_scaled.dtype != np.float32 or not np.isfinite(train_scaled).all():
        raise RuntimeError("Scaled validation train data is invalid")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required by the frozen validation protocol")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()

    model = construct_official_couta(config, device, epochs=1)
    log_buffer = io.StringIO()
    fit_started = time.perf_counter()
    with contextlib.redirect_stdout(log_buffer):
        result = model.fit(train_scaled)
    fit_elapsed = time.perf_counter() - fit_started
    official_log = log_buffer.getvalue()
    print(official_log, end="", flush=True)
    if result is not None:
        raise RuntimeError("Official COUTA fit unexpectedly returned a value")
    loss_values = [float(value) for value in re.findall(r"loss:\s*([0-9eE+.-]+)", official_log)]
    if not loss_values or not all(math.isfinite(value) for value in loss_values):
        raise RuntimeError("Official COUTA did not emit a finite validation loss")

    total_parameters = sum(parameter.numel() for parameter in model.net.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.net.parameters()
        if parameter.requires_grad
    )
    if total_parameters != 2897 or trainable_parameters != 2897:
        raise RuntimeError(
            f"Unexpected official COUTA parameters: {total_parameters}/{trainable_parameters}"
        )

    score_started = time.perf_counter()
    score_before = model.decision_function(train_scaled)
    score_elapsed = time.perf_counter() - score_started
    if score_before.shape != (VALIDATION_POINTS,):
        raise RuntimeError("Official COUTA validation score length mismatch")
    if not np.isfinite(score_before).all() or not np.all(score_before[:29] == 0):
        raise RuntimeError("Official COUTA score alignment/finite check failed")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle = make_bundle(model, scaler, config)
    bundle["validation_epochs"] = 1
    bundle["formal_checkpoint"] = False
    torch.save(bundle, BUNDLE_PATH)
    torch.save(model.net.state_dict(), STATE_PATH)
    restored_bundle = torch.load(BUNDLE_PATH, map_location=device, weights_only=False)
    restored_model, restored_scaler = restore_from_bundle(restored_bundle, device)
    restored_scaled = np.asarray(restored_scaler.transform(train), dtype=np.float32)
    score_after = restored_model.decision_function(restored_scaled)
    max_abs_error = float(np.max(np.abs(score_before - score_after)))
    scaler_max_abs_error = float(np.max(np.abs(train_scaled - restored_scaled)))
    if max_abs_error > 1e-7 or scaler_max_abs_error > 1e-7:
        raise RuntimeError(
            f"Checkpoint restoration mismatch: score={max_abs_error}, scaler={scaler_max_abs_error}"
        )

    peak_memory = torch.cuda.max_memory_allocated()
    incremental_memory = max(0, peak_memory - baseline_memory)
    elapsed = time.perf_counter() - started
    audit = {
        "model_source_modified": False,
        "deepod_core_modified": False,
        "ray_installed": False,
        "ray_stub_used": True,
        "ray_tune_used": False,
        "fit_auto_hyper_used": False,
        "training_ray_used": False,
        "deepod_testbed_used": False,
        "ts_metrics_used": False,
        "best_f1_search": False,
        "training_test_access": False,
        "training_test_label_access": False,
        "score_label_access": False,
        "score_search": False,
        "ratio_search": False,
        "threshold_search": False,
        "parameter_search": False,
        "oracle_search": False,
        "internal_threshold_used_for_paper": False,
        "internal_prediction_used": False,
        "contamination_prediction_used": False,
        "formal_training_started": False,
        "formal_scores_generated": False,
        "formal_evaluation_run": False,
        "formal_efficiency_run": False,
    }
    payload = {
        "status": "READY",
        "dataset": "PSM",
        "model": "COUTA",
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds": elapsed,
        "config_path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "validation_subset_points": VALIDATION_POINTS,
        "validation_epochs": 1,
        "formal_epochs_unchanged": 20,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "import": {
            "official_class": f"{COUTA.__module__}.{COUTA.__name__}",
            "status": "PASS",
            "ray_call_blocked": ray_blocked,
            "ray_audit": dict(RAY_AUDIT),
        },
        "data": {
            "formal_train_shape": [132481, 25],
            "formal_test_shape": [87841, 25],
            "formal_label_shape": [87841],
            "subset_shape": list(train.shape),
            "scaled_dtype": str(train_scaled.dtype),
        },
        "scaler": {
            "fit_split": "train",
            "mean_shape": list(scaler.mean_.shape),
            "scale_shape": list(scaler.scale_.shape),
            "var_shape": list(scaler.var_.shape),
            "n_features_in": int(scaler.n_features_in_),
            "restore_max_abs_error": scaler_max_abs_error,
        },
        "parameters": {"total": total_parameters, "trainable": trainable_parameters},
        "official_fit": {
            "status": "PASS", "elapsed_seconds": fit_elapsed,
            "loss_values": loss_values, "loss_finite": True,
            "internal_threshold_created_but_not_used_for_paper": model.threshold_ is not None,
        },
        "decision_function": {
            "status": "PASS", "elapsed_seconds": score_elapsed,
            "score_shape": list(score_before.shape), "finite": True,
            "prefix_padding_points": 29,
            "prefix_all_zero": bool(np.all(score_before[:29] == 0)),
            "inference_windows": VALIDATION_POINTS - 30 + 1,
        },
        "checkpoint": {
            "bundle": str(BUNDLE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "bundle_sha256": sha256(BUNDLE_PATH),
            "state_dict": str(STATE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "state_dict_sha256": sha256(STATE_PATH),
            "score_restore_max_abs_error": max_abs_error,
            "status": "PASS",
        },
        "gpu_memory": {
            "peak_mib": peak_memory / 1024 ** 2,
            "incremental_mib": incremental_memory / 1024 ** 2,
        },
        "file_access": adapter.audit(),
        "audit": audit,
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("COUTA PSM validation: READY", flush=True)
    print(f"parameters={total_parameters}/{trainable_parameters}", flush=True)
    print(f"score_shape={score_before.shape} prefix29_zero=True", flush=True)
    print(f"checkpoint_max_abs_error={max_abs_error:.3e}", flush=True)
    print(f"gpu_peak_mib={payload['gpu_memory']['peak_mib']:.3f}", flush=True)
    print(f"End Time: {payload['finished_at']}", flush=True)
    print(f"Elapsed Time: {elapsed:.3f}s", flush=True)
    print(f"report={REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
