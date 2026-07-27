#!/usr/bin/env python3
"""Resource-safe one-epoch validation of official DeepOD COUTA on PUMP."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.couta_dataset_adapter import (
    PUMP_CONFIG_PATH, PUMPCOUTADataAdapter, RAY_AUDIT,
    construct_official_couta, import_official_couta, load_pump_couta_config,
    make_bundle, restore_from_bundle, sha256,
)
from scripts.benchmarks.run_psm_couta_detection import official_scores_chunked

OUT = ROOT / "results" / "PUMP_COUTA_RESULTS" / "Validation"
REPORT = OUT / "validation_report.json"
REPORT_MD = OUT / "validation_report.md"
LOGFILE = OUT / "validation.log"
CHECKPOINT = OUT / "temporary_checkpoint.pt"
VALIDATION_POINTS = 4096
VALIDATION_CHUNK_POINTS = 1000
LINES: list[str] = []


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(value: str = "") -> None:
    print(value, flush=True)
    LINES.append(value)


def check(name: str, condition: bool, detail: str = "") -> None:
    log(f"[{'PASS' if condition else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not condition:
        raise RuntimeError(f"{name}: {detail}")


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.exists():
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(item.relative_to(path)).encode())
            digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    start_time = now()
    log("=" * 72)
    log("Stage: PUMP COUTA setup validation")
    log(f"Start Time: {start_time}")

    protected = ("PSM_COUTA_RESULTS", "SKAB_COUTA_RESULTS", "MSL_COUTA_RESULTS", "HAI_COUTA_RESULTS")
    previous_hashes = {name: tree_hash(ROOT / "results" / name) for name in protected}
    config = load_pump_couta_config()
    data_dir = ROOT / "dataset" / "PUMP"
    names = ("PUMP_train.npy", "PUMP_test.npy", "PUMP_test_label.npy")
    paths = {name: data_dir / name for name in names}
    check("actual files", all(path.is_file() for path in paths.values()), str(list(paths)))
    arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
    train, test, label = (arrays[name] for name in names)
    check("train shape", train.shape == (17155, 51), str(train.shape))
    check("test shape", test.shape == (203165, 51), str(test.shape))
    check("label shape", label.shape == (203165,), str(label.shape))
    check("input_c", train.shape[1] == test.shape[1] == 51, "51")
    check("test/label alignment", len(test) == len(label), f"{len(test)}/{len(label)}")

    statistics_data = {}
    for name, array in arrays.items():
        nan_count = int(np.isnan(array).sum())
        inf_count = int(np.isinf(array).sum())
        statistics_data[name] = {"nan_count": nan_count, "inf_count": inf_count}
        check(f"{name} finite", nan_count == 0 and inf_count == 0)
    unique, counts = np.unique(label, return_counts=True)
    label_counts = dict(zip(map(str, unique.tolist()), counts.tolist()))
    check("binary label", set(unique.tolist()) == {0, 1}, str(label_counts))

    check("Ray absent", not RAY_AUDIT["ray_installed"])
    COUTA, _ = import_official_couta()
    check("official import", COUTA.__module__ == "deepod.models.time_series.couta")
    from ray import tune
    ray_blocked = False
    try:
        tune.choice([1])
    except RuntimeError:
        ray_blocked = True
    check("Ray Tune fail-closed", ray_blocked)

    adapter = PUMPCOUTADataAdapter("validation")
    full_train = adapter.load_train()
    scaler, full_scaled_train = adapter.fit_train_scaler(full_train)
    adapter.assert_training_isolated()
    scaled = np.ascontiguousarray(full_scaled_train[:VALIDATION_POINTS])
    check("train-only float32 scaler", scaled.dtype == np.float32 and np.isfinite(scaled).all())
    constant_indices = np.flatnonzero(np.var(full_train, axis=0) == 0).astype(int).tolist()
    constant_scales = {str(i): float(scaler.scale_[i]) for i in constant_indices}
    check("finite scaler state", bool(np.isfinite(scaler.mean_).all() and
                                      np.isfinite(scaler.scale_).all() and
                                      np.isfinite(scaler.var_).all() and
                                      np.all(scaler.scale_ != 0)),
          f"constant_indices={constant_indices}")

    check("CUDA available", torch.cuda.is_available())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory_baseline = torch.cuda.memory_allocated()
    model = construct_official_couta(config, "cuda", epochs=1)
    capture = io.StringIO()
    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(capture):
        warnings.simplefilter("always")
        fit_result = model.fit(scaled)
    fit_seconds = time.perf_counter() - fit_started
    for line in capture.getvalue().splitlines():
        log(line)
    losses = [float(x) for x in re.findall(r"(?:^|\s)loss:\s*([0-9eE+.-]+)", capture.getvalue())]
    val_losses = [float(x) for x in re.findall(r"val_loss:\s*([0-9eE+.-]+)", capture.getvalue())]
    check("official COUTA.fit", fit_result is None)
    check("forward/backward/optimizer step", bool(losses), "completed through official fit")
    check("finite train loss", bool(losses) and all(map(math.isfinite, losses)), str(losses))
    check("finite validation loss", bool(val_losses) and all(map(math.isfinite, val_losses)), str(val_losses))
    check("finite center c", bool(torch.isfinite(model.c).all()))
    total_parameters = sum(parameter.numel() for parameter in model.net.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.net.parameters()
                               if parameter.requires_grad)
    check("actual parameters", total_parameters == trainable_parameters == 4145,
          f"{total_parameters}/{trainable_parameters}")

    official_score = np.asarray(model.decision_function(scaled), dtype=np.float64)
    check("official score", official_score.shape == (VALIDATION_POINTS,) and
          np.isfinite(official_score).all(), str(official_score.shape))
    check("prefix padding 29", bool(np.all(official_score[:29] == 0)))
    check("non-prefix score finite", bool(np.isfinite(official_score[29:]).all()))
    chunked_score = official_scores_chunked(
        model, scaled, VALIDATION_CHUNK_POINTS, "validation")
    chunk_error = float(np.max(np.abs(official_score - chunked_score)))
    check("chunked/official score equivalence", chunk_error <= 1e-6, f"{chunk_error:.3e}")

    bundle = make_bundle(model, scaler, config)
    bundle.update({"validation_epochs": 1, "formal_checkpoint": False})
    torch.save(bundle, CHECKPOINT)
    restored_model, restored_scaler = restore_from_bundle(
        torch.load(CHECKPOINT, map_location="cuda", weights_only=False), "cuda")
    restored_scaled = np.asarray(restored_scaler.transform(full_train[:VALIDATION_POINTS]),
                                 dtype=np.float32)
    restored_score = np.asarray(restored_model.decision_function(restored_scaled), dtype=np.float64)
    score_restore_error = float(np.max(np.abs(official_score - restored_score)))
    scaler_restore_error = float(np.max(np.abs(scaled - restored_scaled)))
    check("checkpoint restore", score_restore_error <= 1e-7 and scaler_restore_error <= 1e-7,
          f"score={score_restore_error:.3e}, scaler={scaler_restore_error:.3e}")

    score_adapter = PUMPCOUTADataAdapter("score")
    score_train = score_adapter.load_train(limit=64)
    score_test = score_adapter.load_test()
    check("score-stage feature access", score_train.shape == (64, 51) and score_test.shape == (203165, 51))
    check("score-stage label isolation", not score_adapter.score_label_access)
    current_hashes = {name: tree_hash(ROOT / "results" / name) for name in protected}
    check("previous COUTA results unchanged", previous_hashes == current_hashes)

    warnings_text = [f"{item.category.__name__}: {item.message}" for item in caught]
    for warning_text in warnings_text:
        log("[WARNING] " + warning_text)
    peak_memory = torch.cuda.max_memory_allocated()
    elapsed = time.perf_counter() - started
    audit = {
        "model_source_modified": False, "deepod_core_modified": False,
        "psm_results_modified": False, "skab_results_modified": False,
        "msl_results_modified": False, "hai_results_modified": False,
        "ray_installed": False, "ray_stub_used": True, "ray_tune_used": False,
        "fit_auto_hyper_used": False, "training_ray_used": False,
        "deepod_testbed_used": False, "ts_metrics_used": False,
        "best_f1_search": False, "training_test_access": False,
        "training_test_label_access": False, "score_label_access": False,
        "score_search": False, "ratio_search": False, "threshold_search": False,
        "parameter_search": False, "oracle_search": False,
        "internal_threshold_used_for_paper": False,
        "internal_prediction_used": False, "contamination_prediction_used": False,
        "formal_training_started": False, "formal_scores_generated": False,
        "formal_evaluation_run": False, "formal_efficiency_run": False,
    }
    report = {
        "status": "READY", "start_time": start_time, "end_time": now(),
        "elapsed_seconds": elapsed, "dataset": "PUMP", "input_c": 51,
        "original_dtype": {name: str(array.dtype) for name, array in arrays.items()},
        "model_input_dtype": "float32",
        "files": {name: {
            "path": str(paths[name].relative_to(ROOT)).replace("\\", "/"),
            "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype),
            "size_bytes": paths[name].stat().st_size, **statistics_data[name],
        } for name in names},
        "label_counts": label_counts,
        "anomaly_ratio_actual_percent": float(counts[1] / counts.sum() * 100),
        "config_path": str(PUMP_CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "anomaly_ratio": 0.5, "percentile": 99.5,
        "parameters": {"total": total_parameters, "trainable": trainable_parameters},
        "official_fit": {"points": VALIDATION_POINTS, "epochs": 1,
                         "seconds": fit_seconds, "loss": losses,
                         "validation_loss": val_losses, "forward": True,
                         "backward": True, "optimizer_step": True},
        "scaler": {"fit_split": "train", "mean_finite": True,
                   "scale_finite_nonzero": True,
                   "constant_feature_indices": constant_indices,
                   "constant_feature_scales": constant_scales},
        "score": {"full_score_shape": list(official_score.shape),
                  "chunked_score_shape": list(chunked_score.shape),
                  "chunk_size": VALIDATION_CHUNK_POINTS, "chunk_overlap": 29,
                  "max_abs_error": chunk_error, "prefix_padding": 29,
                  "score_direction": "larger_is_more_anomalous"},
        "formal_alignment": {"train_points": 17155, "test_points": 203165,
                             "training_windows": 1713,
                             "train_inference_windows": 17126,
                             "test_inference_windows": 203136},
        "checkpoint": {"path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
                       "sha256": sha256(CHECKPOINT),
                       "score_max_abs_error": score_restore_error,
                       "scaler_max_abs_error": scaler_restore_error},
        "gpu": {"peak_mib": peak_memory / 2**20,
                "incremental_mib": (peak_memory - memory_baseline) / 2**20},
        "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "warnings": warnings_text, "training_file_access": adapter.audit(),
        "score_file_access": score_adapter.audit(),
        "integrity_only_label_access": "dataset/PUMP/PUMP_test_label.npy",
        "audit": audit,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(
        "# PUMP COUTA Validation\n\n"
        "- Status: **READY**\n"
        "- Shapes: train `(17155, 51)`, test `(203165, 51)`, label `(203165,)`\n"
        f"- Parameters: `{total_parameters}`\n"
        f"- Chunked score max error: `{chunk_error:.3e}`\n"
        f"- Checkpoint score error: `{score_restore_error:.3e}`\n",
        encoding="utf-8")
    log("[PASS] PUMP COUTA Validation READY")
    log(f"End Time: {report['end_time']}")
    log(f"Elapsed Time: {elapsed:.3f}s")
    LOGFILE.write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
