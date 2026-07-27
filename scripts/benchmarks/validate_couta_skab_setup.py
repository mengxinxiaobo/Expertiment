#!/usr/bin/env python3
"""Train-only, one-epoch validation of official DeepOD COUTA on SKAB."""

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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.couta_dataset_adapter import (
    RAY_AUDIT, SKAB_CONFIG_PATH, SKABCOUTADataAdapter, construct_official_couta,
    import_official_couta, load_skab_couta_config, make_bundle,
    restore_from_bundle, sha256,
)

OUT = ROOT / "results" / "SKAB_COUTA_RESULTS" / "Validation"
JSON_PATH = OUT / "validation_report.json"
MD_PATH = OUT / "validation_report.md"
LOG_PATH = OUT / "validation.log"
CKPT_PATH = OUT / "temporary_checkpoint.pt"
POINTS = 2048
LOG_LINES: list[str] = []


def now() -> str: return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
def emit(message: str = "") -> None:
    print(message, flush=True); LOG_LINES.append(message)
def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    emit(f"[{status}] {name}{': ' + detail if detail else ''}")
    if not condition: raise RuntimeError(f"Validation failed: {name}: {detail}")
def tree_hash(path: Path) -> str:
    h = hashlib.sha256()
    if path.exists():
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(str(file.relative_to(path)).encode()); h.update(bytes.fromhex(sha256(file)))
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start_iso, started = now(), time.perf_counter()
    emit("=" * 72); emit("Stage: SKAB COUTA setup validation"); emit(f"Start Time: {start_iso}")
    psm_before = tree_hash(ROOT / "results" / "PSM_COUTA_RESULTS")
    config = load_skab_couta_config()

    # Integrity-only dataset audit; labels are never passed to model/scaler/config selection.
    data_dir = ROOT / "dataset" / "SKAB"
    files = {name: data_dir / name for name in
             ("SKAB_train.npy", "SKAB_test.npy", "SKAB_test_label.npy")}
    arrays = {name: np.load(path, allow_pickle=False) for name, path in files.items()}
    train_full, test_full, label_full = arrays.values()
    data_stats = {name: {"nan_count": int(np.isnan(array).sum()),
                         "inf_count": int(np.isinf(array).sum())}
                  for name, array in arrays.items()}
    check("actual files", all(p.is_file() for p in files.values()))
    check("train shape", train_full.shape == (12450, 8), str(train_full.shape))
    check("test shape", test_full.shape == (5710, 8), str(test_full.shape))
    check("label shape", label_full.shape == (5710, 1), str(label_full.shape))
    check("finite data", all(np.isfinite(a).all() for a in arrays.values()))
    unique, counts = np.unique(label_full, return_counts=True)
    check("binary labels", set(unique.tolist()) == {0, 1}, str(dict(zip(unique.tolist(), counts.tolist()))))
    check("timeline alignment", test_full.shape[0] == label_full.shape[0])
    del test_full, label_full, arrays

    check("Ray absent", not RAY_AUDIT["ray_installed"])
    COUTA, _ = import_official_couta()
    check("official COUTA import", COUTA.__module__ == "deepod.models.time_series.couta")
    from ray import tune
    blocked = False
    try: tune.choice([1, 2])
    except RuntimeError: blocked = True
    check("Ray Tune fail-closed", blocked and RAY_AUDIT["ray_tune_attempt_blocked"])

    adapter = SKABCOUTADataAdapter("validation")
    train = adapter.load_train(limit=POINTS)
    scaler, scaled = adapter.fit_train_scaler(train)
    adapter.assert_training_isolated()
    check("train-only scaler", scaled.dtype == np.float32 and np.isfinite(scaled).all())
    check("CUDA available", torch.cuda.is_available())
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    model = construct_official_couta(config, "cuda", epochs=1)
    fit_output = io.StringIO()
    caught: list[warnings.WarningMessage]
    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(fit_output):
        warnings.simplefilter("always")
        result = model.fit(scaled)
    fit_seconds = time.perf_counter() - fit_started
    for line in fit_output.getvalue().splitlines(): emit(line)
    losses = [float(x) for x in re.findall(r"(?:^|\s)loss:\s*([0-9eE+.-]+)", fit_output.getvalue())]
    val_losses = [float(x) for x in re.findall(r"val_loss:\s*([0-9eE+.-]+)", fit_output.getvalue())]
    check("official fit return", result is None)
    check("finite training loss", bool(losses) and all(math.isfinite(x) for x in losses), str(losses))
    check("finite validation loss", bool(val_losses) and all(math.isfinite(x) for x in val_losses), str(val_losses))
    check("finite center c", bool(torch.isfinite(model.c).all()))
    total = sum(p.numel() for p in model.net.parameters())
    trainable = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    check("actual parameters", total == trainable == 2081, f"{total}/{trainable}")

    score = np.asarray(model.decision_function(scaled))
    check("decision_function finite", score.shape == (POINTS,) and np.isfinite(score).all())
    check("prefix alignment", np.all(score[:29] == 0), "prefix_padding=29")
    bundle = make_bundle(model, scaler, config)
    bundle.update({"validation_epochs": 1, "formal_checkpoint": False})
    torch.save(bundle, CKPT_PATH)
    with warnings.catch_warnings(record=True) as restore_warnings:
        warnings.simplefilter("always")
        restored, restored_scaler = restore_from_bundle(
            torch.load(CKPT_PATH, map_location="cuda", weights_only=False), "cuda")
    caught.extend(restore_warnings)
    restored_scaled = np.asarray(restored_scaler.transform(train), dtype=np.float32)
    restored_score = np.asarray(restored.decision_function(restored_scaled))
    score_error = float(np.max(np.abs(score - restored_score)))
    scaler_error = float(np.max(np.abs(scaled - restored_scaled)))
    check("checkpoint restoration", score_error <= 1e-7 and scaler_error <= 1e-7,
          f"score_error={score_error:.3e}, scaler_error={scaler_error:.3e}")
    peak = torch.cuda.max_memory_allocated()
    psm_after = tree_hash(ROOT / "results" / "PSM_COUTA_RESULTS")
    check("PSM results unchanged", psm_before == psm_after)

    elapsed = time.perf_counter() - started
    warning_text = [f"{w.category.__name__}: {w.message}" for w in caught]
    for warning in warning_text: emit(f"[WARNING] {warning}")
    audit = {
        "model_source_modified": False, "deepod_core_modified": False,
        "psm_results_modified": False, "ray_installed": False,
        "ray_stub_used": True, "ray_tune_used": False,
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
        "status": "READY", "start_time": start_iso, "end_time": now(),
        "elapsed_seconds": elapsed, "dataset": "SKAB", "input_c": 8,
        "files": {k: {"path": str(v.relative_to(ROOT)).replace("\\", "/"),
                      "size_bytes": v.stat().st_size,
                      "shape": list(np.load(v, mmap_mode="r").shape),
                      "dtype": str(np.load(v, mmap_mode="r").dtype),
                      **data_stats[k]} for k,v in files.items()},
        "label_unique_counts": dict(zip(map(str, unique.tolist()), counts.tolist())),
        "config_path": str(SKAB_CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "anomaly_ratio": 0.5, "percentile": 99.5,
        "parameters": {"total": total, "trainable": trainable},
        "official_fit": {"points": POINTS, "epochs": 1, "seconds": fit_seconds,
                         "loss": losses, "val_loss": val_losses},
        "score": {"definition": "sum((rep-c)^2)+sum((rep_dup-c)^2)",
                  "shape": list(score.shape), "finite": True, "prefix_padding": 29},
        "formal_alignment": {"train_points": 12450, "train_windows": 12421,
                             "test_points": 5710, "test_windows": 5681},
        "checkpoint": {"path": str(CKPT_PATH.relative_to(ROOT)).replace("\\", "/"),
                       "sha256": sha256(CKPT_PATH), "score_max_abs_error": score_error,
                       "scaler_max_abs_error": scaler_error},
        "gpu": {"peak_mib": peak / 2**20, "incremental_mib": (peak-baseline)/2**20},
        "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "warnings": warning_text, "training_file_access": adapter.audit(),
        "integrity_only_label_access": str(files["SKAB_test_label.npy"].relative_to(ROOT)).replace("\\", "/"),
        "audit": audit,
    }
    JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = f"""# SKAB COUTA Validation\n\n- Status: **READY**\n- Shapes: train `(12450, 8)`, test `(5710, 8)`, label `(5710, 1)`\n- Parameters: `{total}` total / `{trainable}` trainable\n- Official fit: 1 epoch on {POINTS} train-only points, {fit_seconds:.3f}s\n- Score alignment: {score.shape[0]} points, first 29 zero, PASS\n- Checkpoint restoration max error: `{score_error:.3e}`\n- GPU peak: `{peak/2**20:.3f} MiB`\n- Formal training/evaluation/efficiency: not run\n"""
    MD_PATH.write_text(md, encoding="utf-8")
    emit("[PASS] SKAB COUTA Validation READY")
    emit(f"End Time: {report['end_time']}"); emit(f"Elapsed Time: {elapsed:.3f}s")
    LOG_PATH.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
