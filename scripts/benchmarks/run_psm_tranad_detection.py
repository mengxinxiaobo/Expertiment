#!/usr/bin/env python3
"""Shared frozen TranAD training and detection implementation.

The unmodified official TranAD model class is imported through the external
adapter. Training can access the registered train split only. Test labels load only in
the final evaluation stage, after the checkpoint and both score files exist.
"""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asca_ad.model import AdaptiveSparseAnchorSolverV4
from scripts.benchmarks.adapters.tranad_dataset_adapter import (
    TranADPSMDataAdapter,
    import_official_tranad_class,
    load_tranad_psm_config,
)


OUTPUT_ROOT = ROOT / "results" / "PSM_TRANAD_RESULTS"
DETECTION_ROOT = OUTPUT_ROOT / "Detection"
SCORES_ROOT = OUTPUT_ROOT / "scores"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_ROOT / "TranAD_PSM_state_dict.pt"
TRAIN_SCORE_PATH = SCORES_ROOT / "train_score.npy"
TEST_SCORE_PATH = SCORES_ROOT / "test_score.npy"
COMPARISON_CSV = DETECTION_ROOT / "comparison_detection.csv"
COMPARISON_JSON = DETECTION_ROOT / "comparison_detection.json"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
LABEL_PATH = ROOT / "dataset" / "PSM" / "PSM_test_label.npy"
HEARTBEAT_SECONDS = 300.0
DATASET_NAME = "PSM"
EXPECTED_TRAIN_POINTS = 132481
EXPECTED_TEST_POINTS = 87841


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace a previous formal TranAD run.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate checkpoint resume and one label-free score batch only.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def stage_start(name: str) -> float:
    print("=" * 72, flush=True)
    print(name, flush=True)
    print(f"Start Time: {now_iso()}", flush=True)
    print("=" * 72, flush=True)
    return time.perf_counter()


def stage_end(name: str, started: float) -> float:
    elapsed = time.perf_counter() - started
    print(f"{name} Finished", flush=True)
    print(f"End Time: {now_iso()}", flush=True)
    print(f"Elapsed Time: {format_duration(elapsed)}", flush=True)
    return elapsed


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_run_state(overwrite: bool) -> bool:
    """Return True only for an unambiguous checkpoint-only resume."""

    for directory in (DETECTION_ROOT, SCORES_ROOT, CHECKPOINT_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return False
    completed = [
        path
        for path in (COMPARISON_CSV, COMPARISON_JSON, PROTOCOL_PATH)
        if path.exists()
    ]
    if completed:
        raise FileExistsError(
            "Completed TranAD artifacts already exist; refusing overwrite: "
            f"{[str(path) for path in completed]}"
        )
    partial_scores = [
        path for path in (TRAIN_SCORE_PATH, TEST_SCORE_PATH) if path.exists()
    ]
    if partial_scores:
        raise RuntimeError(
            "Partial score artifacts exist and cannot be mixed automatically: "
            f"{[str(path) for path in partial_scores]}"
        )
    return CHECKPOINT_PATH.exists()


def construct_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_class = import_official_tranad_class(
        float(config["training"]["learning_rate"])
    )
    model = model_class(int(config["model"]["input_channels"])).float()
    if int(model.n_window) != int(config["model"]["window"]):
        raise RuntimeError("Official TranAD window differs from frozen protocol")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected_parameters = int(config["model"]["expected_parameters"])
    if parameters != expected_parameters or trainable != expected_parameters:
        raise RuntimeError(f"Unexpected TranAD parameters: {parameters}/{trainable}")
    return model.to(device)


def tranad_loss(
    model: torch.nn.Module,
    batch: torch.Tensor,
    epoch_number: int,
) -> torch.Tensor:
    source = batch.permute(1, 0, 2)
    target = source[-1].unsqueeze(0)
    first, second = model(source, target)
    first_loss = F.mse_loss(first, target, reduction="mean")
    second_loss = F.mse_loss(second, target, reduction="mean")
    first_weight = 1.0 / float(epoch_number)
    return first_weight * first_loss + (1.0 - first_weight) * second_loss


def train_model(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = stage_start(
        f"Stage 1: Train TranAD on {config['dataset']} (train split only)"
    )
    adapter = TranADPSMDataAdapter("train")
    loader = adapter.loader("train", shuffle=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    epochs = int(config["training"]["epochs"])
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        heartbeat_started = epoch_started
        loss_sum = 0.0
        sample_count = 0
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device=device, dtype=torch.float32, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            loss = tranad_loss(model, batch, epoch)
            loss.backward()
            optimizer.step()
            current_batch = int(batch.shape[0])
            loss_sum += float(loss.detach().item()) * current_batch
            sample_count += current_batch

            current_time = time.perf_counter()
            if current_time - heartbeat_started >= HEARTBEAT_SECONDS:
                elapsed = current_time - training_started
                completed_fraction = (
                    ((epoch - 1) * len(loader) + batch_index)
                    / float(epochs * len(loader))
                )
                eta = elapsed * (1.0 - completed_fraction) / completed_fraction
                print(
                    f"[Heartbeat] Current Time={now_iso()} "
                    f"Elapsed={format_duration(elapsed)} "
                    f"Epoch={epoch}/{epochs} Batch={batch_index}/{len(loader)} "
                    f"Current Loss={loss.detach().item():.8f} "
                    f"Estimated Remaining={format_duration(eta)}",
                    flush=True,
                )
                heartbeat_started = current_time

        epoch_time = time.perf_counter() - epoch_started
        elapsed = time.perf_counter() - training_started
        mean_loss = loss_sum / sample_count
        mean_epoch_time = elapsed / epoch
        remaining = mean_epoch_time * (epochs - epoch)
        record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "epoch_seconds": epoch_time,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": remaining,
        }
        history.append(record)
        print(
            f"Epoch: {epoch}/{epochs} | Train Loss: {mean_loss:.8f} | "
            f"Epoch Time: {format_duration(epoch_time)} | "
            f"Elapsed Time: {format_duration(elapsed)} | "
            f"Estimated Remaining Time: {format_duration(remaining)}",
            flush=True,
        )

    adapter.assert_training_isolated()
    adapter.assert_label_free()
    training_elapsed = stage_end("Stage 1: TranAD Training", started)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "completed_epochs": epochs,
        "history": history,
        "training_elapsed_seconds": training_elapsed,
        "training_access_audit": adapter.audit(),
    }
    temporary = CHECKPOINT_PATH.with_suffix(".pt.tmp")
    torch.save(checkpoint_payload, temporary)
    temporary.replace(CHECKPOINT_PATH)
    print(f"checkpoint={CHECKPOINT_PATH}", flush=True)
    return {
        "history": history,
        "elapsed_seconds": training_elapsed,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(CHECKPOINT_PATH),
    }, adapter.audit()


def load_completed_training_checkpoint(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and load a checkpoint-only interrupted formal run."""

    started = stage_start("Resume: Validate completed training checkpoint")
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "config",
        "completed_epochs",
        "history",
        "training_elapsed_seconds",
        "training_access_audit",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"Resume checkpoint missing fields: {missing}")
    checkpoint_config = payload["config"]
    if checkpoint_config.get("dataset") != config["dataset"]:
        raise RuntimeError("Resume checkpoint dataset mismatch")
    for section in ("model", "training", "preprocessing"):
        if checkpoint_config.get(section) != config.get(section):
            raise RuntimeError(f"Resume checkpoint {section} configuration mismatch")
    epochs = int(config["training"]["epochs"])
    if int(payload["completed_epochs"]) != epochs:
        raise RuntimeError("Resume checkpoint does not contain all frozen epochs")
    history = payload["history"]
    if len(history) != epochs or [item["epoch"] for item in history] != list(
        range(1, epochs + 1)
    ):
        raise RuntimeError("Resume checkpoint epoch history is incomplete")

    audit = payload["training_access_audit"]
    expected_train = (
        f"dataset/{config['dataset']}/{config['dataset']}_train.npy"
    )
    if audit.get("files_accessed") != [expected_train]:
        raise RuntimeError("Resume checkpoint training-file audit failed")
    if audit.get("scaler_fit_files") != [expected_train]:
        raise RuntimeError("Resume checkpoint scaler audit failed")
    for flag in (
        "training_test_feature_access",
        "training_label_access",
        "test_label_access",
    ):
        if audit.get(flag) is not False:
            raise RuntimeError(f"Resume checkpoint audit flag failed: {flag}")

    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    checkpoint_hash = sha256(CHECKPOINT_PATH)
    stage_end("Resume: Checkpoint Validation", started)
    print("training_checkpoint=complete; training=skipped", flush=True)
    return {
        "history": history,
        "elapsed_seconds": float(payload["training_elapsed_seconds"]),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_hash,
        "resumed_after_completed_training": True,
    }, audit


@torch.no_grad()
def collect_scores(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split: str,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    started = time.perf_counter()
    heartbeat_started = started
    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=False)
        source = batch.permute(1, 0, 2)
        target = source[-1].unsqueeze(0)
        _first, second = model(source, target)
        score = (second - target).square()[0].mean(dim=1)
        parts.append(score.detach().cpu().numpy().astype(np.float32, copy=False))
        current_time = time.perf_counter()
        if current_time - heartbeat_started >= HEARTBEAT_SECONDS:
            elapsed = current_time - started
            fraction = batch_index / float(len(loader))
            eta = elapsed * (1.0 - fraction) / fraction
            print(
                f"[Score][{split}] current_time={now_iso()} "
                f"batch={batch_index}/{len(loader)} "
                f"elapsed={format_duration(elapsed)} "
                f"estimated_remaining={format_duration(eta)}",
                flush=True,
            )
            heartbeat_started = current_time
    scores = np.concatenate(parts).reshape(-1)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"TranAD {split} score contains NaN or Inf")
    return scores


def generate_scores(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    started = stage_start("Stage 2: Generate label-free TranAD scores")
    adapter = TranADPSMDataAdapter("score")
    train_score = collect_scores(
        model, adapter.loader("train", shuffle=False), device, "train"
    )
    test_score = collect_scores(
        model, adapter.loader("test", shuffle=False), device, "test"
    )
    adapter.assert_label_free()
    expected_train = int(config["expected_shapes"]["train"][0])
    expected_test = int(config["expected_shapes"]["test"][0])
    if train_score.shape != (expected_train,):
        raise RuntimeError(f"Unexpected train score shape: {train_score.shape}")
    if test_score.shape != (expected_test,):
        raise RuntimeError(f"Unexpected test score shape: {test_score.shape}")
    save_npy_atomic(TRAIN_SCORE_PATH, train_score)
    save_npy_atomic(TEST_SCORE_PATH, test_score)
    elapsed = stage_end("Stage 2: Score Generation", started)
    print(f"train_score.shape={train_score.shape}", flush=True)
    print(f"test_score.shape={test_score.shape}", flush=True)
    return {
        "train_score_shape": list(train_score.shape),
        "test_score_shape": list(test_score.shape),
        "elapsed_seconds": elapsed,
        "access_audit": adapter.audit(),
        "test_label_access": False,
    }


def save_npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


@torch.no_grad()
def resume_preflight(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    """Exercise the exact score forward on one train and one test batch."""

    _training, audit = load_completed_training_checkpoint(model, config)
    if audit.get("test_label_access") is not False:
        raise RuntimeError("Preflight checkpoint label audit failed")
    adapter = TranADPSMDataAdapter("score")
    model.eval()
    for split in ("train", "test"):
        batch = next(iter(adapter.loader(split, shuffle=False))).to(
            device=device, dtype=torch.float32
        )
        source = batch.permute(1, 0, 2)
        target = source[-1].unsqueeze(0)
        _first, second = model(source, target)
        score = (second - target).square()[0].mean(dim=1)
        if tuple(score.shape) != (int(batch.shape[0]),):
            raise RuntimeError(f"Preflight {split} score shape failed")
        if not torch.isfinite(score).all():
            raise RuntimeError(f"Preflight {split} score is not finite")
        print(
            f"preflight_{split}=PASS batch={tuple(batch.shape)} "
            f"score={tuple(score.shape)}",
            flush=True,
        )
    adapter.assert_label_free()
    print("preflight_training=False label_access=False artifacts_written=False", flush=True)


def binary_metrics(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    started = stage_start("Stage 3: Fixed percentile evaluation")
    train_score = np.load(TRAIN_SCORE_PATH, allow_pickle=False).reshape(-1)
    test_score = np.load(TEST_SCORE_PATH, allow_pickle=False).reshape(-1)

    # This is the first and only stage that opens the registered test-label file.
    labels = np.load(LABEL_PATH, allow_pickle=False).reshape(-1).astype(np.int64)
    if labels.shape != test_score.shape:
        raise RuntimeError(
            f"Label/score length mismatch: {labels.shape} != {test_score.shape}"
        )
    if not set(np.unique(labels)).issubset({0, 1}):
        raise RuntimeError(f"{config['dataset']} labels are not binary")

    percentile = float(config["evaluation"]["percentile"])
    threshold = float(
        np.percentile(np.concatenate((train_score, test_score)), percentile)
    )
    raw_prediction = (test_score > threshold).astype(np.int64)
    pa_prediction = AdaptiveSparseAnchorSolverV4._point_adjust(
        raw_prediction.copy(), labels
    )
    raw = binary_metrics(raw_prediction, labels)
    pa = binary_metrics(pa_prediction, labels)
    np.save(DETECTION_ROOT / "pred_raw.npy", raw_prediction, allow_pickle=False)
    np.save(DETECTION_ROOT / "pred_pa.npy", pa_prediction, allow_pickle=False)

    result = {
        "model": "TranAD",
        "dataset": config["dataset"],
        "threshold": threshold,
        "anomaly_ratio": float(config["evaluation"]["anomaly_ratio"]),
        "percentile": percentile,
        "threshold_rule": (
            f"percentile(concat(train_score,test_score),{percentile:g})"
        ),
        "raw": raw,
        "pa": pa,
        "point_adjustment": (
            "asca_ad.model.AdaptiveSparseAnchorSolverV4._point_adjust; "
            "existing PPLAD-compatible fixed segment adjustment"
        ),
        "evaluated_points": int(labels.size),
        "label_access_stage": "final_evaluation_only",
        "label_sha256": sha256(LABEL_PATH),
    }
    write_json(COMPARISON_JSON, result)
    fields = (
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "PA-Accuracy",
        "PA-Precision",
        "PA-Recall",
        "PA-F1",
        "Threshold",
    )
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "Model": "TranAD",
                "Accuracy": raw["accuracy"],
                "Precision": raw["precision"],
                "Recall": raw["recall"],
                "F1": raw["f1"],
                "PA-Accuracy": pa["accuracy"],
                "PA-Precision": pa["precision"],
                "PA-Recall": pa["recall"],
                "PA-F1": pa["f1"],
                "Threshold": threshold,
            }
        )
    elapsed = stage_end("Stage 3: Evaluation", started)
    print(
        f"RAW Precision={raw['precision']:.6f} Recall={raw['recall']:.6f} "
        f"F1={raw['f1']:.6f}",
        flush=True,
    )
    print(
        f"PA Accuracy={pa['accuracy']:.6f} Precision={pa['precision']:.6f} "
        f"Recall={pa['recall']:.6f} F1={pa['f1']:.6f}",
        flush=True,
    )
    result["elapsed_seconds"] = elapsed
    return result


def write_protocol(
    config: dict[str, Any],
    device: torch.device,
    training: dict[str, Any],
    training_audit: dict[str, Any],
    scoring: dict[str, Any],
    evaluation: dict[str, Any],
    total_elapsed: float,
) -> None:
    parameters = 57273
    dataset_name = config["dataset"]
    dataset_root = ROOT / "dataset" / dataset_name
    payload = {
        "dataset": dataset_name,
        "seed": int(config["seed"]),
        "dtype": config["dtype"],
        "dataset_files": {
            "train": {
                "path": f"dataset/{dataset_name}/{dataset_name}_train.npy",
                "shape": config["expected_shapes"]["train"],
                "sha256": sha256(dataset_root / f"{dataset_name}_train.npy"),
            },
            "test": {
                "path": f"dataset/{dataset_name}/{dataset_name}_test.npy",
                "shape": config["expected_shapes"]["test"],
                "sha256": sha256(dataset_root / f"{dataset_name}_test.npy"),
            },
            "label": {
                "path": (
                    f"dataset/{dataset_name}/{dataset_name}_test_label.npy"
                ),
                "access_stage": "final_evaluation_only",
                "sha256": evaluation["label_sha256"],
            },
        },
        "preprocessing": config["preprocessing"],
        "model": {
            **config["model"],
            "parameters": parameters,
            "source_modified": False,
            "runtime_compatibility": (
                "external adapter ignores PyTorch >=2 causal-hint keywords; "
                "official Transformer computation unchanged"
            ),
        },
        "training": {
            **config["training"],
            **training,
            "access_audit": training_audit,
        },
        "scoring": scoring,
        "evaluation": {
            **config["evaluation"],
            "threshold": evaluation["threshold"],
            "label_access_stage": "final_evaluation_only",
        },
        "audit": {
            "training_test_feature_access": False,
            "training_label_access": False,
            "score_stage_label_access": False,
            "test_label_access_before_evaluation": False,
            "pot_called": False,
            "spot_called": False,
            "bf_search_called": False,
            "score_search": False,
            "ratio_search": False,
            "threshold_search": False,
            "parameter_search": False,
            "oracle_search": False,
            "model_source_modified": False,
        },
        "checkpoint": {
            "path": training["checkpoint"],
            "sha256": training["checkpoint_sha256"],
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "total_elapsed_seconds": total_elapsed,
    }
    write_json(PROTOCOL_PATH, payload)


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    config = load_tranad_psm_config()
    set_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("Formal TranAD experiment requires the current CUDA GPU")
    device = torch.device("cuda:0")
    print(f"{config['dataset']} TranAD fixed-protocol formal experiment", flush=True)
    print(f"training_data={config['dataset']}_train.npy only", flush=True)
    print("training_test_access=False", flush=True)
    print("training_label_access=False", flush=True)
    print("score_stage_label_access=False", flush=True)
    print("POT=disabled SPOT=disabled bf_search=disabled", flush=True)
    print("score_search=False ratio_search=False threshold_search=False", flush=True)
    print("parameter_search=False oracle_search=False", flush=True)
    print(f"device={device} seed={config['seed']} dtype={config['dtype']}", flush=True)

    model = construct_model(config, device)
    if args.preflight:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError("Preflight requires a completed checkpoint")
        resume_preflight(model, config, device)
        return

    resume_checkpoint = prepare_run_state(args.overwrite)
    if resume_checkpoint:
        training, training_audit = load_completed_training_checkpoint(model, config)
    else:
        training, training_audit = train_model(model, config, device)
    scoring = generate_scores(model, config, device)
    evaluation = evaluate(config)
    total_elapsed = time.perf_counter() - total_started
    write_protocol(
        config,
        device,
        training,
        training_audit,
        scoring,
        evaluation,
        total_elapsed,
    )
    print("=" * 72, flush=True)
    print(f"{config['dataset']} TranAD formal detection completed", flush=True)
    print(f"Total Elapsed Time: {format_duration(total_elapsed)}", flush=True)
    print(f"comparison_csv={COMPARISON_CSV}", flush=True)
    print(f"protocol={PROTOCOL_PATH}", flush=True)


if __name__ == "__main__":
    main()
