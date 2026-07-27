#!/usr/bin/env python3
"""Re-train ASCA-AD V4 on SKAB/MSL/PSM without touching paper artifacts.

This is a reproducibility verification experiment.  Both the existing and the
freshly trained checkpoint are evaluated with the same frozen per-dataset
protocol.  No score mode, anomaly ratio, or threshold search is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_ROOT = ROOT / "results" / "ASCA_RETRAIN_VERIFY"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "ASCA_RETRAIN_VERIFY"
ENTRY_SCRIPT = Path(__file__).resolve()
SEED = 42

LOCAL_LAGS = [1, 2, 3, 4, 5, 6, 7, 8]
GLOBAL_LAGS = [12, 16, 20, 24, 28, 32, 40, 48]
CHECKPOINT_NAME = (
    "{dataset}_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
    "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
)

DATASETS: dict[str, dict[str, Any]] = {
    "SKAB": {
        "shape": {"train": (12450, 8), "test": (5710, 8), "label": (5710, 1)},
        "channels": 8,
        "anomaly_ratio": 0.5,
    },
    "MSL": {
        "shape": {"train": (58317, 55), "test": (73729, 55), "label": (73729,)},
        "channels": 55,
        "anomaly_ratio": 0.8,
    },
    "PSM": {
        "shape": {"train": (132481, 25), "test": (87841, 25), "label": (87841,)},
        "channels": 25,
        "anomaly_ratio": 0.8,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=tuple(DATASETS))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Reuse both checkpoints and recompute scores/metrics without training.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_dataset(dataset: str) -> None:
    expected = DATASETS[dataset]["shape"]
    base = ROOT / "dataset" / dataset
    paths = {
        "train": base / f"{dataset}_train.npy",
        "test": base / f"{dataset}_test.npy",
        "label": base / f"{dataset}_test_label.npy",
    }
    for split, path in paths.items():
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if split == "label":
            expected_length = int(expected["test"][0])
            valid_label_shape = (
                values.size == expected_length
                and values.ndim in (1, 2)
                and (values.ndim == 1 or values.shape[1] == 1)
            )
            if not valid_label_shape:
                raise RuntimeError(
                    f"Unexpected {dataset} label shape {values.shape}; expected "
                    f"({expected_length},) or ({expected_length}, 1)"
                )
        elif tuple(values.shape) != tuple(expected[split]):
            raise RuntimeError(
                f"Unexpected {dataset} {split} shape {values.shape}; "
                f"expected {expected[split]}"
            )


def existing_checkpoint(dataset: str) -> Path:
    return ROOT / "checkpoints" / dataset / CHECKPOINT_NAME.format(dataset=dataset)


def retrained_checkpoint(dataset: str) -> Path:
    return CHECKPOINT_ROOT / dataset / CHECKPOINT_NAME.format(dataset=dataset)


def build_config(dataset: str, checkpoint_dir: Path, result_dir: Path) -> dict[str, Any]:
    item = DATASETS[dataset]
    return {
        "dataset": dataset,
        "data_path": dataset,
        "input_c": int(item["channels"]),
        "output_c": int(item["channels"]),
        "win_size": 100,
        "batch_size": 128,
        "num_epochs": 10,
        "lr": 1e-3,
        "anormly_ratio": float(item["anomaly_ratio"]),
        "index": 137,
        "mode": "train",
        "seed": SEED,
        "local_candidate_lags": LOCAL_LAGS,
        "global_candidate_lags": GLOBAL_LAGS,
        "local_topk": 2,
        "global_topk": 4,
        "selector_hidden": 8,
        "fitter_hidden": 8,
        "selector_temperature": 0.5,
        "similarity_tau": 1.0,
        "sigma_min": 0.03,
        "sigma_max": 1.5,
        "area_weight": 0.1,
        "selector_balance_weight": 0.05,
        "gap_weight": 1.0,
        "relation_input": "instance",
        "score_modes": ["total"],
        "primary_score": "total",
        "score_normalization": "official",
        "threshold_source": "original",
        "quantile_method": "exact",
        "quantile_buffer": 50000,
        # Compatibility fields required by the repository's base Solver.
        "local_size": 1,
        "global_size": [20],
        "d_model": 8,
        "loss_fuc": "MSE",
        "r": 0.5,
        "similar": "MSE",
        "rec_timeseries": True,
        "model_save_path": str(checkpoint_dir),
        "result_path": str(result_dir),
        "use_gpu": torch.cuda.is_available(),
        "use_multi_gpu": False,
        "gpu": 0,
        "devices": "0",
    }


class _LabelFreeWindowDataset(Dataset):
    """Window dataset used only while fitting ASCA; it never owns test labels."""

    def __init__(self, values: np.ndarray, win_size: int, non_overlapping: bool) -> None:
        self.values = values
        self.win_size = int(win_size)
        self.non_overlapping = bool(non_overlapping)

    def __len__(self) -> int:
        step = self.win_size if self.non_overlapping else 1
        return (self.values.shape[0] - self.win_size) // step + 1

    def __getitem__(self, index: int):
        step = self.win_size if self.non_overlapping else 1
        start = int(index) * step
        window = np.float32(self.values[start : start + self.win_size])
        # The inherited training loop expects a pair but discards its second item.
        # A constant placeholder prevents any test-label dependency.
        placeholder = np.zeros(self.win_size, dtype=np.float32)
        return window, placeholder


def label_free_loader_factory(dataset: str):
    """Build official-shaped loaders with train-only scaler fitting and no labels."""
    base = ROOT / "dataset" / dataset
    original_load = np.load
    cache: dict[str, np.ndarray] = {}

    def initialize() -> None:
        if cache:
            return
        train_raw = original_load(
            base / f"{dataset}_train.npy", allow_pickle=False
        )
        test_raw = original_load(
            base / f"{dataset}_test.npy", allow_pickle=False
        )
        train_raw = np.nan_to_num(train_raw)
        test_raw = np.nan_to_num(test_raw)
        scaler = StandardScaler()
        scaler.fit(train_raw)
        cache["train"] = scaler.transform(train_raw)
        cache["test"] = scaler.transform(test_raw)

    def get_loader_segment(
        index,
        data_path,
        batch_size,
        win_size=100,
        step=100,
        mode="train",
        dataset=dataset,
    ):
        del index, data_path, step
        initialize()
        values = cache["train"] if mode == "train" else cache["test"]
        window_dataset = _LabelFreeWindowDataset(
            values, win_size, non_overlapping=(mode == "thre")
        )
        return DataLoader(
            dataset=window_dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            num_workers=0,
            drop_last=False,
        )

    return get_loader_segment


def load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Unsupported ASCA checkpoint format: {path}")
    return payload


def metric_values(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_checkpoint(
    dataset: str,
    checkpoint: Path,
    checkpoint_type: str,
    result_dir: Path,
) -> dict[str, Any]:
    from asca_ad.model import AdaptiveSparseAnchorSolverV4

    config = build_config(dataset, checkpoint.parent, result_dir)
    runner = AdaptiveSparseAnchorSolverV4(config)
    runner.model.load_state_dict(load_payload(checkpoint)["model"], strict=True)
    runner.model.eval()

    captured: dict[str, Any] = {}
    original_thresholds = runner._stream_thresholds
    test_globals = inspect.unwrap(runner.test).__globals__
    original_combiner = test_globals["combine_all_evaluation_scores"]

    def capture_thresholds(loaders):
        values = original_thresholds(loaders)
        captured["threshold"] = float(values["total"])
        captured["threshold_samples"] = int(runner._threshold_sample_counts["total"])
        return values

    runner._stream_thresholds = capture_thresholds

    def capture_combiner(pred, gt, energy):
        # The official function performs point adjustment in place.  Preserve
        # the genuinely raw prediction before delegating to it.
        captured["raw_prediction"] = (
            np.asarray(pred).astype(np.int64).reshape(-1).copy()
        )
        captured["labels"] = np.asarray(gt).astype(np.int64).reshape(-1).copy()
        return original_combiner(pred, gt, energy)

    test_globals["combine_all_evaluation_scores"] = capture_combiner
    try:
        runner.test()
    finally:
        test_globals["combine_all_evaluation_scores"] = original_combiner

    prefix = result_dir / f"{dataset}_adaptive_anchor_v4_total"
    raw_prediction = captured["raw_prediction"]
    pa_prediction = np.loadtxt(str(prefix) + "_pred_pa.txt", dtype=np.int64).reshape(-1)
    labels = np.loadtxt(str(prefix) + "_label.txt", dtype=np.int64).reshape(-1)
    if not np.array_equal(labels, captured["labels"]):
        raise RuntimeError(f"{dataset} captured and saved labels do not match")
    np.save(result_dir / "pred_raw.npy", raw_prediction, allow_pickle=False)
    np.save(result_dir / "pred_pa.npy", pa_prediction, allow_pickle=False)
    if not (raw_prediction.size == pa_prediction.size == labels.size):
        raise RuntimeError(f"{dataset} prediction/label lengths do not match")

    raw = metric_values(raw_prediction, labels)
    pa = metric_values(pa_prediction, labels)
    result = {
        "dataset": dataset,
        "checkpoint_type": checkpoint_type,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "parameters": int(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)),
        "score_mode": "total",
        "anomaly_ratio": float(DATASETS[dataset]["anomaly_ratio"]),
        "percentile": float(100.0 - DATASETS[dataset]["anomaly_ratio"]),
        "threshold": captured["threshold"],
        "threshold_samples": captured["threshold_samples"],
        "threshold_rule": "percentile(concat(train_score,test_score),100-anomaly_ratio)",
        "raw": raw,
        "pa": pa,
        "evaluated_points": int(labels.size),
    }
    write_json(result_dir / "metrics.json", result)
    return result


def run_worker(dataset: str, overwrite: bool) -> None:
    validate_dataset(dataset)
    set_seed()
    from asca_ad.model import AdaptiveSparseAnchorSolverV4
    import solver as base_solver

    old_checkpoint = existing_checkpoint(dataset)
    new_checkpoint = retrained_checkpoint(dataset)
    if not old_checkpoint.exists():
        raise FileNotFoundError(f"Existing checkpoint not found: {old_checkpoint}")
    original_hash_before = sha256(old_checkpoint)

    if new_checkpoint.exists() and not overwrite:
        raise FileExistsError(
            f"Verification checkpoint already exists: {new_checkpoint}. "
            "Use --overwrite only to intentionally repeat this verification run."
        )
    new_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    dataset_output = OUTPUT_ROOT / dataset
    train_result_dir = dataset_output / "training"
    train_result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"ASCA-AD V4 retrain verification: {dataset}")
    print(f"seed={SEED}, epochs=10, batch_size=128, lr=0.001, optimizer=Adam")
    print(f"window=100, score_mode=total, anomaly_ratio={DATASETS[dataset]['anomaly_ratio']}")
    print("training_test_label_access=False (guarded)")
    print("scaler_fit=train_only; test=transform_only")
    print(f"new_checkpoint={new_checkpoint}")

    config = build_config(dataset, new_checkpoint.parent, train_result_dir)
    original_loader_factory = base_solver.get_loader_segment
    original_np_load = np.load

    def guarded_np_load(file, *args, **kwargs):
        if str(file).replace("\\", "/").endswith("_test_label.npy"):
            raise RuntimeError(
                f"Training attempted to read forbidden test label file: {file}"
            )
        return original_np_load(file, *args, **kwargs)

    # Solver construction normally creates all four official loaders, whose
    # dataset constructors read test labels even for mode='train'.  Replace
    # only that data boundary during fitting; runner.train/model/loss stay the
    # repository's existing implementation.  The official loader is restored
    # before either checkpoint is evaluated.
    base_solver.get_loader_segment = label_free_loader_factory(dataset)
    np.load = guarded_np_load
    try:
        runner = AdaptiveSparseAnchorSolverV4(config)
        if Path(runner.checkpoint_path).resolve() != new_checkpoint.resolve():
            raise RuntimeError("Generated verification checkpoint path is not the frozen target")
        started = time.perf_counter()
        runner.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_seconds = float(time.perf_counter() - started)
    finally:
        np.load = original_np_load
        base_solver.get_loader_segment = original_loader_factory

    if not new_checkpoint.exists():
        raise RuntimeError(f"Training did not create checkpoint: {new_checkpoint}")

    existing_result = evaluate_checkpoint(
        dataset, old_checkpoint, "existing_checkpoint", dataset_output / "existing_checkpoint"
    )
    retrained_result = evaluate_checkpoint(
        dataset, new_checkpoint, "retrained_checkpoint", dataset_output / "retrained_checkpoint"
    )

    original_hash_after = sha256(old_checkpoint)
    if original_hash_after != original_hash_before:
        raise RuntimeError(f"Original checkpoint changed during verification: {old_checkpoint}")

    worker_result = {
        "dataset": dataset,
        "training_succeeded": True,
        "training_seconds": training_seconds,
        "training_configuration": {
            "seed": SEED,
            "epochs": 10,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "optimizer": "torch.optim.Adam",
            "learning_rate_schedule": "lr(epoch)=initial_lr*0.5^(epoch-1)",
            "gradient_clipping_max_norm": 5.0,
            "window_size": 100,
            "loss": "fit_loss + 0.1*area_loss + 0.05*anchor_coverage_loss",
            "input_dtype": "float32",
            "standard_scaler": "fit on train only; test transform only",
            "training_test_label_access": False,
            "training_label_policy": (
                "label-free loader plus hard failure on any *_test_label.npy read"
            ),
            "test_label_access": False,
            "test_label_access_guard": "np.load raises on *_test_label.npy during training",
            "local_lags": LOCAL_LAGS,
            "global_lags": GLOBAL_LAGS,
            "local_topk": 2,
            "global_topk": 4,
            "score_mode": "total",
        },
        "original_checkpoint_unchanged": True,
        "existing_metrics_reused": False,
        "existing_checkpoint_scores_recomputed": True,
        "existing_checkpoint": existing_result,
        "retrained_checkpoint": retrained_result,
        "pa_f1_difference_retrained_minus_existing": float(
            retrained_result["pa"]["f1"] - existing_result["pa"]["f1"]
        ),
    }
    write_json(dataset_output / "verification.json", worker_result)
    print(
        f"{dataset}: existing_PA_F1={existing_result['pa']['f1']:.6f}, "
        f"retrained_PA_F1={retrained_result['pa']['f1']:.6f}, "
        f"difference={worker_result['pa_f1_difference_retrained_minus_existing']:+.6f}"
    )


def reevaluate_worker(dataset: str) -> None:
    """Recompute both checkpoint results while preserving the completed training record."""
    validate_dataset(dataset)
    set_seed()
    old_checkpoint = existing_checkpoint(dataset)
    new_checkpoint = retrained_checkpoint(dataset)
    verification_path = OUTPUT_ROOT / dataset / "verification.json"
    for path in (old_checkpoint, new_checkpoint, verification_path):
        if not path.exists():
            raise FileNotFoundError(f"Evaluation-only input not found: {path}")

    previous = json.loads(verification_path.read_text(encoding="utf-8"))
    old_hash_before = sha256(old_checkpoint)
    new_hash_before = sha256(new_checkpoint)
    dataset_output = OUTPUT_ROOT / dataset
    existing_result = evaluate_checkpoint(
        dataset, old_checkpoint, "existing_checkpoint", dataset_output / "existing_checkpoint"
    )
    retrained_result = evaluate_checkpoint(
        dataset, new_checkpoint, "retrained_checkpoint", dataset_output / "retrained_checkpoint"
    )
    if sha256(old_checkpoint) != old_hash_before or sha256(new_checkpoint) != new_hash_before:
        raise RuntimeError(f"{dataset} checkpoint changed during evaluation-only run")

    previous.update(
        {
            "original_checkpoint_unchanged": True,
            "retrained_checkpoint_unchanged_during_evaluation": True,
            "existing_metrics_reused": False,
            "existing_checkpoint_scores_recomputed": True,
            "retrained_checkpoint_scores_recomputed": True,
            "evaluation_refreshed_without_training": True,
            "existing_checkpoint": existing_result,
            "retrained_checkpoint": retrained_result,
            "pa_f1_difference_retrained_minus_existing": float(
                retrained_result["pa"]["f1"] - existing_result["pa"]["f1"]
            ),
        }
    )
    write_json(verification_path, previous)
    print(
        f"{dataset}: evaluation refreshed; "
        f"existing_PA_F1={existing_result['pa']['f1']:.6f}, "
        f"retrained_PA_F1={retrained_result['pa']['f1']:.6f}"
    )


def environment_info() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def aggregate(run_workers: bool, overwrite: bool, evaluation_only: bool = False) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if run_workers:
        for dataset in DATASETS:
            completed = OUTPUT_ROOT / dataset / "verification.json"
            if completed.exists() and not overwrite and not evaluation_only:
                print(f"[{dataset}] complete verification exists; skip: {completed}")
                continue
            command = [sys.executable, str(ENTRY_SCRIPT), "--worker", dataset]
            if evaluation_only:
                command.append("--evaluation-only")
            if overwrite:
                command.append("--overwrite")
            subprocess.run(command, cwd=ROOT, check=True)

    verifications = []
    for dataset in DATASETS:
        path = OUTPUT_ROOT / dataset / "verification.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing verification result: {path}")
        verifications.append(json.loads(path.read_text(encoding="utf-8")))

    rows: list[dict[str, Any]] = []
    for verification in verifications:
        for checkpoint_type in ("existing_checkpoint", "retrained_checkpoint"):
            item = verification[checkpoint_type]
            rows.append(
                {
                    "Dataset": item["dataset"],
                    "Checkpoint_Type": checkpoint_type,
                    "Precision": item["raw"]["precision"],
                    "Recall": item["raw"]["recall"],
                    "F1": item["raw"]["f1"],
                    "PA-Precision": item["pa"]["precision"],
                    "PA-Recall": item["pa"]["recall"],
                    "PA-F1": item["pa"]["f1"],
                    "Threshold": item["threshold"],
                }
            )

    fields = list(rows[0])
    csv_path = OUTPUT_ROOT / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    protocol = {
        "experiment": "ASCA-AD V4 retrain reproducibility verification",
        "replaces_paper_results": False,
        "datasets": {name: value for name, value in DATASETS.items()},
        "training": {
            "seed": SEED,
            "epochs": 10,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "optimizer": "torch.optim.Adam",
            "learning_rate_schedule": "lr(epoch)=initial_lr*0.5^(epoch-1)",
            "gradient_clipping_max_norm": 5.0,
            "window_size": 100,
            "loss": "fit_loss + 0.1*area_loss + 0.05*anchor_coverage_loss",
            "input_dtype": "float32",
            "standard_scaler": "fit on train only; test transform only",
            "test_labels_used_by_forward_loss_optimizer": False,
            "test_label_file_hard_guard_by_dataset": {
                verification["dataset"]: bool(
                    verification.get("training_configuration", {}).get(
                        "test_label_access_guard"
                    )
                )
                for verification in verifications
            },
            "audit_note": (
                "Runs completed before the hard guard may have loaded labels through "
                "the repository's original loader, but the training loop discarded "
                "them and they never entered forward, loss, or optimizer updates."
            ),
            "local_lags": LOCAL_LAGS,
            "global_lags": GLOBAL_LAGS,
            "local_topk": 2,
            "global_topk": 4,
        },
        "evaluation": {
            "score_mode": "total",
            "score_search": False,
            "ratio_search": False,
            "threshold_search": False,
            "threshold_rule": "percentile(concat(train_score,test_score),100-anomaly_ratio)",
            "point_adjustment": "asca_ad.model PPLAD-compatible point adjustment",
            "test_labels_used_for": "metrics and point adjustment only; never parameter selection",
            "existing_checkpoint_evaluation": (
                "scores and metrics recomputed in this run; historical metrics are not reused"
            ),
            "comparison_variable": "checkpoint weights",
        },
        "environment": environment_info(),
        "runs": verifications,
    }
    write_json(OUTPUT_ROOT / "protocol.json", protocol)
    print(f"comparison={csv_path}")
    print(f"protocol={OUTPUT_ROOT / 'protocol.json'}")


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.evaluation_only:
            reevaluate_worker(args.worker)
        else:
            run_worker(args.worker, args.overwrite)
    else:
        aggregate(
            not args.aggregate_only,
            args.overwrite,
            evaluation_only=args.evaluation_only,
        )


if __name__ == "__main__":
    main()
