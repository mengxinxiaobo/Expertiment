#!/usr/bin/env python3
"""Fixed-protocol HAI detection experiment for ASCA-AD V4/PPLAD/LTFAD.

Model workers never open HAI_test_label.npy.  They only train (where required)
and emit official flattened per-window anomaly energy.  The aggregate process
is the sole label consumer and applies one frozen evaluation implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.asca_adapter import ASCAV4ScoreAdapter
from scripts.benchmarks.adapters.common import LabelFreeWindowDataset
from scripts.benchmarks.adapters.hai_ltfad_dataset_adapter import (
    HAILTFADDataAdapter,
    load_frozen_config,
    patched_ltfad_hai_loader,
)
from scripts.benchmarks.adapters.ltfad_adapter import LTFADScoreAdapter
from scripts.benchmarks.adapters.pplad_adapter import PPLADScoreAdapter


SEED = 42
ANOMALY_RATIO = 0.98
PERCENTILE = 99.02
EXPECTED_TRAIN = (896400, 86)
EXPECTED_TEST = (284400, 86)
EXPECTED_LABEL = (284400,)
OUTPUT_ROOT = ROOT / "results" / "HAI_PAPER_RESULTS"
DETECTION_ROOT = OUTPUT_ROOT / "Detection"
SCORES_ROOT = OUTPUT_ROOT / "scores"
ASCA_CHECKPOINT = (
    ROOT / "checkpoints" / "HAI" /
    "HAI_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
    "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
)
ENTRY_SCRIPT = Path(__file__).resolve()
MODELS = ("asca", "pplad", "ltfad")


class TimedProgressLoader:
    """Read-only DataLoader proxy that reports training progress and ETA.

    The proxy is attached by the benchmark only; the official LTFAD solver and
    model remain unchanged.  Timing resumes after each yielded batch, so it
    includes the solver's forward/backward/optimizer work for that batch.
    """

    def __init__(
        self,
        loader: Any,
        *,
        model_name: str,
        total_epochs: int,
        report_every: int = 100,
    ) -> None:
        self._loader = loader
        self._model_name = model_name
        self._total_epochs = int(total_epochs)
        self._report_every = max(1, int(report_every))
        self._epoch = 0
        self._completed_batches = 0
        self._started = time.perf_counter()

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def __iter__(self):
        self._epoch += 1
        epoch = self._epoch
        batches = len(self._loader)
        epoch_started = time.perf_counter()
        print(
            f"[{self._model_name}][train] epoch={epoch}/{self._total_epochs} "
            f"batches={batches} started",
            flush=True,
        )
        for batch_index, batch in enumerate(self._loader, start=1):
            yield batch
            now = time.perf_counter()
            epoch_elapsed = now - epoch_started
            if batch_index % self._report_every == 0 or batch_index == batches:
                seconds_per_batch = epoch_elapsed / batch_index
                epoch_remaining = max(0, batches - batch_index) * seconds_per_batch
                overall_completed = self._completed_batches + batch_index
                overall_elapsed = now - self._started
                overall_rate = overall_elapsed / max(1, overall_completed)
                overall_remaining_batches = (
                    max(0, self._total_epochs - epoch) * batches
                    + max(0, batches - batch_index)
                )
                overall_remaining = overall_remaining_batches * overall_rate
                print(
                    f"[{self._model_name}][train] epoch={epoch}/{self._total_epochs} "
                    f"batch={batch_index}/{batches} "
                    f"elapsed={epoch_elapsed:.1f}s "
                    f"speed={seconds_per_batch:.4f}s/batch "
                    f"epoch_eta={epoch_remaining:.1f}s "
                    f"total_eta={overall_remaining:.1f}s",
                    flush=True,
                )
        self._completed_batches += batches
        print(
            f"[{self._model_name}][train] epoch={epoch}/{self._total_epochs} "
            f"completed elapsed={time.perf_counter() - epoch_started:.1f}s",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODELS)
    parser.add_argument("--aggregate-only", action="store_true")
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


def load_torch(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    return payload


def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate_feature_files() -> None:
    base = ROOT / "dataset" / "HAI"
    for name, expected in (
        ("HAI_train.npy", EXPECTED_TRAIN),
        ("HAI_test.npy", EXPECTED_TEST),
    ):
        values = np.load(base / name, mmap_mode="r", allow_pickle=False)
        if tuple(values.shape) != expected:
            raise RuntimeError(f"Unexpected {name} shape {values.shape}; expected {expected}")
        if values.dtype != np.float32:
            raise RuntimeError(f"Unexpected {name} dtype {values.dtype}; expected float32")


def score_paths(model: str) -> tuple[Path, Path]:
    prefix = {"asca": "ASCA", "pplad": "PPLAD", "ltfad": "LTFAD"}[model]
    return (
        SCORES_ROOT / f"{prefix}_train_energy.npy",
        SCORES_ROOT / f"{prefix}_test_energy.npy",
    )


@torch.no_grad()
def write_flattened_energy(
    score_adapter,
    values: np.ndarray,
    stride: int,
    batch_size: int,
    output_path: Path,
    split: str,
) -> dict[str, Any]:
    dataset = LabelFreeWindowDataset(values, score_adapter.window_size, stride=stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    output_length = len(dataset) * int(score_adapter.window_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npy")
    energy = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(output_length,)
    )
    score_adapter.eval()
    offset = 0
    first_input = None
    first_score = None
    for windows, _placeholder, _starts in loader:
        windows = windows.float().to(device(), non_blocking=True)
        scores = score_adapter.window_scores(windows)
        expected = (int(windows.shape[0]), int(score_adapter.window_size))
        if tuple(scores.shape) != expected:
            raise RuntimeError(
                f"{score_adapter.model_name} score shape {tuple(scores.shape)} != {expected}"
            )
        if not torch.isfinite(scores).all():
            raise RuntimeError(f"{score_adapter.model_name} produced NaN/Inf scores")
        values_flat = scores.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        energy[offset:offset + values_flat.size] = values_flat
        offset += int(values_flat.size)
        if first_input is None:
            first_input = tuple(int(v) for v in windows.shape)
            first_score = tuple(int(v) for v in scores.shape)
    if offset != output_length:
        raise RuntimeError(f"Energy write length {offset} != expected {output_length}")
    energy.flush()
    del energy
    temporary.replace(output_path)
    print(
        f"[{score_adapter.model_name}][{split}] input={first_input}, "
        f"window_score={first_score}, windows={len(dataset)}, energy={output_length}"
    )
    return {
        "input_shape": first_input,
        "window_score_shape": first_score,
        "window_count": int(len(dataset)),
        "energy_length": int(output_length),
        "stride": int(stride),
        "path": str(output_path.relative_to(ROOT)),
    }


def asca_solver_view(model, run_device: torch.device):
    def prepare(windows: torch.Tensor) -> torch.Tensor:
        x = windows.float().to(run_device, non_blocking=True)
        mean = x.mean(dim=1, keepdim=True).detach()
        variance = x.var(dim=1, keepdim=True, unbiased=False).detach()
        return (x - mean) / torch.sqrt(variance + 1e-5)

    def forward_batch(windows: torch.Tensor):
        return model(prepare(windows))

    def score_dict(details: dict[str, torch.Tensor]):
        score = details["score_total"]
        minimum = score.min(dim=-1, keepdim=True).values
        maximum = score.max(dim=-1, keepdim=True).values
        scaled = (score - minimum) / (maximum - minimum + 1e-5)
        return {"total": torch.softmax(scaled, dim=-1)}

    return SimpleNamespace(
        model=model,
        win_size=100,
        score_modes=["total"],
        score_normalization="official",
        _forward_batch=forward_batch,
        _score_dict=score_dict,
    )


def run_asca(overwrite: bool) -> None:
    validate_feature_files()
    set_seed()
    train_path, test_path = score_paths("asca")
    metadata_path = DETECTION_ROOT / "ASCA" / "metadata.json"
    if metadata_path.exists() and train_path.exists() and test_path.exists() and not overwrite:
        print(f"[ASCA] complete artifacts exist; skip: {metadata_path}")
        return
    from asca_ad.model import AdaptiveSparseAnchorCompetitiveModelV4

    payload = load_torch(ASCA_CHECKPOINT)
    config = payload.get("config", {})
    expected_metadata = {
        "dataset": "HAI",
        "input_c": 86,
        "local_candidate_lags": [1, 2, 3, 4, 5, 6, 7, 8],
        "global_candidate_lags": [12, 16, 20, 24, 28, 32, 40, 48],
        "local_topk": 2,
        "global_topk": 4,
    }
    for key, value in expected_metadata.items():
        if config.get(key) != value:
            raise RuntimeError(f"ASCA checkpoint metadata {key}={config.get(key)!r} != {value!r}")
    model = AdaptiveSparseAnchorCompetitiveModelV4(
        local_candidate_lags=expected_metadata["local_candidate_lags"],
        global_candidate_lags=expected_metadata["global_candidate_lags"],
        local_topk=2,
        global_topk=4,
        selector_hidden=int(config.get("selector_hidden", 8)),
        fitter_hidden=int(config.get("fitter_hidden", 8)),
    ).to(device())
    model.load_state_dict(payload["model"], strict=True)
    view = asca_solver_view(model, device())
    adapter = ASCAV4ScoreAdapter(view)
    data = HAILTFADDataAdapter(root=ROOT, phase="train")
    train_record = write_flattened_energy(adapter, data.train_values, 1, 128, train_path, "train")
    test_record = write_flattened_energy(adapter, data.test_values, 100, 128, test_path, "test")
    data.assert_training_label_free()
    write_json(metadata_path, {
        "model": "ASCA-AD V4",
        "training_performed": False,
        "checkpoint": str(ASCA_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(ASCA_CHECKPOINT),
        "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "config": {**expected_metadata, "window": 100, "score_mode": "total"},
        "train_score": train_record,
        "test_score": test_record,
        "test_label_accessed": data.test_label_accessed,
        "files_accessed": data.files_accessed,
    })


def import_official_solver(baseline: str):
    baseline_root = ROOT / "BaselineModels" / baseline
    sys.path.insert(0, str(baseline_root))
    os.chdir(ROOT)
    return importlib.import_module("solver")


def pplad_config(model_dir: Path) -> dict[str, Any]:
    return {
        "index": 137, "dataset": "HAI", "data_path": "HAI",
        "input_c": 86, "output_c": 86, "win_size": 90,
        "batch_size": 128, "num_epochs": 3, "lr": 1e-4,
        "d_model": 128, "local_size": [9], "global_size": [18],
        "r": 0.8, "similar": "MSE", "loss_fuc": "MSE",
        "anormly_ratio": ANOMALY_RATIO, "model_save_path": str(model_dir),
        "mode": "train", "rec_timeseries": True,
    }


def run_pplad(overwrite: bool) -> None:
    validate_feature_files()
    set_seed()
    train_path, test_path = score_paths("pplad")
    model_dir = DETECTION_ROOT / "PPLAD"
    metadata_path = model_dir / "metadata.json"
    checkpoint = model_dir / "PPLAD_state_dict.pt"
    if metadata_path.exists() and train_path.exists() and test_path.exists() and not overwrite:
        print(f"[PPLAD] complete artifacts exist; skip: {metadata_path}")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    solver_module = import_official_solver("PPLAD-main")
    config = pplad_config(model_dir)
    with patched_ltfad_hai_loader(solver_module, "train", root=ROOT) as data:
        runner = solver_module.Solver(config)
        training_performed = not checkpoint.exists() or overwrite
        if training_performed:
            started = time.perf_counter()
            runner.train()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_seconds = float(time.perf_counter() - started)
            torch.save({"model": runner.model.state_dict(), "config": config}, checkpoint)
        else:
            training_seconds = None
            runner.model.load_state_dict(load_torch(checkpoint)["model"], strict=True)
        score_adapter = PPLADScoreAdapter(runner, solver_module)
        train_record = write_flattened_energy(
            score_adapter, data.train_values, 1, 128, train_path, "train"
        )
        test_record = write_flattened_energy(
            score_adapter, data.test_values, 90, 128, test_path, "test"
        )
        data.assert_training_label_free()
        access = list(data.files_accessed)
        label_access = bool(data.test_label_accessed)
    write_json(metadata_path, {
        "model": "PPLAD", "training_performed": training_performed,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "parameters": int(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)),
        "config": config, "config_source": "BaselineModels/PPLAD-main/scripts/HAI.sh",
        "train_score": train_record, "test_score": test_record,
        "test_label_accessed": label_access, "files_accessed": access,
    })


def ltfad_config(model_dir: Path) -> dict[str, Any]:
    frozen = load_frozen_config()
    return {
        "index": 137, "dataset": "HAI", "data_path": "HAI",
        "input_c": 86, "output_c": 86, "win_size": int(frozen["win_size"]),
        "batch_size": int(frozen["batch_size"]),
        "num_epochs": int(frozen["num_epochs"]), "lr": float(frozen["lr"]),
        "d_model": int(frozen["d_model"]), "local_size": list(frozen["local_size"]),
        "global_size": list(frozen["global_size"]), "r": float(frozen["r"]),
        "loss_fuc": "MSE", "anormly_ratio": ANOMALY_RATIO,
        "model_save_path": str(model_dir), "mode": "train",
    }


def run_ltfad(overwrite: bool) -> None:
    validate_feature_files()
    set_seed()
    train_path, test_path = score_paths("ltfad")
    model_dir = DETECTION_ROOT / "LTFAD"
    metadata_path = model_dir / "metadata.json"
    checkpoint = model_dir / "LTFAD_state_dict.pt"
    if metadata_path.exists() and train_path.exists() and test_path.exists() and not overwrite:
        print(f"[LTFAD] complete artifacts exist; skip: {metadata_path}")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    solver_module = import_official_solver("LTFAD-main")
    config = ltfad_config(model_dir)
    with patched_ltfad_hai_loader(solver_module, "train", root=ROOT) as data:
        runner = solver_module.Solver(config)
        training_performed = not checkpoint.exists() or overwrite
        if training_performed:
            runner.train_loader = TimedProgressLoader(
                runner.train_loader,
                model_name="LTFAD",
                total_epochs=int(config["num_epochs"]),
                report_every=1,
            )
            original_test = runner.test
            runner.test = lambda: None
            started = time.perf_counter()
            try:
                runner.run()
            finally:
                runner.test = original_test
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_seconds = float(time.perf_counter() - started)
            torch.save({"model": runner.model.state_dict(), "config": config}, checkpoint)
        else:
            training_seconds = None
            runner.model.load_state_dict(load_torch(checkpoint)["model"], strict=True)
        score_adapter = LTFADScoreAdapter(runner, solver_module)
        train_record = write_flattened_energy(
            score_adapter, data.train_values, 1, 128, train_path, "train"
        )
        test_record = write_flattened_energy(
            score_adapter, data.test_values, 90, 128, test_path, "test"
        )
        data.assert_training_label_free()
        access = list(data.files_accessed)
        label_access = bool(data.test_label_accessed)
    write_json(metadata_path, {
        "model": "LTFAD", "training_performed": training_performed,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "parameters": int(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)),
        "config": config,
        "config_source": "scripts/benchmarks/configs/hai_ltfad_frozen.json",
        "official_hai_configuration_available": False,
        "train_score": train_record, "test_score": test_record,
        "test_label_accessed": label_access, "files_accessed": access,
    })


def binary_metrics(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def point_adjust(prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from asca_ad.model import AdaptiveSparseAnchorSolverV4
    return AdaptiveSparseAnchorSolverV4._point_adjust(
        prediction.astype(np.int64, copy=True), labels.astype(np.int64, copy=False)
    )


def evaluate_all() -> None:
    label_path = ROOT / "dataset" / "HAI" / "HAI_test_label.npy"
    labels = np.load(label_path, allow_pickle=False).reshape(-1).astype(np.int64)
    if labels.shape != EXPECTED_LABEL or not set(np.unique(labels)).issubset({0, 1}):
        raise RuntimeError("Invalid HAI evaluation labels")
    rows = []
    detailed = []
    for worker, display in (("asca", "ASCA-AD V4"), ("pplad", "PPLAD"), ("ltfad", "LTFAD")):
        train_path, test_path = score_paths(worker)
        train_energy = np.load(train_path, mmap_mode="r", allow_pickle=False)
        test_energy = np.load(test_path, mmap_mode="r", allow_pickle=False)
        if test_energy.shape != labels.shape:
            raise RuntimeError(f"{display} test energy {test_energy.shape} != labels {labels.shape}")
        combined = np.concatenate((np.asarray(train_energy), np.asarray(test_energy)))
        threshold = float(np.percentile(combined, PERCENTILE))
        del combined
        raw_prediction = (np.asarray(test_energy) > threshold).astype(np.int64)
        pa_prediction = point_adjust(raw_prediction, labels)
        raw = binary_metrics(raw_prediction, labels)
        pa = binary_metrics(pa_prediction, labels)
        model_dir = DETECTION_ROOT / ({"asca": "ASCA", "pplad": "PPLAD", "ltfad": "LTFAD"}[worker])
        np.save(model_dir / "pred_raw.npy", raw_prediction, allow_pickle=False)
        np.save(model_dir / "pred_pa.npy", pa_prediction, allow_pickle=False)
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        result = {
            "model": display, "anomaly_ratio": ANOMALY_RATIO,
            "percentile": PERCENTILE, "threshold": threshold,
            "threshold_rule": "percentile(concat(train_energy,test_energy),99.02)",
            "train_energy_length": int(train_energy.size),
            "test_energy_length": int(test_energy.size),
            "raw": raw, "pa": pa, "metadata": metadata,
        }
        write_json(model_dir / "metrics.json", result)
        detailed.append(result)
        rows.append({
            "Model": display,
            "Accuracy": raw["accuracy"], "Precision": raw["precision"],
            "Recall": raw["recall"], "F1": raw["f1"],
            "PA-Accuracy": pa["accuracy"], "PA-Precision": pa["precision"],
            "PA-Recall": pa["recall"], "PA-F1": pa["f1"],
            "Threshold": threshold,
        })
        print(
            f"{display}: threshold={threshold:.12g}, RAW_F1={raw['f1']:.6f}, "
            f"PA_F1={pa['f1']:.6f}"
        )
    csv_path = DETECTION_ROOT / "comparison_detection.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(DETECTION_ROOT / "comparison_detection.json", {"results": detailed})
    protocol = {
        "dataset": "HAI", "seed": SEED,
        "data": {
            "train_shape": EXPECTED_TRAIN, "test_shape": EXPECTED_TEST,
            "label_shape": EXPECTED_LABEL, "dtype": "float32",
            "standard_scaler": "fit train only; transform train/test",
            "train_sha256": sha256(ROOT / "dataset/HAI/HAI_train.npy"),
            "test_sha256": sha256(ROOT / "dataset/HAI/HAI_test.npy"),
            "label_sha256": sha256(label_path),
        },
        "evaluation": {
            "anomaly_ratio": ANOMALY_RATIO, "percentile": PERCENTILE,
            "threshold_rule": "independent per model: percentile(concat(train_energy,test_energy),99.02)",
            "point_adjustment": "asca_ad.model.AdaptiveSparseAnchorSolverV4._point_adjust",
            "score_search": False, "ratio_search": False,
            "threshold_search": False, "test_label_parameter_selection": False,
        },
        "label_access_policy": {
            "model_workers": "forbidden; metadata test_label_accessed must be false",
            "aggregate_evaluator": "labels loaded for final raw/PA metrics only",
        },
        "environment": {
            "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "models": [item["metadata"] for item in detailed],
    }
    for metadata in protocol["models"]:
        if metadata.get("test_label_accessed") is not False:
            raise RuntimeError(f"Worker label-access audit failed: {metadata['model']}")
    write_json(OUTPUT_ROOT / "protocol.json", protocol)
    print(f"comparison_csv={csv_path}")
    print(f"protocol={OUTPUT_ROOT / 'protocol.json'}")


def aggregate(run_workers: bool, overwrite: bool) -> None:
    if run_workers:
        for model in MODELS:
            command = [sys.executable, str(ENTRY_SCRIPT), "--worker", model]
            if overwrite:
                command.append("--overwrite")
            subprocess.run(command, cwd=ROOT, check=True)
    evaluate_all()


def main() -> None:
    args = parse_args()
    if args.worker == "asca":
        run_asca(args.overwrite)
    elif args.worker == "pplad":
        run_pplad(args.overwrite)
    elif args.worker == "ltfad":
        run_ltfad(args.overwrite)
    else:
        aggregate(not args.aggregate_only, args.overwrite)


if __name__ == "__main__":
    main()
