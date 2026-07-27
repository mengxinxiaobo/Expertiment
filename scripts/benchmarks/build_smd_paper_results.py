#!/usr/bin/env python3
"""Fixed-protocol SMD detection for ASCA-AD V4, PPLAD, and LTFAD."""

from __future__ import annotations

import argparse
import csv
import importlib
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.asca_adapter import ASCAV4ScoreAdapter
from scripts.benchmarks.adapters.ltfad_adapter import LTFADScoreAdapter
from scripts.benchmarks.adapters.pplad_adapter import PPLADScoreAdapter
from scripts.benchmarks.adapters.smd_dataset_adapter import (
    SMDLabelFreeDataAdapter,
    load_smd_frozen_config,
    official_solver_config,
    patched_ltfad_smd_loader,
    patched_pplad_smd_loader,
)
from scripts.benchmarks.adapters.baseline_common import sha256
from scripts.benchmarks.build_hai_paper_results import TimedProgressLoader
from scripts.benchmarks.build_pump_paper_results import (
    asca_solver_view,
    binary_metrics,
    point_adjust,
)


SEED = 42
ANOMALY_RATIO = 0.9
PERCENTILE = 99.1
EXPECTED_TRAIN = (708405, 38)
EXPECTED_TEST = (708420, 38)
EXPECTED_LABEL = (708420,)
CONFIG_PATH = ROOT / "configs" / "smd_three_models_frozen.json"
OUTPUT_ROOT = ROOT / "results" / "SMD_PAPER_RESULTS"
DETECTION_ROOT = OUTPUT_ROOT / "Detection"
SCORES_ROOT = OUTPUT_ROOT / "scores"
ENTRY_SCRIPT = Path(__file__).resolve()
MODELS = ("asca", "pplad", "ltfad")


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


def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_torch(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Unsupported checkpoint: {path}")
    return payload


def validate_features() -> None:
    base = ROOT / "dataset" / "SMD"
    for name, expected in (
        ("SMD_train.npy", EXPECTED_TRAIN),
        ("SMD_test.npy", EXPECTED_TEST),
    ):
        values = np.load(base / name, mmap_mode="r", allow_pickle=False)
        if tuple(values.shape) != expected or values.dtype != np.float32:
            raise RuntimeError(
                f"Unexpected {name}: shape={values.shape}, dtype={values.dtype}"
            )


def score_paths(worker: str) -> tuple[Path, Path]:
    prefix = {"asca": "ASCA", "pplad": "PPLAD", "ltfad": "LTFAD"}[worker]
    return (
        SCORES_ROOT / f"{prefix}_train_energy.npy",
        SCORES_ROOT / f"{prefix}_test_energy.npy",
    )


class FixedStrideWindows(Dataset):
    """Official complete-window semantics; incomplete tail is dropped."""

    def __init__(self, values: np.ndarray, window: int, stride: int) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.window = int(window)
        self.stride = int(stride)
        if self.values.ndim != 2 or self.values.shape[1] != 38:
            raise ValueError(f"Expected SMD [N,38], got {self.values.shape}")
        self.count = (len(self.values) - self.window) // self.stride + 1
        if self.count <= 0:
            raise ValueError("SMD split shorter than window")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        start = int(index) * self.stride
        window = self.values[start : start + self.window]
        return torch.from_numpy(window), torch.zeros(self.window), start


@torch.no_grad()
def write_energy(
    adapter,
    values: np.ndarray,
    stride: int,
    output_path: Path,
    split: str,
) -> dict[str, Any]:
    dataset = FixedStrideWindows(values, adapter.window_size, stride)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    output_length = len(dataset) * int(adapter.window_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npy")
    energy = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(output_length,)
    )
    adapter.eval()
    offset = 0
    first_input = None
    first_score = None
    for windows, _placeholder, _starts in loader:
        windows = windows.float().to(device(), non_blocking=True)
        scores = adapter.window_scores(windows)
        expected = (int(windows.shape[0]), int(adapter.window_size))
        if tuple(scores.shape) != expected or not torch.isfinite(scores).all():
            raise RuntimeError(
                f"{adapter.model_name} invalid score: {tuple(scores.shape)}"
            )
        flattened = (
            scores.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        )
        energy[offset : offset + flattened.size] = flattened
        offset += int(flattened.size)
        if first_input is None:
            first_input = tuple(int(value) for value in windows.shape)
            first_score = tuple(int(value) for value in scores.shape)
    if offset != output_length:
        raise RuntimeError(f"Energy length {offset} != {output_length}")
    energy.flush()
    del energy
    temporary.replace(output_path)
    print(
        f"[{adapter.model_name}][{split}] input={first_input}, "
        f"window_score={first_score}, windows={len(dataset)}, energy={output_length}",
        flush=True,
    )
    return {
        "input_shape": first_input,
        "window_score_shape": first_score,
        "window_count": int(len(dataset)),
        "energy_length": int(output_length),
        "stride": int(stride),
        "tail_points_dropped": (
            int(len(values) - output_length)
            if stride == int(adapter.window_size)
            else 0
        ),
        "path": str(output_path.relative_to(ROOT)),
    }


def import_solver(repository: str):
    sys.path.insert(0, str(ROOT / "BaselineModels" / repository))
    os.chdir(ROOT)
    return importlib.import_module("solver")


def run_asca(overwrite: bool) -> None:
    validate_features()
    set_seed()
    frozen = load_smd_frozen_config(CONFIG_PATH)["asca"]
    checkpoint = ROOT / frozen["checkpoint"]
    if sha256(checkpoint) != frozen["checkpoint_sha256"]:
        raise RuntimeError("ASCA checkpoint SHA-256 mismatch")
    train_path, test_path = score_paths("asca")
    model_dir = DETECTION_ROOT / "ASCA"
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists() and train_path.exists() and test_path.exists() and not overwrite:
        print(f"[ASCA] complete artifacts exist; skip: {metadata_path}")
        return

    from asca_ad.model import AdaptiveSparseAnchorCompetitiveModelV4

    payload = load_torch(checkpoint)
    metadata = payload.get("config", {})
    expected = {
        "dataset": "SMD",
        "input_c": 38,
        "local_candidate_lags": frozen["local_candidate_lags"],
        "global_candidate_lags": frozen["global_candidate_lags"],
        "local_topk": 2,
        "global_topk": 4,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"ASCA checkpoint metadata mismatch: {key}")
    model = AdaptiveSparseAnchorCompetitiveModelV4(
        local_candidate_lags=expected["local_candidate_lags"],
        global_candidate_lags=expected["global_candidate_lags"],
        local_topk=2,
        global_topk=4,
        selector_hidden=int(metadata.get("selector_hidden", 8)),
        fitter_hidden=int(metadata.get("fitter_hidden", 8)),
    ).to(device())
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    score_adapter = ASCAV4ScoreAdapter(asca_solver_view(model, device()))
    data = SMDLabelFreeDataAdapter("pplad", phase="score", root=ROOT)
    train_record = write_energy(
        score_adapter, data.train_values, 1, train_path, "train"
    )
    test_record = write_energy(
        score_adapter, data.test_values, 100, test_path, "test"
    )
    data.assert_label_free()
    write_json(metadata_path, {
        "model": "ASCA-AD V4",
        "training_performed": False,
        "checkpoint": frozen["checkpoint"],
        "checkpoint_sha256": sha256(checkpoint),
        "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "config": {**expected, "window": 100, "score_mode": "total"},
        "train_score": train_record,
        "test_score": test_record,
        "test_label_accessed": False,
        "files_accessed": data.files_accessed,
    })


def run_baseline(worker: str, overwrite: bool) -> None:
    validate_features()
    set_seed()
    display = "PPLAD" if worker == "pplad" else "LTFAD"
    repository = "PPLAD-main" if worker == "pplad" else "LTFAD-main"
    patch = (
        patched_pplad_smd_loader
        if worker == "pplad"
        else patched_ltfad_smd_loader
    )
    model_dir = DETECTION_ROOT / display
    checkpoint = model_dir / f"{display}_state_dict.pt"
    metadata_path = model_dir / "metadata.json"
    train_path, test_path = score_paths(worker)
    if metadata_path.exists() and train_path.exists() and test_path.exists() and not overwrite:
        print(f"[{display}] complete artifacts exist; skip: {metadata_path}")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    solver_module = import_solver(repository)
    config = official_solver_config(worker, str(model_dir), CONFIG_PATH)
    with patch(solver_module, "train", root=ROOT) as training_data:
        runner = solver_module.Solver(config)
        training_performed = not checkpoint.exists() or overwrite
        if training_performed:
            started = time.perf_counter()
            if worker == "pplad":
                runner.train()
            else:
                runner.train_loader = TimedProgressLoader(
                    runner.train_loader,
                    model_name="LTFAD",
                    total_epochs=int(config["num_epochs"]),
                    report_every=100,
                )
                original_test = runner.test
                runner.test = lambda: None
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
        training_data.assert_training_access()
        training_files = list(training_data.files_accessed)
    runner.model.eval()
    score_adapter = (
        PPLADScoreAdapter(runner, solver_module)
        if worker == "pplad"
        else LTFADScoreAdapter(runner, solver_module)
    )
    score_data = SMDLabelFreeDataAdapter(worker, phase="score", root=ROOT)
    train_record = write_energy(
        score_adapter, score_data.train_values, 1, train_path, "train"
    )
    test_record = write_energy(
        score_adapter,
        score_data.test_values,
        int(config["win_size"]),
        test_path,
        "test",
    )
    score_data.assert_label_free()
    frozen = load_smd_frozen_config(CONFIG_PATH)[worker]
    write_json(metadata_path, {
        "model": display,
        "training_performed": training_performed,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "parameters": int(
            sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
        ),
        "config": config,
        "configuration_status": frozen["configuration_status"],
        "official_optimal_claimed": False,
        "train_score": train_record,
        "test_score": test_record,
        "training_files_accessed": training_files,
        "score_files_accessed": score_data.files_accessed,
        "training_test_label_accessed": False,
        "test_label_accessed": False,
    })


def evaluate_all() -> None:
    label_path = ROOT / "dataset" / "SMD" / "SMD_test_label.npy"
    labels_all = np.load(label_path, allow_pickle=False).reshape(-1).astype(np.int64)
    if labels_all.shape != EXPECTED_LABEL or not set(np.unique(labels_all)).issubset({0, 1}):
        raise RuntimeError("Invalid SMD labels")
    rows = []
    detailed = []
    for worker, display in (
        ("asca", "ASCA-AD V4"),
        ("pplad", "PPLAD"),
        ("ltfad", "LTFAD"),
    ):
        train_path, test_path = score_paths(worker)
        train_energy = np.load(train_path, mmap_mode="r", allow_pickle=False)
        test_energy = np.load(test_path, mmap_mode="r", allow_pickle=False)
        if test_energy.size > labels_all.size:
            raise RuntimeError(f"{display} test score longer than SMD labels")
        labels = labels_all[: test_energy.size]
        combined = np.concatenate((np.asarray(train_energy), np.asarray(test_energy)))
        threshold = float(np.percentile(combined, PERCENTILE))
        del combined
        raw_prediction = (np.asarray(test_energy) > threshold).astype(np.int64)
        pa_prediction = point_adjust(raw_prediction, labels)
        raw = binary_metrics(raw_prediction, labels)
        pa = binary_metrics(pa_prediction, labels)
        model_dir = DETECTION_ROOT / {
            "asca": "ASCA", "pplad": "PPLAD", "ltfad": "LTFAD"
        }[worker]
        model_metadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if model_metadata.get("test_label_accessed") is not False:
            raise RuntimeError(f"{display} label access audit failed")
        if model_metadata.get("training_test_label_accessed", False) is not False:
            raise RuntimeError(f"{display} training label access audit failed")
        np.save(model_dir / "pred_raw.npy", raw_prediction, allow_pickle=False)
        np.save(model_dir / "pred_pa.npy", pa_prediction, allow_pickle=False)
        result = {
            "model": display,
            "anomaly_ratio": ANOMALY_RATIO,
            "percentile": PERCENTILE,
            "threshold": threshold,
            "threshold_rule": "percentile(concat(train_energy,test_energy),99.1)",
            "train_energy_length": int(train_energy.size),
            "test_energy_length": int(test_energy.size),
            "evaluated_label_length": int(labels.size),
            "raw": raw,
            "pa": pa,
            "metadata": model_metadata,
        }
        write_json(model_dir / "metrics.json", result)
        detailed.append(result)
        rows.append({
            "Model": display,
            "Accuracy": raw["accuracy"],
            "Precision": raw["precision"],
            "Recall": raw["recall"],
            "F1": raw["f1"],
            "PA-Accuracy": pa["accuracy"],
            "PA-Precision": pa["precision"],
            "PA-Recall": pa["recall"],
            "PA-F1": pa["f1"],
            "Threshold": threshold,
        })
        print(
            f"{display}: threshold={threshold:.12g}, RAW_F1={raw['f1']:.6f}, "
            f"PA_F1={pa['f1']:.6f}, evaluated_points={labels.size}",
            flush=True,
        )

    csv_path = DETECTION_ROOT / "comparison_detection.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(DETECTION_ROOT / "comparison_detection.json", {"results": detailed})
    frozen = load_smd_frozen_config(CONFIG_PATH)
    write_json(OUTPUT_ROOT / "protocol.json", {
        "dataset": "SMD",
        "seed": SEED,
        "frozen_config": str(CONFIG_PATH.relative_to(ROOT)),
        "frozen_config_sha256": sha256(CONFIG_PATH),
        "data": {
            "train_shape": EXPECTED_TRAIN,
            "test_shape": EXPECTED_TEST,
            "label_shape": EXPECTED_LABEL,
            "dtype": "float32",
            "standard_scaler": "fit train only; transform train/test",
            "train_sha256": sha256(ROOT / "dataset/SMD/SMD_train.npy"),
            "test_sha256": sha256(ROOT / "dataset/SMD/SMD_test.npy"),
            "label_sha256": sha256(label_path),
        },
        "evaluation": {
            "anomaly_ratio": ANOMALY_RATIO,
            "percentile": PERCENTILE,
            "threshold_rule": (
                "independent per model: "
                "percentile(concat(train_energy,test_energy),99.1)"
            ),
            "point_adjustment": (
                "asca_ad.model.AdaptiveSparseAnchorSolverV4._point_adjust"
            ),
            "score_search": False,
            "ratio_search": False,
            "threshold_search": False,
            "parameter_search": False,
            "test_label_parameter_selection": False,
        },
        "windowing": {
            "train_stride": 1,
            "test_stride": "model window",
            "incomplete_test_tail": "drop; never pad or overlap",
            "evaluated_lengths": {
                result["model"]: result["evaluated_label_length"]
                for result in detailed
            },
        },
        "label_access": {
            "training": False,
            "score_generation": False,
            "final_evaluator_only": True,
        },
        "asca_checkpoint_reference": frozen["asca"],
        "models": [result["metadata"] for result in detailed],
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    })
    print(f"comparison_csv={csv_path}", flush=True)
    print(f"protocol={OUTPUT_ROOT / 'protocol.json'}", flush=True)


def aggregate(run_workers: bool, overwrite: bool) -> None:
    if run_workers:
        for worker in MODELS:
            command = [sys.executable, str(ENTRY_SCRIPT), "--worker", worker]
            if overwrite:
                command.append("--overwrite")
            subprocess.run(command, cwd=ROOT, check=True)
    evaluate_all()


def main() -> None:
    args = parse_args()
    if args.worker == "asca":
        run_asca(args.overwrite)
    elif args.worker in ("pplad", "ltfad"):
        run_baseline(args.worker, args.overwrite)
    else:
        aggregate(not args.aggregate_only, args.overwrite)


if __name__ == "__main__":
    main()
