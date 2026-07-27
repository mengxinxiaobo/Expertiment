"""Train/load official LTFAD and generate label-free SKAB anomaly energy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.baseline_common import (  # noqa: E402
    OfficialTrainDatasetView,
    environment_record,
    ensure_score_output_policy,
    load_scaled_skab,
    load_torch_payload,
    relative,
    resolve_official_device,
    set_seed,
    sha256,
    write_json,
)
from scripts.benchmarks.adapters.common import (  # noqa: E402
    LabelFreeWindowDataset,
    collect_point_energy,
)
from scripts.benchmarks.adapters.ltfad_adapter import LTFADScoreAdapter  # noqa: E402


LTFAD_ROOT = ROOT / "BaselineModels" / "LTFAD-main"
if str(LTFAD_ROOT) not in sys.path:
    sys.path.insert(0, str(LTFAD_ROOT))
import solver as official_solver  # type: ignore  # noqa: E402


OUTPUT_ROOT = ROOT / "results" / "SKAB_BENCHMARK"
SCORE_DIR = OUTPUT_ROOT / "scores"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "LTFAD_state_dict.pt"
METADATA_PATH = OUTPUT_ROOT / "LTFAD_score_metadata.json"

CONFIG: dict[str, Any] = {
    "dataset": "SKAB",
    "data_path": "SKAB",
    "index": 137,
    "win_size": 90,
    "local_size": [5],
    "global_size": [13],
    "d_model": 128,
    "batch_size": 128,
    "num_epochs": 2,
    "lr": 1e-4,
    "input_c": 8,
    "output_c": 8,
    "r": 0.1,
    "loss_fuc": "MSE",
}
SEED = 42
STRIDE = 1
EXPECTED_PARAMETERS = 14646


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate official LTFAD SKAB scores")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite-scores", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    return parser.parse_args()


def build_solver(
    train_loader: DataLoader,
    device: torch.device,
    batch_size: int,
):
    solver = object.__new__(official_solver.Solver)
    config = dict(CONFIG)
    config["batch_size"] = int(batch_size)
    solver.__dict__.update(config)
    solver.train_loader = train_loader
    solver.device = device
    solver.build_model()
    solver.model = solver.model.to(device)
    solver.optimizer = torch.optim.Adam(solver.model.parameters(), lr=solver.lr)
    solver.criterion = nn.MSELoss()
    solver.criterion_keep = nn.MSELoss(reduction="none")
    return solver


def checkpoint_source() -> dict[str, str]:
    return {
        "solver": relative(ROOT, LTFAD_ROOT / "solver.py"),
        "solver_sha256": sha256(LTFAD_ROOT / "solver.py"),
        "model": relative(ROOT, LTFAD_ROOT / "model" / "LTFAD.py"),
        "model_sha256": sha256(LTFAD_ROOT / "model" / "LTFAD.py"),
        "best_parameter": relative(ROOT, LTFAD_ROOT / "best_parameter.txt"),
        "best_parameter_sha256": sha256(LTFAD_ROOT / "best_parameter.txt"),
    }


def checkpoint_record(dataset_metadata: dict[str, Any], model) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "model_name": "LTFAD",
        "config": CONFIG,
        "seed": SEED,
        "dataset_sha256": {
            "train": dataset_metadata["train"]["sha256"],
            "test": dataset_metadata["test"]["sha256"],
        },
        "source": checkpoint_source(),
    }


def validate_checkpoint(payload: dict[str, Any], dataset_metadata: dict[str, Any]) -> None:
    if payload.get("model_name") != "LTFAD" or payload.get("config") != CONFIG:
        raise RuntimeError("Existing LTFAD checkpoint config does not match frozen protocol")
    hashes = payload.get("dataset_sha256", {})
    for split in ("train", "test"):
        if hashes.get(split) != dataset_metadata[split]["sha256"]:
            raise RuntimeError(f"Existing LTFAD checkpoint {split} dataset hash mismatch")
    if payload.get("source") != checkpoint_source():
        raise RuntimeError("Existing LTFAD checkpoint official source hash mismatch")


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_official_ltfad(solver, adapter: LTFADScoreAdapter) -> None:
    # Do not call Solver.run(): official run() unconditionally calls test().
    for epoch in range(solver.num_epochs):
        solver.model.train()
        started = time.perf_counter()
        batches = 0
        for windows, _placeholder in solver.train_loader:
            solver.optimizer.zero_grad()
            inputs = windows.float().to(solver.device)
            revin = official_solver.RevIN(num_features=solver.input_c)
            normalized = revin(inputs, "norm")
            local_targets, global_targets = adapter._relations(normalized)
            # Official method performs loss, backward, and optimizer.step().
            solver.train(local_targets, global_targets)
            batches += 1
        official_solver.adjust_learning_rate(solver.optimizer, epoch + 1, solver.lr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"LTFAD epoch {epoch + 1}/{solver.num_epochs}: "
            f"batches={batches}, seconds={time.perf_counter() - started:.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.batch_size != CONFIG["batch_size"]:
        raise ValueError("LTFAD batch_size is frozen to 128")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    device = resolve_official_device(args.device, "LTFAD")
    set_seed(SEED)

    outputs = [
        SCORE_DIR / "LTFAD_train_energy.npy",
        SCORE_DIR / "LTFAD_test_energy.npy",
        SCORE_DIR / "LTFAD_train_window_starts.npy",
        SCORE_DIR / "LTFAD_test_window_starts.npy",
    ]
    ensure_score_output_policy(outputs, args.overwrite_scores)
    train, test, dataset_metadata = load_scaled_skab(ROOT)
    train_dataset = LabelFreeWindowDataset(train, CONFIG["win_size"], STRIDE)
    test_dataset = LabelFreeWindowDataset(test, CONFIG["win_size"], STRIDE)
    train_loader = DataLoader(
        OfficialTrainDatasetView(train_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    solver = build_solver(train_loader, device, args.batch_size)
    parameter_count = sum(p.numel() for p in solver.model.parameters() if p.requires_grad)
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"LTFAD parameters {parameter_count} != {EXPECTED_PARAMETERS}")
    adapter = LTFADScoreAdapter(solver, official_solver)

    trained = False
    if CHECKPOINT_PATH.exists() and not args.retrain:
        payload = load_torch_payload(CHECKPOINT_PATH)
        validate_checkpoint(payload, dataset_metadata)
        solver.model.load_state_dict(payload["model"], strict=True)
        print(f"Loaded LTFAD checkpoint: {CHECKPOINT_PATH}")
    else:
        print("Training official LTFAD with label-free SKAB windows")
        train_official_ltfad(solver, adapter)
        payload = checkpoint_record(dataset_metadata, solver.model)
        save_checkpoint(CHECKPOINT_PATH, payload)
        trained = True
        print(f"Saved LTFAD checkpoint: {CHECKPOINT_PATH}")

    set_seed(SEED)
    solver.model.eval()
    score_loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    train_scores = collect_point_energy(
        adapter,
        DataLoader(train_dataset, **score_loader_options),
        total_length=len(train),
        split="train",
        device=device,
    )
    test_scores = collect_point_energy(
        adapter,
        DataLoader(test_dataset, **score_loader_options),
        total_length=len(test),
        split="test",
        device=device,
    )
    assert len(train_scores.energy) == 12450
    assert len(test_scores.energy) == 5710

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = (
        (outputs[0], train_scores.energy),
        (outputs[1], test_scores.energy),
        (outputs[2], train_scores.window_starts),
        (outputs[3], test_scores.window_starts),
    )
    for path, values in arrays:
        np.save(path, values, allow_pickle=False)

    metadata = {
        "model": "LTFAD",
        "status": "scores_generated",
        "config": CONFIG,
        "seed": SEED,
        "trained_this_run": trained,
        "checkpoint": relative(ROOT, CHECKPOINT_PATH),
        "checkpoint_sha256": sha256(CHECKPOINT_PATH),
        "dataset": dataset_metadata,
        "data_processing": {
            "dtype": "float32",
            "standard_scaler": "fit_train_only",
            "uses_test_label": False,
        },
        "scoring": {
            "stride": STRIDE,
            "overlap_aggregation": "mean",
            "train_window_count": len(train_dataset),
            "test_window_count": len(test_dataset),
            "train_window_score_shape_first_batch": list(train_scores.window_score_shape),
            "test_window_score_shape_first_batch": list(test_scores.window_score_shape),
            "train_energy_shape": list(train_scores.energy.shape),
            "test_energy_shape": list(test_scores.energy.shape),
            "official_complex_dropout_active_in_eval": True,
            "score_seed_reset": SEED,
        },
        "source": checkpoint_source(),
        "environment": environment_record(device),
        "parameter_search": False,
        "threshold_calculated": False,
    }
    write_json(METADATA_PATH, metadata)
    print(f"LTFAD_train_energy.shape={train_scores.energy.shape}")
    print(f"LTFAD_test_energy.shape={test_scores.energy.shape}")
    print(f"metadata={METADATA_PATH}")


if __name__ == "__main__":
    main()
