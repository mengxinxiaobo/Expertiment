#!/usr/bin/env python3
"""Import/loader/shape audit for the frozen PUMP compatibility layer.

This script does not instantiate a model, execute forward, train, evaluate, or
open PUMP_test_label.npy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.common import LabelFreeWindowDataset
from scripts.benchmarks.adapters.pump_dataset_adapter import (
    PUMPLabelFreeDataAdapter,
    load_pump_frozen_config,
    patched_ltfad_pump_loader,
    patched_pplad_pump_loader,
)


OUTPUT = ROOT / "results" / "PUMP_CHECK" / "adapter_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_solver(repository: str):
    baseline = ROOT / "BaselineModels" / repository
    sys.path.insert(0, str(baseline))
    os.chdir(ROOT)
    module = importlib.import_module("solver")
    if not hasattr(module, "Solver") or not hasattr(module, "get_loader_segment"):
        raise ImportError(f"{repository} solver contract unavailable")
    return module


def validate_asca(config: dict) -> dict:
    from asca_ad.model import AdaptiveSparseAnchorCompetitiveModelV4

    if not callable(AdaptiveSparseAnchorCompetitiveModelV4):
        raise ImportError("ASCA-AD V4 model class is unavailable")
    model_config = config["asca"]
    checkpoint = ROOT / model_config["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_hash = sha256(checkpoint)
    if actual_hash != model_config["checkpoint_sha256"]:
        raise RuntimeError("ASCA checkpoint SHA-256 differs from frozen config")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("config", {})
    expected = {
        "dataset": "PUMP",
        "input_c": 51,
        "local_candidate_lags": [1, 2, 3, 4, 5, 6, 7, 8],
        "global_candidate_lags": [12, 16, 20, 24, 28, 32, 40, 48],
        "local_topk": 2,
        "global_topk": 4,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"ASCA checkpoint {key} mismatch")
    train = np.load(ROOT / "dataset/PUMP/PUMP_train.npy", mmap_mode="r")
    loader = DataLoader(
        LabelFreeWindowDataset(train, window_size=100, stride=1),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    windows, placeholders, _starts = next(iter(loader))
    if tuple(windows.shape) != (128, 100, 51):
        raise RuntimeError(f"Unexpected ASCA input shape: {tuple(windows.shape)}")
    return {
        "model": "ASCA-AD V4",
        "import": True,
        "loader": True,
        "input_shape": list(windows.shape),
        "placeholder_shape": list(placeholders.shape),
        "checkpoint": model_config["checkpoint"],
        "checkpoint_sha256": actual_hash,
        "training": False,
        "test_label_access": False,
        "forward": False,
    }


def validate_baseline(worker: str, config: dict) -> dict:
    repository = "PPLAD-main" if worker == "pplad" else "LTFAD-main"
    solver = import_solver(repository)
    patch = (
        patched_pplad_pump_loader
        if worker == "pplad"
        else patched_ltfad_pump_loader
    )
    frozen = config[worker]
    with patch(solver, "train", root=ROOT) as adapter:
        # Exercise all loader modes created by official Solver.__init__. In
        # train phase they must all remain backed by PUMP_train.npy only.
        loaders = {
            mode: solver.get_loader_segment(
                frozen["index"],
                "dataset/PUMP",
                batch_size=frozen["batch_size"],
                win_size=frozen["win_size"],
                mode=mode,
                dataset="PUMP",
            )
            for mode in ("train", "val", "test", "thre")
        }
        windows, placeholders = next(iter(loaders["train"]))
        expected = (128, int(frozen["win_size"]), 51)
        if tuple(windows.shape) != expected:
            raise RuntimeError(
                f"Unexpected {worker} input: {tuple(windows.shape)} != {expected}"
            )
        adapter.assert_training_access()
        adapter.assert_label_free()
        files = list(adapter.files_accessed)
    return {
        "model": "PPLAD" if worker == "pplad" else "LTFAD",
        "repository_import": True,
        "loader": True,
        "input_shape": list(windows.shape),
        "placeholder_shape": list(placeholders.shape),
        "files_accessed": files,
        "configuration_status": "fixed fallback configuration",
        "training": False,
        "test_label_access": False,
        "forward": False,
    }


def run_worker(worker: str) -> None:
    config = load_pump_frozen_config()
    result = (
        validate_asca(config)
        if worker == "asca"
        else validate_baseline(worker, config)
    )
    path = OUTPUT.with_name(f"adapter_validation_{worker}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{result['model']}: import=PASS loader=PASS "
        f"input_shape={tuple(result['input_shape'])} "
        f"test_label_access={result['test_label_access']}"
    )


def aggregate() -> None:
    for worker in ("asca", "pplad", "ltfad"):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", worker],
            cwd=ROOT,
            check=True,
        )
    models = []
    for worker in ("asca", "pplad", "ltfad"):
        path = OUTPUT.with_name(f"adapter_validation_{worker}.json")
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "dataset": "PUMP",
        "status": "READY",
        "validation_scope": ["import", "dataset loader", "input shape"],
        "training": False,
        "forward": False,
        "metrics_generated": False,
        "test_label_access": False,
        "model_source_modified": False,
        "models": models,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PUMP compatibility validation: READY")
    print(f"report={OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("asca", "pplad", "ltfad"))
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker)
    else:
        aggregate()


if __name__ == "__main__":
    main()
