from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .config import PROJECT_ROOT, load_dataset_config, load_experiment_config, select_datasets


def _import_module_from_path(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing runner file: {path}")

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runner_path(runner_file: str) -> Path:
    path = PROJECT_ROOT / "scripts" / "dataset_runners" / runner_file
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset runner: {path}")
    return path


def train_one_dataset(
    dataset: str,
    seed: int | None = None,
    epochs: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Train one dataset through the currently validated dataset runner.

    By default, existing checkpoints are reused. Use force=True only when you
    intentionally want to retrain and overwrite the current checkpoint.
    """
    dataset_config = load_dataset_config()
    experiment_config = load_experiment_config()

    key = dataset.strip().upper()
    if key not in dataset_config:
        valid = ", ".join(dataset_config.keys())
        raise ValueError(f"Unknown dataset: {key}. Valid datasets: {valid}")

    cfg = dataset_config[key]
    final_seed = int(seed) if seed is not None else int(experiment_config.get("default_seed", 42))
    final_epochs = int(epochs) if epochs is not None else int(cfg.get("epochs", 3))

    runner_file = str(cfg["runner"])
    runner_path = _runner_path(runner_file)
    module = _import_module_from_path(runner_path, f"asca_ad_train_runner_{key.lower()}")

    if hasattr(module, "set_seed"):
        module.set_seed(final_seed)

    print()
    print("=" * 80)
    print(f"[{key}] Training")
    print("=" * 80)
    print(f"[{key}] runner : {runner_path}")
    print(f"[{key}] seed   : {final_seed}")
    print(f"[{key}] epochs : {final_epochs}")

    channels = module.verify_data(PROJECT_ROOT)
    _OriginalSolver, V4Solver = module.bootstrap_project(PROJECT_ROOT)
    config = module.v4_config(PROJECT_ROOT, channels, final_seed, final_epochs)

    runner = V4Solver(config)
    checkpoint = Path(runner.checkpoint_path)

    if checkpoint.exists() and not force:
        print(f"[{key}] checkpoint exists; skip training: {checkpoint}")
        return {
            "dataset": key,
            "trained": False,
            "skipped": True,
            "reason": "checkpoint_exists",
            "checkpoint": str(checkpoint),
            "runner": str(runner_path),
            "epochs": final_epochs,
            "seed": final_seed,
        }

    start = time.perf_counter()
    runner.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    seconds = time.perf_counter() - start
    print(f"[{key}] training seconds: {seconds:.3f}")
    print(f"[{key}] checkpoint      : {checkpoint}")

    return {
        "dataset": key,
        "trained": True,
        "skipped": False,
        "checkpoint": str(checkpoint),
        "runner": str(runner_path),
        "epochs": final_epochs,
        "seed": final_seed,
        "training_seconds": float(seconds),
    }


def train_datasets(
    dataset: str,
    seed: int | None = None,
    epochs: int | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    dataset_config = load_dataset_config()
    datasets = select_datasets(dataset, dataset_config)

    results: list[dict[str, Any]] = []
    for item in datasets:
        results.append(
            train_one_dataset(
                dataset=item,
                seed=seed,
                epochs=epochs,
                force=force,
            )
        )
    return results
