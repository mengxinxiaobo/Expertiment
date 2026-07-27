"""Shared label-free runtime helpers for SKAB baseline score generators."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


SEED = 42
TRAIN_SHAPE = (12450, 8)
TEST_SHAPE = (5710, 8)


class OfficialTrainDatasetView(Dataset):
    """Expose `(window, placeholder)` while retaining label-free windowing."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        window, placeholder, _start = self.dataset[index]
        return window, placeholder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_official_device(raw: str, model_name: str) -> torch.device:
    if raw == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{model_name}: CUDA requested but unavailable")
    if torch.cuda.is_available() and device != torch.device("cuda:0"):
        raise RuntimeError(
            f"{model_name} official code creates internal tensors on cuda:0; "
            f"requested device {device} is unsupported without modifying the model"
        )
    return device


def load_scaled_skab(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dataset_dir = root / "dataset" / "SKAB"
    train_path = dataset_dir / "SKAB_train.npy"
    test_path = dataset_dir / "SKAB_test.npy"
    train = np.load(train_path, allow_pickle=False)
    test = np.load(test_path, allow_pickle=False)
    if train.shape != TRAIN_SHAPE or test.shape != TEST_SHAPE:
        raise RuntimeError(
            f"Unexpected SKAB shapes: train={train.shape}, test={test.shape}"
        )
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError("SKAB features contain NaN or Inf")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train).astype(np.float32, copy=False)
    test_scaled = scaler.transform(test).astype(np.float32, copy=False)
    if not np.isfinite(train_scaled).all() or not np.isfinite(test_scaled).all():
        raise RuntimeError("StandardScaler produced NaN or Inf")
    metadata = {
        "train": {
            "path": relative(root, train_path),
            "shape": list(TRAIN_SHAPE),
            "sha256": sha256(train_path),
        },
        "test": {
            "path": relative(root, test_path),
            "shape": list(TEST_SHAPE),
            "sha256": sha256(test_path),
        },
    }
    return train_scaled, test_scaled, metadata


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dictionary: {path}")
    return payload


def ensure_score_output_policy(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing score artifacts. Use "
            f"--overwrite-scores to regenerate them:\n{names}"
        )
