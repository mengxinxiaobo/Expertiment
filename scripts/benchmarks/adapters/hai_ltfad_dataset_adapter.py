"""External HAI dataset adapter for the unmodified official LTFAD solver.

The official LTFAD repository has no HAI loader.  This module supplies the
same ``get_loader_segment`` contract without editing anything below
``BaselineModels/LTFAD-main``.

Training and evaluation are deliberately separate phases:

* ``phase='train'`` never opens ``HAI_test_label.npy`` and returns an in-memory
  zero placeholder as the second dataset item.
* ``phase='evaluation'`` loads labels for the official test/evaluation path.

StandardScaler is always fitted on HAI_train.npy only.  Both train and test are
then transformed and stored as float32 arrays shared by all four DataLoaders.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Literal

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = (
    ROOT / "scripts" / "benchmarks" / "configs" / "hai_ltfad_frozen.json"
)
Phase = Literal["train", "evaluation"]


def load_frozen_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("dataset") != "HAI":
        raise ValueError(f"Expected frozen HAI config, got: {config.get('dataset')}")
    if bool(config.get("parameter_search")) or bool(config.get("threshold_search")):
        raise ValueError("Frozen HAI LTFAD config must disable all searches")
    return config


class _HAIWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        labels: np.ndarray | None,
        win_size: int,
        stride: int,
    ) -> None:
        if values.ndim != 2:
            raise ValueError(f"HAI values must be [N,C], got {values.shape}")
        if values.dtype != np.float32:
            raise TypeError(f"HAI adapter requires float32, got {values.dtype}")
        if labels is not None and labels.reshape(-1).size != values.shape[0]:
            raise ValueError("HAI values/labels length mismatch")
        self.values = values
        self.labels = None if labels is None else labels.reshape(-1)
        self.win_size = int(win_size)
        self.stride = int(stride)

    def __len__(self) -> int:
        return (self.values.shape[0] - self.win_size) // self.stride + 1

    def __getitem__(self, index: int):
        start = int(index) * self.stride
        stop = start + self.win_size
        window = self.values[start:stop]
        if self.labels is None:
            label = np.zeros(self.win_size, dtype=np.float32)
        else:
            label = np.float32(self.labels[start:stop])
        return window, label


@dataclass
class HAILTFADDataAdapter:
    root: Path = ROOT
    config_path: Path = DEFAULT_CONFIG_PATH
    phase: Phase = "train"

    def __post_init__(self) -> None:
        if self.phase not in ("train", "evaluation"):
            raise ValueError(f"Unknown adapter phase: {self.phase}")
        self.config = load_frozen_config(self.config_path)
        self.data_dir = self.root / "dataset" / "HAI"
        self.files_accessed: list[str] = []
        self.test_label_accessed = False
        self._train: np.ndarray | None = None
        self._test: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def _load_array(self, name: str) -> np.ndarray:
        path = self.data_dir / name
        if name == "HAI_test_label.npy" and self.phase == "train":
            raise RuntimeError("Training phase is forbidden from reading HAI_test_label.npy")
        values = np.load(path, allow_pickle=False)
        self.files_accessed.append(str(path.relative_to(self.root)))
        if name == "HAI_test_label.npy":
            self.test_label_accessed = True
        return values

    def prepare(self) -> None:
        if self._train is not None:
            return
        train_raw = self._load_array("HAI_train.npy")
        test_raw = self._load_array("HAI_test.npy")
        expected = self.config["expected_shapes"]
        if tuple(train_raw.shape) != tuple(expected["train"]):
            raise ValueError(f"Unexpected HAI train shape: {train_raw.shape}")
        if tuple(test_raw.shape) != tuple(expected["test"]):
            raise ValueError(f"Unexpected HAI test shape: {test_raw.shape}")
        if not np.isfinite(train_raw).all() or not np.isfinite(test_raw).all():
            raise ValueError("HAI train/test contains NaN or Inf")

        scaler = StandardScaler()
        scaler.fit(train_raw)
        self._train = np.asarray(scaler.transform(train_raw), dtype=np.float32)
        self._test = np.asarray(scaler.transform(test_raw), dtype=np.float32)
        if self.phase == "evaluation":
            labels = self._load_array("HAI_test_label.npy")
            if tuple(labels.shape) not in (
                tuple(expected["label"]),
                (int(expected["label"][0]), 1),
            ):
                raise ValueError(f"Unexpected HAI label shape: {labels.shape}")
            unique = set(np.unique(labels).tolist())
            if not unique.issubset({0, 1}):
                raise ValueError(f"HAI labels are not binary: {sorted(unique)}")
            self._labels = labels.reshape(-1)

    @property
    def train_values(self) -> np.ndarray:
        self.prepare()
        assert self._train is not None
        return self._train

    @property
    def test_values(self) -> np.ndarray:
        self.prepare()
        assert self._test is not None
        return self._test

    def get_loader_segment(
        self,
        index,
        data_path,
        batch_size,
        win_size=90,
        step=90,
        mode="train",
        dataset="HAI",
    ) -> DataLoader:
        del index, data_path, step
        if dataset != "HAI":
            raise ValueError(f"HAI adapter cannot serve dataset={dataset!r}")
        frozen_window = int(self.config["win_size"])
        frozen_batch = int(self.config["batch_size"])
        if int(win_size) != frozen_window or int(batch_size) != frozen_batch:
            raise ValueError(
                f"Frozen HAI LTFAD loader requires win={frozen_window}, "
                f"batch={frozen_batch}; got win={win_size}, batch={batch_size}"
            )
        self.prepare()
        if mode == "train":
            values = self.train_values
            labels = None
            stride = 1
        elif mode in ("val", "test"):
            values = self.test_values
            labels = self._labels if self.phase == "evaluation" else None
            stride = 1
        elif mode == "thre":
            values = self.test_values
            labels = self._labels if self.phase == "evaluation" else None
            stride = frozen_window
        else:
            raise ValueError(f"Unsupported official LTFAD loader mode: {mode}")

        window_dataset = _HAIWindowDataset(values, labels, frozen_window, stride)
        return DataLoader(
            window_dataset,
            batch_size=frozen_batch,
            shuffle=(mode == "train"),
            num_workers=0,
            drop_last=False,
        )

    def assert_training_label_free(self) -> None:
        if self.phase != "train":
            raise RuntimeError("Label-free assertion is only meaningful in train phase")
        if self.test_label_accessed:
            raise AssertionError("Training adapter accessed HAI_test_label.npy")
        if any(path.endswith("HAI_test_label.npy") for path in self.files_accessed):
            raise AssertionError("Training access log contains HAI_test_label.npy")


@contextmanager
def patched_ltfad_hai_loader(
    solver_module: ModuleType,
    phase: Phase,
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Iterator[HAILTFADDataAdapter]:
    """Temporarily install the external loader into official ``solver.py``."""
    adapter = HAILTFADDataAdapter(root=root, config_path=config_path, phase=phase)
    original = solver_module.get_loader_segment
    solver_module.get_loader_segment = adapter.get_loader_segment
    try:
        yield adapter
    finally:
        solver_module.get_loader_segment = original


def official_solver_config(
    model_save_path: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    frozen = load_frozen_config(config_path)
    return {
        "index": int(frozen["index"]),
        "dataset": "HAI",
        "data_path": "HAI",
        "input_c": int(frozen["input_c"]),
        "output_c": int(frozen["output_c"]),
        "win_size": int(frozen["win_size"]),
        "batch_size": int(frozen["batch_size"]),
        "num_epochs": int(frozen["num_epochs"]),
        "lr": float(frozen["lr"]),
        "d_model": int(frozen["d_model"]),
        "local_size": list(frozen["local_size"]),
        "global_size": list(frozen["global_size"]),
        "r": float(frozen["r"]),
        "loss_fuc": str(frozen["loss_fuc"]),
        "anormly_ratio": float(frozen["anormly_ratio"]),
        "model_save_path": model_save_path,
        "mode": "train",
    }


def build_training_solver(
    solver_module: ModuleType,
    model_save_path: str,
    root: Path = ROOT,
):
    """Construct official Solver with label-free HAI loaders.

    Keep the returned context open while calling ``runner.run()``.  Because the
    official ``run()`` invokes ``test()`` after training, a formal runner must
    temporarily replace that final call with a no-op, save the trained state,
    and evaluate it in a separate ``phase='evaluation'`` Solver.
    """
    context = patched_ltfad_hai_loader(solver_module, "train", root=root)
    adapter = context.__enter__()
    try:
        runner = solver_module.Solver(official_solver_config(model_save_path))
    except Exception:
        context.__exit__(*__import__("sys").exc_info())
        raise
    return runner, adapter, context


def build_evaluation_solver(
    solver_module: ModuleType,
    model_save_path: str,
    root: Path = ROOT,
):
    """Construct official Solver with HAI labels enabled for final evaluation."""
    context = patched_ltfad_hai_loader(solver_module, "evaluation", root=root)
    adapter = context.__enter__()
    try:
        runner = solver_module.Solver(official_solver_config(model_save_path))
    except Exception:
        context.__exit__(*__import__("sys").exc_info())
        raise
    return runner, adapter, context

