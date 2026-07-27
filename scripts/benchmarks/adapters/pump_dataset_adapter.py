"""Label-free external PUMP loaders for unmodified PPLAD and LTFAD solvers.

The official repositories instantiate train/validation/test/threshold loaders
inside ``Solver.__init__``. Their bundled PUMP loaders open the test-label file
even for training. This adapter preserves the official two-item batch contract
while ensuring that a training-phase Solver opens only ``PUMP_train.npy``.

No phase in this module opens ``PUMP_test_label.npy``. Labels belong exclusively
to the final evaluator, outside model training and score generation.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Literal

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "pump_three_models_frozen.json"
Phase = Literal["train", "score"]
ModelName = Literal["pplad", "ltfad"]


def load_pump_frozen_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("dataset") != "PUMP":
        raise ValueError(f"Expected PUMP config, got {config.get('dataset')!r}")
    evaluation = config.get("evaluation", {})
    forbidden = (
        "score_search",
        "ratio_search",
        "threshold_search",
        "test_label_parameter_selection",
    )
    if any(bool(evaluation.get(key)) for key in forbidden):
        raise ValueError("Frozen PUMP protocol must disable all searches")
    for model_name in ("pplad", "ltfad"):
        model = config.get(model_name, {})
        if model.get("configuration_status") != "fixed fallback configuration":
            raise ValueError(f"{model_name} must be marked fixed fallback configuration")
        if bool(model.get("official_pump_optimal")):
            raise ValueError(f"{model_name} cannot claim official PUMP optimal status")
    return config


class _PUMPWindowDataset(Dataset):
    def __init__(self, values: np.ndarray, win_size: int, stride: int) -> None:
        if values.ndim != 2 or values.shape[1] != 51:
            raise ValueError(f"PUMP values must be [N,51], got {values.shape}")
        if values.dtype != np.float32:
            raise TypeError(f"PUMP adapter requires float32, got {values.dtype}")
        self.values = values
        self.win_size = int(win_size)
        self.stride = int(stride)
        if self.win_size <= 0 or self.win_size > len(values):
            raise ValueError(f"Invalid window={self.win_size} for {values.shape}")

    def __len__(self) -> int:
        return (len(self.values) - self.win_size) // self.stride + 1

    def __getitem__(self, index: int):
        start = int(index) * self.stride
        stop = start + self.win_size
        window = self.values[start:stop]
        placeholder = np.zeros(self.win_size, dtype=np.float32)
        return window, placeholder


@dataclass
class PUMPLabelFreeDataAdapter:
    model_name: ModelName
    phase: Phase = "train"
    root: Path = ROOT
    config_path: Path = DEFAULT_CONFIG_PATH

    def __post_init__(self) -> None:
        if self.model_name not in ("pplad", "ltfad"):
            raise ValueError(f"Unsupported PUMP model: {self.model_name!r}")
        if self.phase not in ("train", "score"):
            raise ValueError(f"Unsupported PUMP adapter phase: {self.phase!r}")
        self.config = load_pump_frozen_config(self.config_path)
        self.model_config = self.config[self.model_name]
        self.data_dir = self.root / "dataset" / "PUMP"
        self.files_accessed: list[str] = []
        self.test_label_accessed = False
        self._train: np.ndarray | None = None
        self._test: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

    def _load_array(self, filename: str) -> np.ndarray:
        if filename == "PUMP_test_label.npy":
            raise RuntimeError("PUMP model adapters are forbidden from reading test labels")
        if self.phase == "train" and filename != "PUMP_train.npy":
            raise RuntimeError(
                f"PUMP training phase may only read PUMP_train.npy, got {filename}"
            )
        path = self.data_dir / filename
        values = np.load(path, allow_pickle=False)
        self.files_accessed.append(str(path.relative_to(self.root)).replace("\\", "/"))
        return values

    def _prepare_train(self) -> None:
        if self._train is not None:
            return
        train = self._load_array("PUMP_train.npy")
        expected = tuple(self.config["expected_shapes"]["train"])
        if tuple(train.shape) != expected:
            raise ValueError(f"Unexpected PUMP train shape: {train.shape}")
        if not np.isfinite(train).all():
            raise ValueError("PUMP train contains NaN or Inf")
        self._scaler = StandardScaler()
        self._scaler.fit(train)
        self._train = np.asarray(self._scaler.transform(train), dtype=np.float32)

    def _prepare_test(self) -> None:
        if self.phase != "score":
            raise RuntimeError("PUMP test features are unavailable during training")
        if self._test is not None:
            return
        self._prepare_train()
        test = self._load_array("PUMP_test.npy")
        expected = tuple(self.config["expected_shapes"]["test"])
        if tuple(test.shape) != expected:
            raise ValueError(f"Unexpected PUMP test shape: {test.shape}")
        if not np.isfinite(test).all():
            raise ValueError("PUMP test contains NaN or Inf")
        assert self._scaler is not None
        self._test = np.asarray(self._scaler.transform(test), dtype=np.float32)

    @property
    def train_values(self) -> np.ndarray:
        self._prepare_train()
        assert self._train is not None
        return self._train

    @property
    def test_values(self) -> np.ndarray:
        self._prepare_test()
        assert self._test is not None
        return self._test

    def get_loader_segment(
        self,
        index,
        data_path,
        batch_size,
        win_size=100,
        step=100,
        mode="train",
        dataset="PUMP",
    ) -> DataLoader:
        del index, data_path, step
        if dataset != "PUMP":
            raise ValueError(f"PUMP adapter cannot serve dataset={dataset!r}")
        frozen_window = int(self.model_config["win_size"])
        frozen_batch = int(self.model_config["batch_size"])
        if int(win_size) != frozen_window or int(batch_size) != frozen_batch:
            raise ValueError(
                f"Frozen {self.model_name} loader requires win={frozen_window}, "
                f"batch={frozen_batch}; got win={win_size}, batch={batch_size}"
            )

        if self.phase == "train":
            # Official Solver creates all four loaders up front. Supplying the
            # train array to unused non-train loaders prevents any test access.
            values = self.train_values
        elif mode == "train":
            values = self.train_values
        elif mode in ("val", "test", "thre"):
            values = self.test_values
        else:
            raise ValueError(f"Unsupported official loader mode: {mode!r}")

        stride = frozen_window if mode == "thre" else 1
        dataset_view = _PUMPWindowDataset(values, frozen_window, stride)
        return DataLoader(
            dataset_view,
            batch_size=frozen_batch,
            shuffle=(mode == "train" and self.phase == "train"),
            num_workers=0,
            drop_last=False,
        )

    def assert_training_access(self) -> None:
        if self.phase != "train":
            raise RuntimeError("Training access audit requires phase='train'")
        allowed = {"dataset/PUMP/PUMP_train.npy"}
        if not self.files_accessed or set(self.files_accessed) != allowed:
            raise AssertionError(
                f"Unexpected PUMP training files: {self.files_accessed}"
            )
        if self.test_label_accessed:
            raise AssertionError("PUMP training adapter accessed test labels")

    def assert_label_free(self) -> None:
        if self.test_label_accessed:
            raise AssertionError("PUMP adapter accessed test labels")
        if any(path.endswith("PUMP_test_label.npy") for path in self.files_accessed):
            raise AssertionError("PUMP access log contains test labels")


@contextmanager
def patched_pump_loader(
    solver_module: ModuleType,
    model_name: ModelName,
    phase: Phase,
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Iterator[PUMPLabelFreeDataAdapter]:
    adapter = PUMPLabelFreeDataAdapter(
        model_name=model_name,
        phase=phase,
        root=root,
        config_path=config_path,
    )
    original = solver_module.get_loader_segment
    solver_module.get_loader_segment = adapter.get_loader_segment
    try:
        yield adapter
    finally:
        solver_module.get_loader_segment = original


def patched_pplad_pump_loader(
    solver_module: ModuleType,
    phase: Phase,
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
):
    return patched_pump_loader(
        solver_module, "pplad", phase, root=root, config_path=config_path
    )


def patched_ltfad_pump_loader(
    solver_module: ModuleType,
    phase: Phase,
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
):
    return patched_pump_loader(
        solver_module, "ltfad", phase, root=root, config_path=config_path
    )


def official_solver_config(
    model_name: ModelName,
    model_save_path: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    frozen = load_pump_frozen_config(config_path)
    source = frozen[model_name]
    config = {
        "index": int(source["index"]),
        "dataset": "PUMP",
        "data_path": "PUMP",
        "input_c": int(source["input_c"]),
        "output_c": int(source["output_c"]),
        "win_size": int(source["win_size"]),
        "batch_size": int(source["batch_size"]),
        "num_epochs": int(source["num_epochs"]),
        "lr": float(source["lr"]),
        "d_model": int(source["d_model"]),
        "local_size": list(source["local_size"]),
        "global_size": list(source["global_size"]),
        "r": float(source["r"]),
        "loss_fuc": str(source["loss_fuc"]),
        "anormly_ratio": float(source["anormly_ratio"]),
        "model_save_path": model_save_path,
        "mode": "train",
    }
    if model_name == "pplad":
        config["similar"] = str(source["similar"])
        config["rec_timeseries"] = True
    return config
