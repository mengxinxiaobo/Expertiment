"""Label-free external SMD loaders for unmodified PPLAD and LTFAD solvers.

During training, every loader requested by official Solver.__init__ is backed
by SMD_train.npy, so neither SMD_test.npy nor SMD_test_label.npy is opened.
During score generation, train/test feature arrays are available, but this
module never opens labels in any phase.
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
DEFAULT_CONFIG_PATH = ROOT / "configs" / "smd_three_models_frozen.json"
Phase = Literal["train", "score"]
ModelName = Literal["pplad", "ltfad"]


def load_smd_frozen_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("dataset") != "SMD":
        raise ValueError(f"Expected SMD config, got {config.get('dataset')!r}")
    evaluation = config.get("evaluation", {})
    forbidden = (
        "score_search",
        "ratio_search",
        "threshold_search",
        "parameter_search",
        "test_label_parameter_selection",
    )
    if any(bool(evaluation.get(key)) for key in forbidden):
        raise ValueError("Frozen SMD protocol must disable all searches")
    if config["pplad"].get("configuration_status") != "fixed SMD configuration":
        raise ValueError("PPLAD must be marked fixed SMD configuration")
    if config["ltfad"].get("configuration_status") != "fixed fallback configuration":
        raise ValueError("LTFAD must be marked fixed fallback configuration")
    if config["pplad"].get("official_optimal_claimed"):
        raise ValueError("PPLAD configuration cannot claim official optimal status")
    if config["ltfad"].get("official_optimal_claimed"):
        raise ValueError("LTFAD configuration cannot claim official optimal status")
    return config


class _SMDWindowDataset(Dataset):
    def __init__(self, values: np.ndarray, win_size: int, stride: int) -> None:
        if values.ndim != 2 or values.shape[1] != 38:
            raise ValueError(f"SMD values must be [N,38], got {values.shape}")
        if values.dtype != np.float32:
            raise TypeError(f"SMD adapter requires float32, got {values.dtype}")
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
class SMDLabelFreeDataAdapter:
    model_name: ModelName
    phase: Phase = "train"
    root: Path = ROOT
    config_path: Path = DEFAULT_CONFIG_PATH

    def __post_init__(self) -> None:
        if self.model_name not in ("pplad", "ltfad"):
            raise ValueError(f"Unsupported SMD model: {self.model_name!r}")
        if self.phase not in ("train", "score"):
            raise ValueError(f"Unsupported SMD phase: {self.phase!r}")
        self.config = load_smd_frozen_config(self.config_path)
        self.model_config = self.config[self.model_name]
        self.data_dir = self.root / "dataset" / "SMD"
        self.files_accessed: list[str] = []
        self.test_label_accessed = False
        self._train: np.ndarray | None = None
        self._test: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

    def _load_array(self, filename: str) -> np.ndarray:
        if filename == "SMD_test_label.npy":
            raise RuntimeError("SMD model adapters cannot read test labels")
        if self.phase == "train" and filename != "SMD_train.npy":
            raise RuntimeError(
                f"SMD training may only read SMD_train.npy, got {filename}"
            )
        path = self.data_dir / filename
        values = np.load(path, allow_pickle=False)
        self.files_accessed.append(
            str(path.relative_to(self.root)).replace("\\", "/")
        )
        return values

    def _prepare_train(self) -> None:
        if self._train is not None:
            return
        train = self._load_array("SMD_train.npy")
        if tuple(train.shape) != tuple(self.config["expected_shapes"]["train"]):
            raise ValueError(f"Unexpected SMD train shape: {train.shape}")
        if not np.isfinite(train).all():
            raise ValueError("SMD train contains NaN or Inf")
        self._scaler = StandardScaler()
        self._scaler.fit(train)
        self._train = np.asarray(self._scaler.transform(train), dtype=np.float32)

    def _prepare_test(self) -> None:
        if self.phase != "score":
            raise RuntimeError("SMD test features are unavailable during training")
        if self._test is not None:
            return
        self._prepare_train()
        test = self._load_array("SMD_test.npy")
        if tuple(test.shape) != tuple(self.config["expected_shapes"]["test"]):
            raise ValueError(f"Unexpected SMD test shape: {test.shape}")
        if not np.isfinite(test).all():
            raise ValueError("SMD test contains NaN or Inf")
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
        dataset="SMD",
    ) -> DataLoader:
        del index, data_path, step
        if dataset != "SMD":
            raise ValueError(f"SMD adapter cannot serve dataset={dataset!r}")
        frozen_window = int(self.model_config["win_size"])
        frozen_batch = int(self.model_config["batch_size"])
        if int(win_size) != frozen_window or int(batch_size) != frozen_batch:
            raise ValueError(
                f"Frozen {self.model_name} loader requires win={frozen_window}, "
                f"batch={frozen_batch}; got win={win_size}, batch={batch_size}"
            )

        if self.phase == "train":
            # Official Solver constructs val/test/thre loaders at initialization.
            # They intentionally use train values here and remain unused while
            # the formal worker trains with official test() disabled.
            values = self.train_values
        elif mode == "train":
            values = self.train_values
        elif mode in ("val", "test", "thre"):
            values = self.test_values
        else:
            raise ValueError(f"Unsupported official loader mode: {mode!r}")

        stride = frozen_window if mode == "thre" else 1
        dataset_view = _SMDWindowDataset(values, frozen_window, stride)
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
        allowed = {"dataset/SMD/SMD_train.npy"}
        if not self.files_accessed or set(self.files_accessed) != allowed:
            raise AssertionError(f"Unexpected SMD training files: {self.files_accessed}")
        if self.test_label_accessed:
            raise AssertionError("SMD training adapter accessed test labels")

    def assert_label_free(self) -> None:
        if self.test_label_accessed:
            raise AssertionError("SMD adapter accessed test labels")
        if any(path.endswith("SMD_test_label.npy") for path in self.files_accessed):
            raise AssertionError("SMD access log contains test labels")


@contextmanager
def patched_smd_loader(
    solver_module: ModuleType,
    model_name: ModelName,
    phase: Phase,
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Iterator[SMDLabelFreeDataAdapter]:
    adapter = SMDLabelFreeDataAdapter(
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


def patched_pplad_smd_loader(
    solver_module: ModuleType,
    phase: Phase,
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
):
    return patched_smd_loader(
        solver_module, "pplad", phase, root=root, config_path=config_path
    )


def patched_ltfad_smd_loader(
    solver_module: ModuleType,
    phase: Phase,
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
):
    return patched_smd_loader(
        solver_module, "ltfad", phase, root=root, config_path=config_path
    )


def official_solver_config(
    model_name: ModelName,
    model_save_path: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    source = load_smd_frozen_config(config_path)[model_name]
    config = {
        "index": int(source["index"]),
        "dataset": "SMD",
        "data_path": "SMD",
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
