"""Label-isolated dataset adapters for the unmodified TranAD model.

This module contains no training, model forward, score generation, threshold,
or metric code. Model-facing adapters never open a test-label file.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "psm_tranad_frozen.json"
MSL_CONFIG_PATH = ROOT / "configs" / "msl_tranad_frozen.json"
PUMP_CONFIG_PATH = ROOT / "configs" / "pump_tranad_frozen.json"
SKAB_CONFIG_PATH = ROOT / "configs" / "skab_tranad_frozen.json"
SMD_CONFIG_PATH = ROOT / "configs" / "smd_tranad_frozen.json"
HAI_CONFIG_PATH = ROOT / "configs" / "hai_tranad_frozen.json"
Phase = Literal["train", "score"]
Split = Literal["train", "test"]


def load_tranad_frozen_config(
    path: Path,
    expected_dataset: str,
) -> dict[str, Any]:
    """Load common frozen TranAD invariants for one registered dataset."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("dataset") != expected_dataset:
        raise ValueError(
            f"Expected TranAD dataset {expected_dataset}, got {config.get('dataset')}"
        )
    if config.get("dtype") != "float32":
        raise ValueError("TranAD dtype must remain float32")

    model = config["model"]
    training = config["training"]
    evaluation = config["evaluation"]
    preprocessing = config["preprocessing"]
    expected = {
        "window": 10,
        "batch_size": 128,
        "epochs": 5,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
    }
    actual = {
        "window": int(model["window"]),
        "batch_size": int(training["batch_size"]),
        "epochs": int(training["epochs"]),
        "optimizer": training["optimizer"],
        "learning_rate": float(training["learning_rate"]),
    }
    if actual != expected:
        raise ValueError(f"TranAD frozen configuration changed: {actual}")
    if preprocessing.get("fit_split") != "train":
        raise ValueError("StandardScaler must be fitted on train only")
    if preprocessing.get("test_in_fit"):
        raise ValueError("Test data cannot participate in scaler fitting")
    input_channels = int(model["input_channels"])
    if input_channels <= 0:
        raise ValueError("TranAD input_channels must be positive")
    if int(model["expected_parameters"]) <= 0:
        raise ValueError("TranAD expected_parameters must be positive")
    for split in ("train", "test"):
        shape = tuple(config["expected_shapes"][split])
        if len(shape) != 2 or int(shape[1]) != input_channels:
            raise ValueError(f"Invalid frozen {split} shape: {shape}")
    anomaly_ratio = float(evaluation["anomaly_ratio"])
    percentile = float(evaluation["percentile"])
    if not np.isclose(percentile, 100.0 - anomaly_ratio):
        raise ValueError("Percentile must equal 100 - anomaly_ratio")
    for key in (
        "pot_spot_enabled",
        "bf_search_enabled",
        "score_search",
        "ratio_search",
        "threshold_search",
        "f1_threshold_search",
        "parameter_search",
        "test_label_parameter_selection",
    ):
        if evaluation.get(key):
            raise ValueError(f"Frozen TranAD protocol must disable {key}")
    return config


def load_tranad_psm_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and enforce the completed PSM protocol without changing behavior."""

    config = load_tranad_frozen_config(path, "PSM")
    if int(config["model"]["input_channels"]) != 25:
        raise ValueError("Frozen TranAD PSM input_channels must remain 25")
    if int(config["model"]["expected_parameters"]) != 57273:
        raise ValueError("Frozen TranAD PSM parameter count changed")
    if tuple(config["expected_shapes"]["train"]) != (132481, 25):
        raise ValueError("Frozen TranAD PSM train shape changed")
    if tuple(config["expected_shapes"]["test"]) != (87841, 25):
        raise ValueError("Frozen TranAD PSM test shape changed")
    if not np.isclose(float(config["evaluation"]["anomaly_ratio"]), 0.8):
        raise ValueError("Frozen TranAD PSM anomaly_ratio must remain 0.8")
    return config


def load_tranad_msl_config(path: Path = MSL_CONFIG_PATH) -> dict[str, Any]:
    """Load and enforce the registered MSL protocol."""

    config = load_tranad_frozen_config(path, "MSL")
    if int(config["model"]["input_channels"]) != 55:
        raise ValueError("Frozen TranAD MSL input_channels must remain 55")
    if int(config["model"]["expected_parameters"]) != 261243:
        raise ValueError("Frozen TranAD MSL parameter count changed")
    if tuple(config["expected_shapes"]["train"]) != (58317, 55):
        raise ValueError("Frozen TranAD MSL train shape changed")
    if tuple(config["expected_shapes"]["test"]) != (73729, 55):
        raise ValueError("Frozen TranAD MSL test shape changed")
    if not np.isclose(float(config["evaluation"]["anomaly_ratio"]), 0.83):
        raise ValueError("Frozen TranAD MSL anomaly_ratio must remain 0.83")
    return config


def load_tranad_pump_config(path: Path = PUMP_CONFIG_PATH) -> dict[str, Any]:
    """Load and enforce the registered PUMP protocol."""

    config = load_tranad_frozen_config(path, "PUMP")
    if int(config["model"]["input_channels"]) != 51:
        raise ValueError("Frozen TranAD PUMP input_channels must remain 51")
    if int(config["model"]["expected_parameters"]) != 225519:
        raise ValueError("Frozen TranAD PUMP parameter count changed")
    if tuple(config["expected_shapes"]["train"]) != (17155, 51):
        raise ValueError("Frozen TranAD PUMP train shape changed")
    if tuple(config["expected_shapes"]["test"]) != (203165, 51):
        raise ValueError("Frozen TranAD PUMP test shape changed")
    if not np.isclose(float(config["evaluation"]["anomaly_ratio"]), 0.5):
        raise ValueError("Frozen TranAD PUMP anomaly_ratio must remain 0.5")
    return config


def load_tranad_skab_config(path: Path = SKAB_CONFIG_PATH) -> dict[str, Any]:
    """Load and enforce the registered SKAB protocol."""

    config = load_tranad_frozen_config(path, "SKAB")
    if int(config["model"]["input_channels"]) != 8:
        raise ValueError("Frozen TranAD SKAB input_channels must remain 8")
    if int(config["model"]["expected_parameters"]) != 7208:
        raise ValueError("Frozen TranAD SKAB parameter count changed")
    if tuple(config["expected_shapes"]["train"]) != (12450, 8):
        raise ValueError("Frozen TranAD SKAB train shape changed")
    if tuple(config["expected_shapes"]["test"]) != (5710, 8):
        raise ValueError("Frozen TranAD SKAB test shape changed")
    if not np.isclose(float(config["evaluation"]["anomaly_ratio"]), 0.5):
        raise ValueError("Frozen TranAD SKAB anomaly_ratio must remain 0.5")
    return config


def load_tranad_smd_config(path: Path = SMD_CONFIG_PATH) -> dict[str, Any]:
    """Load and enforce the registered SMD protocol."""

    config = load_tranad_frozen_config(path, "SMD")
    if int(config["model"]["input_channels"]) != 38:
        raise ValueError("Frozen TranAD SMD input_channels must remain 38")
    if int(config["model"]["expected_parameters"]) != 127538:
        raise ValueError("Frozen TranAD SMD parameter count changed")
    if tuple(config["expected_shapes"]["train"]) != (708405, 38):
        raise ValueError("Frozen TranAD SMD train shape changed")
    if tuple(config["expected_shapes"]["test"]) != (708420, 38):
        raise ValueError("Frozen TranAD SMD test shape changed")
    if not np.isclose(float(config["evaluation"]["anomaly_ratio"]), 0.9):
        raise ValueError("Frozen TranAD SMD anomaly_ratio must remain 0.9")
    return config


def load_tranad_hai_config(path: Path = HAI_CONFIG_PATH) -> dict[str, Any]:
    """Load and enforce the registered HAI protocol."""

    config = load_tranad_frozen_config(path, "HAI")
    if int(config["model"]["input_channels"]) != 86:
        raise ValueError("Frozen TranAD HAI input_channels must remain 86")
    if int(config["model"]["expected_parameters"]) != 627074:
        raise ValueError("Frozen TranAD HAI parameter count changed")
    if tuple(config["expected_shapes"]["train"]) != (896400, 86):
        raise ValueError("Frozen TranAD HAI train shape changed")
    if tuple(config["expected_shapes"]["test"]) != (284400, 86):
        raise ValueError("Frozen TranAD HAI test shape changed")
    if not np.isclose(float(config["evaluation"]["anomaly_ratio"]), 0.98):
        raise ValueError("Frozen TranAD HAI anomaly_ratio must remain 0.98")
    return config


class TranADPSMWindowDataset(Dataset):
    """One official-style TranAD window for every original time point."""

    def __init__(
        self,
        values: np.ndarray,
        window: int = 10,
        channels: int = 25,
    ) -> None:
        self.channels = int(channels)
        if values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError(
                f"Expected values with shape [N,{self.channels}], got {values.shape}"
            )
        if values.dtype != np.float32:
            raise TypeError(f"Expected float32 values, got {values.dtype}")
        if len(values) == 0:
            raise ValueError("TranAD split is empty")
        self.values = values
        self.window = int(window)
        if self.window != 10:
            raise ValueError("Frozen TranAD window must be 10")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index >= self.window:
            window = self.values[index - self.window : index]
        else:
            prefix = np.repeat(
                self.values[0:1],
                repeats=self.window - index,
                axis=0,
            )
            window = np.concatenate((prefix, self.values[0:index]), axis=0)
        if window.shape != (self.window, self.channels):
            raise RuntimeError(f"Unexpected TranAD window shape: {window.shape}")
        return torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))


@dataclass
class TranADPSMDataAdapter:
    """Config-driven loader with train-only scaling and access auditing.

    The historical class name is retained so completed PSM scripts keep exactly
    the same import contract. ``dataset_name`` defaults to PSM.
    """

    phase: Phase
    root: Path = ROOT
    config_path: Path = CONFIG_PATH
    dataset_name: str = "PSM"

    def __post_init__(self) -> None:
        if self.phase not in ("train", "score"):
            raise ValueError(f"Unsupported TranAD adapter phase: {self.phase}")
        self.dataset_name = self.dataset_name.upper()
        self.config = load_tranad_frozen_config(
            self.config_path, self.dataset_name
        )
        if self.dataset_name == "PSM":
            load_tranad_psm_config(self.config_path)
        elif self.dataset_name == "MSL":
            load_tranad_msl_config(self.config_path)
        elif self.dataset_name == "PUMP":
            load_tranad_pump_config(self.config_path)
        elif self.dataset_name == "SKAB":
            load_tranad_skab_config(self.config_path)
        elif self.dataset_name == "SMD":
            load_tranad_smd_config(self.config_path)
        elif self.dataset_name == "HAI":
            load_tranad_hai_config(self.config_path)
        else:
            raise ValueError(f"Unregistered TranAD dataset: {self.dataset_name}")
        self.data_dir = self.root / "dataset" / self.dataset_name
        self.files_accessed: list[str] = []
        self.scaler_fit_files: list[str] = []
        self.training_test_feature_access = False
        self.training_label_access = False
        self.test_label_access = False
        self._train: np.ndarray | None = None
        self._test: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

    def _record(self, path: Path) -> str:
        relative = str(path.relative_to(self.root)).replace("\\", "/")
        self.files_accessed.append(relative)
        return relative

    def _load_array(self, filename: str) -> np.ndarray:
        label_filename = f"{self.dataset_name}_test_label.npy"
        train_filename = f"{self.dataset_name}_train.npy"
        test_filename = f"{self.dataset_name}_test.npy"
        if filename == label_filename or "label" in filename.lower():
            self.test_label_access = True
            raise RuntimeError("TranAD model adapter is forbidden from reading labels")
        if self.phase == "train" and filename != train_filename:
            if filename == test_filename:
                self.training_test_feature_access = True
            raise RuntimeError(
                f"Training phase may only access {train_filename}, got {filename}"
            )
        path = self.data_dir / filename
        values = np.load(path, allow_pickle=False)
        self._record(path)
        return values

    def _validate(
        self, values: np.ndarray, expected: tuple[int, int], split: str
    ) -> None:
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Unexpected {self.dataset_name} {split} shape "
                f"{values.shape}; expected {expected}"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(
                f"{self.dataset_name} {split} must be numeric, got {values.dtype}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{self.dataset_name} {split} contains NaN or Inf")

    def _prepare_train(self) -> None:
        if self._train is not None:
            return
        train_filename = f"{self.dataset_name}_train.npy"
        raw = self._load_array(train_filename)
        expected = tuple(self.config["expected_shapes"]["train"])
        self._validate(raw, expected, "train")
        self._scaler = StandardScaler()
        self._scaler.fit(raw)
        train_path = self.data_dir / train_filename
        self.scaler_fit_files.append(
            str(train_path.relative_to(self.root)).replace("\\", "/")
        )
        self._train = np.asarray(self._scaler.transform(raw), dtype=np.float32)

    def _prepare_test(self) -> None:
        if self.phase != "score":
            raise RuntimeError(
                f"{self.dataset_name} test features are inaccessible during training"
            )
        if self._test is not None:
            return
        self._prepare_train()
        raw = self._load_array(f"{self.dataset_name}_test.npy")
        expected = tuple(self.config["expected_shapes"]["test"])
        self._validate(raw, expected, "test")
        assert self._scaler is not None
        self._test = np.asarray(self._scaler.transform(raw), dtype=np.float32)

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

    def loader(
        self,
        split: Split,
        *,
        batch_size: int | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        if split == "train":
            values = self.train_values
        elif split == "test":
            values = self.test_values
        else:
            raise ValueError(split)
        frozen_batch = int(self.config["training"]["batch_size"])
        selected_batch = int(batch_size or frozen_batch)
        dataset = TranADPSMWindowDataset(
            values,
            window=int(self.config["model"]["window"]),
            channels=int(self.config["model"]["input_channels"]),
        )
        return DataLoader(
            dataset,
            batch_size=selected_batch,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )

    def assert_training_isolated(self) -> None:
        expected = {
            f"dataset/{self.dataset_name}/{self.dataset_name}_train.npy"
        }
        if self.phase != "train":
            raise AssertionError("Training isolation requires a train-phase adapter")
        if set(self.files_accessed) != expected:
            raise AssertionError(f"Unexpected training access: {self.files_accessed}")
        if set(self.scaler_fit_files) != expected:
            raise AssertionError(
                f"Scaler was not fitted exclusively on train: {self.scaler_fit_files}"
            )
        if (
            self.training_test_feature_access
            or self.training_label_access
            or self.test_label_access
        ):
            raise AssertionError("TranAD training data isolation failed")

    def assert_label_free(self) -> None:
        if self.training_label_access or self.test_label_access:
            raise AssertionError("TranAD adapter attempted to access labels")
        if any("label" in path.lower() for path in self.files_accessed):
            raise AssertionError("TranAD access log contains a label file")

    def audit(self) -> dict[str, Any]:
        scaler_samples = None
        if self._scaler is not None:
            seen = self._scaler.n_samples_seen_
            scaler_samples = int(np.asarray(seen).reshape(-1)[0])
        return {
            "dataset": self.dataset_name,
            "phase": self.phase,
            "files_accessed": list(self.files_accessed),
            "scaler_fit_files": list(self.scaler_fit_files),
            "scaler_fit_samples": scaler_samples,
            "training_test_feature_access": self.training_test_feature_access,
            "training_label_access": self.training_label_access,
            "test_label_access": self.test_label_access,
        }


TranADDatasetAdapter = TranADPSMDataAdapter


class TranADMSLDataAdapter(TranADPSMDataAdapter):
    """Convenience constructor for the frozen MSL configuration."""

    def __init__(
        self,
        phase: Phase,
        root: Path = ROOT,
        config_path: Path = MSL_CONFIG_PATH,
    ) -> None:
        super().__init__(
            phase=phase,
            root=root,
            config_path=config_path,
            dataset_name="MSL",
        )


class TranADPUMPDataAdapter(TranADPSMDataAdapter):
    """Convenience constructor for the frozen PUMP configuration."""

    def __init__(
        self,
        phase: Phase,
        root: Path = ROOT,
        config_path: Path = PUMP_CONFIG_PATH,
    ) -> None:
        super().__init__(
            phase=phase,
            root=root,
            config_path=config_path,
            dataset_name="PUMP",
        )


class TranADSKABDataAdapter(TranADPSMDataAdapter):
    """Convenience constructor for the frozen SKAB configuration."""

    def __init__(
        self,
        phase: Phase,
        root: Path = ROOT,
        config_path: Path = SKAB_CONFIG_PATH,
    ) -> None:
        super().__init__(
            phase=phase,
            root=root,
            config_path=config_path,
            dataset_name="SKAB",
        )


class TranADSMDDataAdapter(TranADPSMDataAdapter):
    """Convenience constructor for the frozen SMD configuration."""

    def __init__(
        self,
        phase: Phase,
        root: Path = ROOT,
        config_path: Path = SMD_CONFIG_PATH,
    ) -> None:
        super().__init__(
            phase=phase,
            root=root,
            config_path=config_path,
            dataset_name="SMD",
        )


class TranADHAIDataAdapter(TranADPSMDataAdapter):
    """Convenience constructor for the frozen HAI configuration."""

    def __init__(
        self,
        phase: Phase,
        root: Path = ROOT,
        config_path: Path = HAI_CONFIG_PATH,
    ) -> None:
        super().__init__(
            phase=phase,
            root=root,
            config_path=config_path,
            dataset_name="HAI",
        )


def import_official_tranad_class(learning_rate: float = 1e-4):
    """Import the unmodified TranAD class without importing official evaluation.

    The repository imports DGL for unrelated graph baselines and imports
    dataset-indexed constants that do not include PSM. This function supplies
    narrowly scoped in-memory modules for those two import-time dependencies.
    It does not copy or alter TranAD source code.
    """

    repository = ROOT / "BaselineModels" / "TranAD-main"
    if not (repository / "src" / "models.py").is_file():
        raise FileNotFoundError(repository / "src" / "models.py")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    constants_name = "src.constants"
    if constants_name not in sys.modules:
        constants = types.ModuleType(constants_name)
        constants.lr = float(learning_rate)
        sys.modules[constants_name] = constants

    if "dgl" not in sys.modules and importlib.util.find_spec("dgl") is None:
        dgl = types.ModuleType("dgl")
        dgl_nn = types.ModuleType("dgl.nn")

        class AuditOnlyGATConv(nn.Module):
            """Unavailable graph layer placeholder; TranAD never instantiates it."""

            pass

        dgl_nn.GATConv = AuditOnlyGATConv
        dgl.nn = dgl_nn
        sys.modules["dgl"] = dgl
        sys.modules["dgl.nn"] = dgl_nn

    module = importlib.import_module("src.models")
    module_path = Path(module.__file__).resolve()
    if repository.resolve() not in module_path.parents:
        raise ImportError(f"Imported src.models from unexpected path: {module_path}")

    # PyTorch >= 2 passes causal-hint keywords to custom Transformer layers.
    # TranAD's 2022 layers predate those keywords and do not use causal masks.
    # Add an in-memory signature bridge while leaving the official files intact.
    dlutils = importlib.import_module("src.dlutils")
    encoder_class = dlutils.TransformerEncoderLayer
    if not getattr(encoder_class, "_tranad_torch_compat", False):
        original_encoder_forward = encoder_class.forward

        def encoder_forward_compat(
            self,
            src,
            src_mask=None,
            src_key_padding_mask=None,
            is_causal=False,
        ):
            del is_causal
            return original_encoder_forward(
                self,
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        encoder_class.forward = encoder_forward_compat
        encoder_class._tranad_torch_compat = True

    decoder_class = dlutils.TransformerDecoderLayer
    if not getattr(decoder_class, "_tranad_torch_compat", False):
        original_decoder_forward = decoder_class.forward

        def decoder_forward_compat(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            tgt_is_causal=False,
            memory_is_causal=False,
        ):
            del tgt_is_causal, memory_is_causal
            return original_decoder_forward(
                self,
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        decoder_class.forward = decoder_forward_compat
        decoder_class._tranad_torch_compat = True

    model_class = getattr(module, "TranAD", None)
    if model_class is None:
        raise ImportError("Official TranAD class is unavailable")
    return model_class
