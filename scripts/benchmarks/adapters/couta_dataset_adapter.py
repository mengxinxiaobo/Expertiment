"""External, fail-closed PSM adapter for the unmodified DeepOD COUTA.

This module owns only optional-Ray import compatibility, file-access auditing,
train-only scaling, and checkpoint serialization. It does not reimplement the
COUTA network, loss, training loop, synthetic negatives, center, or score.
"""

from __future__ import annotations

import hashlib
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
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DEEPOD_ROOT = ROOT / "BaselineModels" / "DeepOD-main"
CONFIG_PATH = ROOT / "configs" / "psm_couta_frozen.json"
SKAB_CONFIG_PATH = ROOT / "configs" / "skab_couta_frozen.json"
MSL_CONFIG_PATH = ROOT / "configs" / "msl_couta_frozen.json"
HAI_CONFIG_PATH = ROOT / "configs" / "hai_couta_frozen.json"
PUMP_CONFIG_PATH = ROOT / "configs" / "pump_couta_frozen.json"
SMD_CONFIG_PATH = ROOT / "configs" / "smd_couta_frozen.json"
Phase = Literal["train", "score", "evaluate", "validation"]

RAY_AUDIT: dict[str, Any] = {
    "ray_installed": importlib.util.find_spec("ray") is not None,
    "ray_stub_used": False,
    "ray_tune_used": False,
    "training_ray_used": False,
    "ray_tune_attempt_blocked": False,
}


def _ray_forbidden(*_args: Any, **_kwargs: Any) -> Any:
    RAY_AUDIT["ray_tune_attempt_blocked"] = True
    raise RuntimeError("Ray Tune is forbidden by the frozen COUTA protocol")


class _ForbiddenSession:
    def report(self, *_args: Any, **_kwargs: Any) -> None:
        RAY_AUDIT["ray_tune_attempt_blocked"] = True
        raise RuntimeError("Ray session.report is forbidden")


class _ForbiddenCheckpoint:
    @classmethod
    def from_dict(cls, *_args: Any, **_kwargs: Any) -> Any:
        RAY_AUDIT["ray_tune_attempt_blocked"] = True
        raise RuntimeError("Ray checkpointing is forbidden")


class _ForbiddenASHAScheduler:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        RAY_AUDIT["ray_tune_attempt_blocked"] = True
        raise RuntimeError("Ray scheduler is forbidden")


def install_fail_closed_ray_stub() -> None:
    """Install process-local Ray modules that fail on every tuning operation."""
    if RAY_AUDIT["ray_installed"]:
        raise RuntimeError(
            "Ray is installed; this adapter expects the audited ray_installed=false environment"
        )
    if RAY_AUDIT["ray_stub_used"]:
        return

    ray = types.ModuleType("ray")
    ray.__path__ = []
    tune = types.ModuleType("ray.tune")
    tune.__path__ = []
    air = types.ModuleType("ray.air")
    air.__path__ = []
    schedulers = types.ModuleType("ray.tune.schedulers")

    tune.grid_search = _ray_forbidden
    tune.choice = _ray_forbidden
    tune.run = _ray_forbidden
    air.session = _ForbiddenSession()
    air.Checkpoint = _ForbiddenCheckpoint
    schedulers.ASHAScheduler = _ForbiddenASHAScheduler
    ray.tune = tune
    ray.air = air
    sys.modules.update({
        "ray": ray,
        "ray.tune": tune,
        "ray.air": air,
        "ray.tune.schedulers": schedulers,
    })
    RAY_AUDIT["ray_stub_used"] = True


def import_official_couta():
    """Return official COUTA classes after installing only the Ray import stub."""
    install_fail_closed_ray_stub()
    if str(DEEPOD_ROOT) not in sys.path:
        sys.path.insert(0, str(DEEPOD_ROOT))
    module = importlib.import_module("deepod.models.time_series.couta")
    module_path = Path(module.__file__).resolve()
    expected = (DEEPOD_ROOT / "deepod" / "models" / "time_series" / "couta.py").resolve()
    if module_path != expected:
        raise ImportError(f"COUTA imported from unexpected path: {module_path}")
    return module.COUTA, module._COUTANet


def load_couta_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "PSM", "model": "COUTA", "seed": 42, "dtype": "float32"
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen COUTA {key} changed: {config.get(key)}")
    if config.get("config_type") != "fixed external configuration":
        raise ValueError("COUTA PSM config must remain fixed external configuration")
    model = config["model_config"]
    frozen_model = {
        "seq_len": 30, "train_stride": 10, "inference_stride": 1,
        "epochs": 20, "batch_size": 64, "lr": 1e-4,
        "hidden_dims": 16, "rep_dim": 16, "rep_hidden": 16,
        "pretext_hidden": 16, "kernel_size": 2, "dropout": 0.0,
        "bias": True, "alpha": 0.1, "neg_batch_ratio": 0.2,
        "train_val_pc": 0.25, "ss_type": "FULL", "random_state": 42,
        "expected_parameters": 2897,
    }
    for key, value in frozen_model.items():
        if model.get(key) != value:
            raise ValueError(f"Frozen COUTA parameter changed: {key}={model.get(key)}")
    if tuple(config["expected_shapes"]["train"]) != (132481, 25):
        raise ValueError("PSM train shape changed")
    if tuple(config["expected_shapes"]["test"]) != (87841, 25):
        raise ValueError("PSM test shape changed")
    evaluation = config["evaluation"]
    if not np.isclose(evaluation["anomaly_ratio"], 0.8):
        raise ValueError("PSM anomaly ratio changed")
    if not np.isclose(evaluation["percentile"], 99.2):
        raise ValueError("PSM percentile changed")
    for key in ("score_search", "ratio_search", "threshold_search",
                "parameter_search", "oracle_search", "best_f1_search"):
        if evaluation.get(key):
            raise ValueError(f"Forbidden search enabled: {key}")
    return config


def load_skab_couta_config(path: Path = SKAB_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, value in {"dataset": "SKAB", "model": "COUTA", "seed": 42,
                       "dtype": "float32"}.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen COUTA {key} changed: {config.get(key)}")
    if config.get("config_type") != "fixed external configuration":
        raise ValueError("COUTA SKAB config must remain fixed external configuration")
    model = config["model_config"]
    frozen = {
        "seq_len": 30, "train_stride": 10, "inference_stride": 1,
        "epochs": 20, "batch_size": 64, "lr": 1e-4,
        "hidden_dims": 16, "rep_dim": 16, "rep_hidden": 16,
        "pretext_hidden": 16, "kernel_size": 2, "dropout": 0.0,
        "bias": True, "alpha": 0.1, "neg_batch_ratio": 0.2,
        "train_val_pc": 0.25, "ss_type": "FULL", "random_state": 42,
        "expected_parameters": 2081,
    }
    for key, value in frozen.items():
        if model.get(key) != value:
            raise ValueError(f"Frozen SKAB COUTA parameter changed: {key}={model.get(key)}")
    if tuple(config["expected_shapes"]["train"]) != (12450, 8):
        raise ValueError("SKAB train shape changed")
    if tuple(config["expected_shapes"]["test"]) != (5710, 8):
        raise ValueError("SKAB test shape changed")
    if tuple(config["expected_shapes"]["label"]) != (5710, 1):
        raise ValueError("SKAB label shape changed")
    evaluation = config["evaluation"]
    if not np.isclose(evaluation["anomaly_ratio"], 0.5) or not np.isclose(
            evaluation["percentile"], 99.5):
        raise ValueError("SKAB fixed evaluation protocol changed")
    for key in ("score_search", "ratio_search", "threshold_search",
                "parameter_search", "oracle_search", "best_f1_search"):
        if evaluation.get(key):
            raise ValueError(f"Forbidden search enabled: {key}")
    return config


def load_msl_couta_config(path: Path = MSL_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, value in {"dataset": "MSL", "model": "COUTA", "seed": 42,
                       "dtype": "float32"}.items():
        if config.get(key) != value: raise ValueError(f"Frozen MSL {key} changed")
    if config.get("config_type") != "fixed external configuration":
        raise ValueError("MSL config must be fixed external configuration")
    frozen = {"seq_len": 30, "train_stride": 10, "inference_stride": 1,
              "epochs": 20, "batch_size": 64, "lr": 1e-4,
              "hidden_dims": 16, "rep_dim": 16, "rep_hidden": 16,
              "pretext_hidden": 16, "kernel_size": 2, "dropout": 0.0,
              "bias": True, "alpha": 0.1, "neg_batch_ratio": 0.2,
              "train_val_pc": 0.25, "ss_type": "FULL", "random_state": 42,
              "expected_parameters": 4337}
    for key, value in frozen.items():
        if config["model_config"].get(key) != value:
            raise ValueError(f"Frozen MSL COUTA parameter changed: {key}")
    if tuple(config["expected_shapes"]["train"]) != (58317,55): raise ValueError("MSL train shape")
    if tuple(config["expected_shapes"]["test"]) != (73729,55): raise ValueError("MSL test shape")
    if tuple(config["expected_shapes"]["label"]) != (73729,): raise ValueError("MSL label shape")
    evaluation = config["evaluation"]
    if not np.isclose(evaluation["anomaly_ratio"], .83) or not np.isclose(evaluation["percentile"], 99.17):
        raise ValueError("MSL evaluation protocol changed")
    for key in ("score_search","ratio_search","threshold_search","parameter_search","oracle_search","best_f1_search"):
        if evaluation.get(key): raise ValueError(f"Forbidden search enabled: {key}")
    return config


def load_hai_couta_config(path: Path = HAI_CONFIG_PATH) -> dict[str, Any]:
    config=json.loads(path.read_text(encoding="utf-8"))
    for key,value in {"dataset":"HAI","model":"COUTA","seed":42,"dtype":"float32"}.items():
        if config.get(key)!=value: raise ValueError(f"Frozen HAI {key} changed")
    if config.get("config_type")!="fixed external configuration": raise ValueError("HAI config type")
    frozen={"seq_len":30,"train_stride":10,"inference_stride":1,"epochs":20,"batch_size":64,"lr":1e-4,
            "hidden_dims":16,"rep_dim":16,"rep_hidden":16,"pretext_hidden":16,"kernel_size":2,"dropout":0.0,
            "bias":True,"alpha":0.1,"neg_batch_ratio":0.2,"train_val_pc":0.25,"ss_type":"FULL",
            "random_state":42,"expected_parameters":5825,"bypass_unused_internal_fit_scoring":True,"score_chunk_points":20000}
    for key,value in frozen.items():
        if config["model_config"].get(key)!=value: raise ValueError(f"Frozen HAI COUTA parameter changed: {key}")
    if tuple(config["expected_shapes"]["train"])!=(896400,86): raise ValueError("HAI train shape")
    if tuple(config["expected_shapes"]["test"])!=(284400,86): raise ValueError("HAI test shape")
    if tuple(config["expected_shapes"]["label"])!=(284400,): raise ValueError("HAI label shape")
    e=config["evaluation"]
    if not np.isclose(e["anomaly_ratio"],.98) or not np.isclose(e["percentile"],99.02): raise ValueError("HAI protocol")
    for key in ("score_search","ratio_search","threshold_search","parameter_search","oracle_search","best_f1_search"):
        if e.get(key): raise ValueError(f"Forbidden search enabled: {key}")
    return config


def load_pump_couta_config(path: Path = PUMP_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, value in {"dataset": "PUMP", "model": "COUTA", "seed": 42,
                       "dtype": "float32"}.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen PUMP {key} changed: {config.get(key)}")
    if config.get("config_type") != "fixed external configuration":
        raise ValueError("PUMP config must remain a fixed external configuration")
    frozen = {
        "seq_len": 30, "train_stride": 10, "inference_stride": 1,
        "epochs": 20, "batch_size": 64, "lr": 1e-4,
        "hidden_dims": 16, "rep_dim": 16, "rep_hidden": 16,
        "pretext_hidden": 16, "kernel_size": 2, "dropout": 0.0,
        "bias": True, "alpha": 0.1, "neg_batch_ratio": 0.2,
        "train_val_pc": 0.25, "ss_type": "FULL", "random_state": 42,
        "expected_parameters": 4145, "score_chunk_points": 20000,
    }
    for key, value in frozen.items():
        if config["model_config"].get(key) != value:
            raise ValueError(f"Frozen PUMP COUTA parameter changed: {key}")
    shapes = config["expected_shapes"]
    if tuple(shapes["train"]) != (17155, 51): raise ValueError("PUMP train shape")
    if tuple(shapes["test"]) != (203165, 51): raise ValueError("PUMP test shape")
    if tuple(shapes["label"]) != (203165,): raise ValueError("PUMP label shape")
    evaluation = config["evaluation"]
    if not np.isclose(evaluation["anomaly_ratio"], 0.5) or not np.isclose(
            evaluation["percentile"], 99.5):
        raise ValueError("PUMP fixed evaluation protocol changed")
    for key in ("score_search", "ratio_search", "threshold_search",
                "parameter_search", "oracle_search", "best_f1_search"):
        if evaluation.get(key):
            raise ValueError(f"Forbidden search enabled: {key}")
    return config


def load_smd_couta_config(path: Path = SMD_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, value in {"dataset": "SMD", "model": "COUTA", "seed": 42,
                       "dtype": "float32"}.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen SMD {key} changed: {config.get(key)}")
    if config.get("config_type") != "fixed external configuration":
        raise ValueError("SMD config must remain a fixed external configuration")
    frozen = {
        "seq_len": 30, "train_stride": 10, "inference_stride": 1,
        "epochs": 20, "batch_size": 64, "lr": 1e-4,
        "hidden_dims": 16, "rep_dim": 16, "rep_hidden": 16,
        "pretext_hidden": 16, "kernel_size": 2, "dropout": 0.0,
        "bias": True, "alpha": 0.1, "neg_batch_ratio": 0.2,
        "train_val_pc": 0.25, "ss_type": "FULL", "random_state": 42,
        "expected_parameters": 3521, "bypass_unused_internal_fit_scoring": True,
        "score_chunk_points": 20000,
    }
    for key, value in frozen.items():
        if config["model_config"].get(key) != value:
            raise ValueError(f"Frozen SMD COUTA parameter changed: {key}")
    shapes = config["expected_shapes"]
    if tuple(shapes["train"]) != (708405, 38): raise ValueError("SMD train shape")
    if tuple(shapes["test"]) != (708420, 38): raise ValueError("SMD test shape")
    if tuple(shapes["label"]) != (708420,): raise ValueError("SMD label shape")
    evaluation = config["evaluation"]
    if not np.isclose(evaluation["anomaly_ratio"], 0.9) or not np.isclose(
            evaluation["percentile"], 99.1):
        raise ValueError("SMD fixed evaluation protocol changed")
    for key in ("score_search", "ratio_search", "threshold_search",
                "parameter_search", "oracle_search", "best_f1_search"):
        if evaluation.get(key):
            raise ValueError(f"Forbidden search enabled: {key}")
    return config


@dataclass
class PSMCOUTADataAdapter:
    phase: Phase
    root: Path = ROOT

    def __post_init__(self) -> None:
        if self.phase not in {"train", "score", "evaluate", "validation"}:
            raise ValueError(self.phase)
        self.data_dir = self.root / "dataset" / "PSM"
        self.files_accessed: list[str] = []
        self.scaler_fit_files: list[str] = []
        self.training_test_access = False
        self.training_test_label_access = False
        self.score_label_access = False

    def _record(self, path: Path) -> None:
        self.files_accessed.append(str(path.relative_to(self.root)).replace("\\", "/"))

    def _load(self, filename: str, *, mmap_mode: str | None = None) -> np.ndarray:
        is_test = filename == "PSM_test.npy"
        is_label = filename == "PSM_test_label.npy"
        if self.phase in {"train", "validation"} and (is_test or is_label):
            self.training_test_access |= is_test
            self.training_test_label_access |= is_label
            raise RuntimeError("COUTA training/validation may access PSM_train.npy only")
        if self.phase == "score" and is_label:
            self.score_label_access = True
            raise RuntimeError("COUTA score stage cannot read test labels")
        if self.phase == "evaluate" and not is_label:
            raise RuntimeError("COUTA evaluate adapter may read only test labels")
        path = self.data_dir / filename
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        self._record(path)
        return array

    @staticmethod
    def _validate(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
        if tuple(array.shape) != shape:
            raise ValueError(f"Unexpected {name} shape {array.shape}; expected {shape}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite numeric data")

    def load_train(self, limit: int | None = None) -> np.ndarray:
        array = self._load("PSM_train.npy", mmap_mode="r" if limit else None)
        self._validate(array, (132481, 25), "PSM train")
        selected = array if limit is None else array[:limit]
        return np.asarray(selected, dtype=np.float32).copy()

    def load_test(self) -> np.ndarray:
        array = self._load("PSM_test.npy")
        self._validate(array, (87841, 25), "PSM test")
        return np.asarray(array, dtype=np.float32)

    def load_label(self) -> np.ndarray:
        array = self._load("PSM_test_label.npy")
        self._validate(array, (87841,), "PSM label")
        return np.asarray(array, dtype=np.int64).reshape(-1)

    def fit_train_scaler(self, train: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
        if self.phase not in {"train", "validation"}:
            raise RuntimeError("Scaler fitting is restricted to train/validation stage")
        scaler = StandardScaler()
        scaler.fit(train)
        self.scaler_fit_files.append("dataset/PSM/PSM_train.npy")
        return scaler, np.asarray(scaler.transform(train), dtype=np.float32)

    def assert_training_isolated(self) -> None:
        if set(self.files_accessed) != {"dataset/PSM/PSM_train.npy"}:
            raise AssertionError(f"Unexpected train access: {self.files_accessed}")
        if self.training_test_access or self.training_test_label_access:
            raise AssertionError("COUTA training isolation failed")
        if self.scaler_fit_files != ["dataset/PSM/PSM_train.npy"]:
            raise AssertionError("Scaler was not fit exclusively on train")

    def audit(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "files_accessed": list(self.files_accessed),
            "scaler_fit_files": list(self.scaler_fit_files),
            "training_test_access": self.training_test_access,
            "training_test_label_access": self.training_test_label_access,
            "score_label_access": self.score_label_access,
        }


@dataclass
class SKABCOUTADataAdapter:
    """SKAB split adapter with the same fail-closed phase semantics as PSM."""
    phase: Phase
    root: Path = ROOT

    def __post_init__(self) -> None:
        if self.phase not in {"train", "score", "evaluate", "validation"}:
            raise ValueError(self.phase)
        self.data_dir = self.root / "dataset" / "SKAB"
        self.files_accessed: list[str] = []
        self.scaler_fit_files: list[str] = []
        self.training_test_access = False
        self.training_test_label_access = False
        self.score_label_access = False

    def _record(self, path: Path) -> None:
        self.files_accessed.append(str(path.relative_to(self.root)).replace("\\", "/"))

    def _load(self, filename: str) -> np.ndarray:
        is_test = filename == "SKAB_test.npy"
        is_label = filename == "SKAB_test_label.npy"
        if self.phase in {"train", "validation"} and (is_test or is_label):
            self.training_test_access |= is_test
            self.training_test_label_access |= is_label
            raise RuntimeError("COUTA training/validation may access SKAB_train.npy only")
        if self.phase == "score" and is_label:
            self.score_label_access = True
            raise RuntimeError("COUTA score stage cannot read test labels")
        if self.phase == "evaluate" and not is_label:
            raise RuntimeError("COUTA evaluate adapter may read only test labels")
        path = self.data_dir / filename
        array = np.load(path, allow_pickle=False)
        self._record(path)
        return array

    @staticmethod
    def _validate(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
        if tuple(array.shape) != shape:
            raise ValueError(f"Unexpected {name} shape {array.shape}; expected {shape}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite numeric data")

    def load_train(self, limit: int | None = None) -> np.ndarray:
        array = self._load("SKAB_train.npy")
        self._validate(array, (12450, 8), "SKAB train")
        selected = array if limit is None else array[:limit]
        return np.asarray(selected, dtype=np.float32).copy()

    def load_test(self) -> np.ndarray:
        array = self._load("SKAB_test.npy")
        self._validate(array, (5710, 8), "SKAB test")
        return np.asarray(array, dtype=np.float32)

    def load_label(self) -> np.ndarray:
        array = self._load("SKAB_test_label.npy")
        self._validate(array, (5710, 1), "SKAB label")
        return np.asarray(array, dtype=np.int64).reshape(-1)

    def fit_train_scaler(self, train: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
        if self.phase not in {"train", "validation"}:
            raise RuntimeError("Scaler fitting is restricted to train/validation stage")
        scaler = StandardScaler().fit(train)
        self.scaler_fit_files.append("dataset/SKAB/SKAB_train.npy")
        return scaler, np.asarray(scaler.transform(train), dtype=np.float32)

    def assert_training_isolated(self) -> None:
        if set(self.files_accessed) != {"dataset/SKAB/SKAB_train.npy"}:
            raise AssertionError(f"Unexpected train access: {self.files_accessed}")
        if self.training_test_access or self.training_test_label_access:
            raise AssertionError("COUTA training isolation failed")
        if self.scaler_fit_files != ["dataset/SKAB/SKAB_train.npy"]:
            raise AssertionError("Scaler was not fit exclusively on SKAB train")

    def audit(self) -> dict[str, Any]:
        return {"phase": self.phase, "files_accessed": list(self.files_accessed),
                "scaler_fit_files": list(self.scaler_fit_files),
                "training_test_access": self.training_test_access,
                "training_test_label_access": self.training_test_label_access,
                "score_label_access": self.score_label_access}


@dataclass
class MSLCOUTADataAdapter:
    phase: Phase
    root: Path = ROOT

    def __post_init__(self) -> None:
        if self.phase not in {"train","score","evaluate","validation"}: raise ValueError(self.phase)
        self.data_dir = self.root / "dataset" / "MSL"
        self.files_accessed: list[str] = []; self.scaler_fit_files: list[str] = []
        self.training_test_access = False; self.training_test_label_access = False; self.score_label_access = False

    def _record(self, path: Path) -> None:
        self.files_accessed.append(str(path.relative_to(self.root)).replace("\\", "/"))

    def _load(self, filename: str) -> np.ndarray:
        is_test, is_label = filename == "MSL_test.npy", filename == "MSL_test_label.npy"
        if self.phase in {"train","validation"} and (is_test or is_label):
            self.training_test_access |= is_test; self.training_test_label_access |= is_label
            raise RuntimeError("COUTA train/validation may access MSL_train.npy only")
        if self.phase == "score" and is_label:
            self.score_label_access = True; raise RuntimeError("Score stage cannot read MSL label")
        if self.phase == "evaluate" and not is_label: raise RuntimeError("Evaluate may read label only")
        path = self.data_dir / filename; array = np.load(path, allow_pickle=False); self._record(path); return array

    @staticmethod
    def _validate(array: np.ndarray, shape: tuple[int,...], name: str) -> None:
        if tuple(array.shape) != shape: raise ValueError(f"Unexpected {name} shape {array.shape}")
        if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_: raise ValueError(f"{name} dtype")
        if not np.isfinite(array).all(): raise ValueError(f"{name} non-finite")

    def load_train(self, limit: int | None = None) -> np.ndarray:
        array=self._load("MSL_train.npy"); self._validate(array,(58317,55),"MSL train")
        return np.asarray(array if limit is None else array[:limit],dtype=np.float32).copy()
    def load_test(self) -> np.ndarray:
        array=self._load("MSL_test.npy"); self._validate(array,(73729,55),"MSL test"); return np.asarray(array,dtype=np.float32)
    def load_label(self) -> np.ndarray:
        array=self._load("MSL_test_label.npy"); self._validate(array,(73729,),"MSL label"); return np.asarray(array,dtype=np.int64)
    def fit_train_scaler(self, train: np.ndarray):
        if self.phase not in {"train","validation"}: raise RuntimeError("Scaler fit restricted to train")
        scaler=StandardScaler().fit(train); self.scaler_fit_files.append("dataset/MSL/MSL_train.npy")
        return scaler,np.asarray(scaler.transform(train),dtype=np.float32)
    def assert_training_isolated(self) -> None:
        if set(self.files_accessed) != {"dataset/MSL/MSL_train.npy"}: raise AssertionError(self.files_accessed)
        if self.training_test_access or self.training_test_label_access: raise AssertionError("MSL isolation failed")
        if self.scaler_fit_files != ["dataset/MSL/MSL_train.npy"]: raise AssertionError("MSL scaler isolation")
    def audit(self):
        return {"phase":self.phase,"files_accessed":list(self.files_accessed),"scaler_fit_files":list(self.scaler_fit_files),
                "training_test_access":self.training_test_access,"training_test_label_access":self.training_test_label_access,
                "score_label_access":self.score_label_access}


@dataclass
class HAICOUTADataAdapter:
    phase: Phase
    root: Path = ROOT
    def __post_init__(self):
        if self.phase not in {"train","score","evaluate","validation"}: raise ValueError(self.phase)
        self.data_dir=self.root/"dataset"/"HAI"; self.files_accessed=[]; self.scaler_fit_files=[]
        self.training_test_access=False; self.training_test_label_access=False; self.score_label_access=False
    def _record(self,path): self.files_accessed.append(str(path.relative_to(self.root)).replace("\\","/"))
    def _load(self,name,mmap_mode=None):
        is_test=name=="HAI_test.npy"; is_label=name=="HAI_test_label.npy"
        if self.phase in {"train","validation"} and (is_test or is_label):
            self.training_test_access|=is_test; self.training_test_label_access|=is_label
            raise RuntimeError("HAI train/validation may access train only")
        if self.phase=="score" and is_label: self.score_label_access=True; raise RuntimeError("HAI score label forbidden")
        if self.phase=="evaluate" and not is_label: raise RuntimeError("HAI evaluate may read label only")
        path=self.data_dir/name; array=np.load(path,mmap_mode=mmap_mode,allow_pickle=False); self._record(path); return array
    @staticmethod
    def _validate(array,shape,name):
        if tuple(array.shape)!=shape: raise ValueError(f"Unexpected {name} shape {array.shape}")
        if not np.issubdtype(array.dtype,np.number) or not np.isfinite(array).all(): raise ValueError(f"Invalid {name}")
    def load_train(self,limit=None):
        array=self._load("HAI_train.npy",mmap_mode="r" if limit else None); self._validate(array,(896400,86),"HAI train")
        return np.asarray(array if limit is None else array[:limit],dtype=np.float32).copy()
    def load_test(self):
        array=self._load("HAI_test.npy"); self._validate(array,(284400,86),"HAI test"); return np.asarray(array,dtype=np.float32)
    def load_label(self):
        array=self._load("HAI_test_label.npy"); self._validate(array,(284400,),"HAI label"); return np.asarray(array,dtype=np.int64)
    def fit_train_scaler(self,train):
        if self.phase not in {"train","validation"}: raise RuntimeError("HAI scaler fit restricted")
        scaler=StandardScaler().fit(train); self.scaler_fit_files.append("dataset/HAI/HAI_train.npy")
        if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_==0):
            raise RuntimeError("Invalid HAI scaler state")
        scaled=np.asarray(scaler.transform(train),dtype=np.float32)
        if not np.isfinite(scaled).all(): raise RuntimeError("Invalid HAI scaled train")
        return scaler,scaled
    def assert_training_isolated(self):
        if set(self.files_accessed)!={"dataset/HAI/HAI_train.npy"}: raise AssertionError(self.files_accessed)
        if self.training_test_access or self.training_test_label_access: raise AssertionError("HAI isolation")
        if self.scaler_fit_files!=["dataset/HAI/HAI_train.npy"]: raise AssertionError("HAI scaler isolation")
    def audit(self):
        return {"phase":self.phase,"files_accessed":list(self.files_accessed),"scaler_fit_files":list(self.scaler_fit_files),
                "training_test_access":self.training_test_access,"training_test_label_access":self.training_test_label_access,
                "score_label_access":self.score_label_access}


@dataclass
class PUMPCOUTADataAdapter:
    """Fail-closed PUMP split adapter; model/evaluation behavior stays external."""
    phase: Phase
    root: Path = ROOT

    def __post_init__(self):
        if self.phase not in {"train", "score", "evaluate", "validation"}:
            raise ValueError(self.phase)
        self.data_dir = self.root / "dataset" / "PUMP"
        self.files_accessed: list[str] = []
        self.scaler_fit_files: list[str] = []
        self.training_test_access = False
        self.training_test_label_access = False
        self.score_label_access = False

    def _record(self, path: Path) -> None:
        self.files_accessed.append(str(path.relative_to(self.root)).replace("\\", "/"))

    def _load(self, name: str, mmap_mode: str | None = None) -> np.ndarray:
        is_test = name == "PUMP_test.npy"
        is_label = name == "PUMP_test_label.npy"
        if self.phase in {"train", "validation"} and (is_test or is_label):
            self.training_test_access |= is_test
            self.training_test_label_access |= is_label
            raise RuntimeError("PUMP train/validation stage may access PUMP_train.npy only")
        if self.phase == "score" and is_label:
            self.score_label_access = True
            raise RuntimeError("PUMP score stage cannot read test labels")
        if self.phase == "evaluate" and not is_label:
            raise RuntimeError("PUMP evaluate stage may read only PUMP_test_label.npy")
        path = self.data_dir / name
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        self._record(path)
        return array

    @staticmethod
    def _validate(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
        if tuple(array.shape) != shape:
            raise ValueError(f"Unexpected {name} shape {array.shape}; expected {shape}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name}")

    def load_train(self, limit: int | None = None) -> np.ndarray:
        array = self._load("PUMP_train.npy", mmap_mode="r" if limit else None)
        self._validate(array, (17155, 51), "PUMP train")
        selected = array if limit is None else array[:limit]
        return np.asarray(selected, dtype=np.float32).copy()

    def load_test(self) -> np.ndarray:
        array = self._load("PUMP_test.npy")
        self._validate(array, (203165, 51), "PUMP test")
        return np.asarray(array, dtype=np.float32)

    def load_label(self) -> np.ndarray:
        array = self._load("PUMP_test_label.npy")
        self._validate(array, (203165,), "PUMP label")
        return np.asarray(array, dtype=np.int64).reshape(-1)

    def fit_train_scaler(self, train: np.ndarray):
        if self.phase not in {"train", "validation"}:
            raise RuntimeError("PUMP scaler fitting is restricted to train")
        scaler = StandardScaler().fit(train)
        self.scaler_fit_files.append("dataset/PUMP/PUMP_train.npy")
        if (not np.isfinite(scaler.mean_).all() or
                not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ == 0)):
            raise RuntimeError("Invalid PUMP scaler state")
        scaled = np.asarray(scaler.transform(train), dtype=np.float32)
        if not np.isfinite(scaled).all():
            raise RuntimeError("Invalid PUMP scaled train")
        return scaler, scaled

    def assert_training_isolated(self) -> None:
        if set(self.files_accessed) != {"dataset/PUMP/PUMP_train.npy"}:
            raise AssertionError(f"Unexpected PUMP train access: {self.files_accessed}")
        if self.training_test_access or self.training_test_label_access:
            raise AssertionError("PUMP training isolation failed")
        if self.scaler_fit_files != ["dataset/PUMP/PUMP_train.npy"]:
            raise AssertionError("PUMP scaler was not fit exclusively on train")

    def audit(self) -> dict[str, Any]:
        return {
            "phase": self.phase, "files_accessed": list(self.files_accessed),
            "scaler_fit_files": list(self.scaler_fit_files),
            "training_test_access": self.training_test_access,
            "training_test_label_access": self.training_test_label_access,
            "score_label_access": self.score_label_access,
        }


@dataclass
class SMDCOUTADataAdapter:
    """Fail-closed SMD split adapter for the unmodified official COUTA."""
    phase: Phase
    root: Path = ROOT

    def __post_init__(self):
        if self.phase not in {"train", "score", "evaluate", "validation"}:
            raise ValueError(self.phase)
        self.data_dir = self.root / "dataset" / "SMD"
        self.files_accessed: list[str] = []
        self.scaler_fit_files: list[str] = []
        self.training_test_access = False
        self.training_test_label_access = False
        self.score_label_access = False

    def _record(self, path: Path) -> None:
        self.files_accessed.append(str(path.relative_to(self.root)).replace("\\", "/"))

    def _load(self, name: str, mmap_mode: str | None = None) -> np.ndarray:
        is_test = name == "SMD_test.npy"
        is_label = name == "SMD_test_label.npy"
        if self.phase in {"train", "validation"} and (is_test or is_label):
            self.training_test_access |= is_test
            self.training_test_label_access |= is_label
            raise RuntimeError("SMD train/validation stage may access SMD_train.npy only")
        if self.phase == "score" and is_label:
            self.score_label_access = True
            raise RuntimeError("SMD score stage cannot read test labels")
        if self.phase == "evaluate" and not is_label:
            raise RuntimeError("SMD evaluate stage may read only SMD_test_label.npy")
        path = self.data_dir / name
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        self._record(path)
        return array

    @staticmethod
    def _validate(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
        if tuple(array.shape) != shape:
            raise ValueError(f"Unexpected {name} shape {array.shape}; expected {shape}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name}")

    def load_train(self, limit: int | None = None) -> np.ndarray:
        array = self._load("SMD_train.npy", mmap_mode="r" if limit else None)
        self._validate(array, (708405, 38), "SMD train")
        selected = array if limit is None else array[:limit]
        return np.asarray(selected, dtype=np.float32).copy()

    def load_test(self) -> np.ndarray:
        array = self._load("SMD_test.npy")
        self._validate(array, (708420, 38), "SMD test")
        return np.asarray(array, dtype=np.float32)

    def load_label(self) -> np.ndarray:
        array = self._load("SMD_test_label.npy")
        self._validate(array, (708420,), "SMD label")
        return np.asarray(array, dtype=np.int64).reshape(-1)

    def fit_train_scaler(self, train: np.ndarray):
        if self.phase not in {"train", "validation"}:
            raise RuntimeError("SMD scaler fitting is restricted to train")
        scaler = StandardScaler().fit(train)
        self.scaler_fit_files.append("dataset/SMD/SMD_train.npy")
        if (not np.isfinite(scaler.mean_).all() or
                not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ == 0)):
            raise RuntimeError("Invalid SMD scaler state")
        scaled = np.asarray(scaler.transform(train), dtype=np.float32)
        if not np.isfinite(scaled).all():
            raise RuntimeError("Invalid SMD scaled train")
        return scaler, scaled

    def assert_training_isolated(self) -> None:
        if set(self.files_accessed) != {"dataset/SMD/SMD_train.npy"}:
            raise AssertionError(f"Unexpected SMD train access: {self.files_accessed}")
        if self.training_test_access or self.training_test_label_access:
            raise AssertionError("SMD training isolation failed")
        if self.scaler_fit_files != ["dataset/SMD/SMD_train.npy"]:
            raise AssertionError("SMD scaler was not fit exclusively on train")

    def audit(self) -> dict[str, Any]:
        return {
            "phase": self.phase, "files_accessed": list(self.files_accessed),
            "scaler_fit_files": list(self.scaler_fit_files),
            "training_test_access": self.training_test_access,
            "training_test_label_access": self.training_test_label_access,
            "score_label_access": self.score_label_access,
        }


def construct_official_couta(config: dict[str, Any], device: str, *, epochs: int | None = None):
    COUTA, _ = import_official_couta()
    m = config["model_config"]
    model = COUTA(
        seq_len=m["seq_len"], stride=m["train_stride"],
        epochs=m["epochs"] if epochs is None else int(epochs),
        batch_size=m["batch_size"], lr=m["lr"], ss_type=m["ss_type"],
        hidden_dims=m["hidden_dims"], rep_dim=m["rep_dim"],
        rep_hidden=m["rep_hidden"], pretext_hidden=m["pretext_hidden"],
        kernel_size=m["kernel_size"], dropout=m["dropout"], bias=m["bias"],
        alpha=m["alpha"], neg_batch_ratio=m["neg_batch_ratio"],
        train_val_pc=m["train_val_pc"], device=device,
        verbose=2, random_state=m["random_state"],
    )
    def forbidden_auto_hyper(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("fit_auto_hyper/_training_ray is forbidden")
    model.fit_auto_hyper = forbidden_auto_hyper
    model._training_ray = forbidden_auto_hyper
    return model


def construct_official_network(config: dict[str, Any], input_dim: int, device: str):
    _, Network = import_official_couta()
    m = config["model_config"]
    net = Network(
        input_dim=input_dim, hidden_dims=m["hidden_dims"], n_output=m["rep_dim"],
        pretext_hidden=m["pretext_hidden"], rep_hidden=m["rep_hidden"],
        out_dim=1, kernel_size=m["kernel_size"], dropout=m["dropout"],
        bias=m["bias"], pretext=True, dup=True,
    ).to(device)
    return net


def scaler_to_state(scaler: StandardScaler) -> dict[str, Any]:
    return {
        "mean": np.asarray(scaler.mean_), "scale": np.asarray(scaler.scale_),
        "var": np.asarray(scaler.var_), "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": int(np.asarray(scaler.n_samples_seen_).reshape(-1)[0]),
    }


def scaler_from_state(state: dict[str, Any]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean"])
    scaler.scale_ = np.asarray(state["scale"])
    scaler.var_ = np.asarray(state["var"])
    scaler.n_features_in_ = int(state["n_features_in"])
    scaler.n_samples_seen_ = int(state["n_samples_seen"])
    return scaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_bundle(model: Any, scaler: StandardScaler, config: dict[str, Any]) -> dict[str, Any]:
    scaler_state = scaler_to_state(scaler)
    return {
        "model_state_dict": model.net.state_dict(),
        "center_c": model.c.detach().cpu(),
        "scaler": scaler_state,
        # Explicit aliases make the formal artifact self-describing while the
        # nested state remains the single restoration source.
        "scaler_mean": scaler_state["mean"],
        "scaler_scale": scaler_state["scale"],
        "scaler_var": scaler_state["var"],
        "n_features_in": int(model.n_features),
        "input_c": int(model.n_features),
        "seq_len": int(model.seq_len),
        "train_stride": int(model.stride),
        "inference_stride": 1,
        "config": config,
        "seed": int(config["seed"]),
        "epoch": int(model.epochs),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset": config["dataset"],
        "dataset_file_names": [f"{config['dataset']}_train.npy", f"{config['dataset']}_test.npy",
                               f"{config['dataset']}_test_label.npy"],
        "train_shape": config["expected_shapes"]["train"],
        "test_shape": config["expected_shapes"]["test"],
        "score_chunk_size": config["model_config"].get("score_chunk_points"),
        "score_chunk_overlap": (int(model.seq_len) - 1
                                if config["model_config"].get("score_chunk_points") else 0),
    }


def restore_from_bundle(bundle: dict[str, Any], device: str):
    config = bundle["config"]
    model = construct_official_couta(config, device)
    model.n_features = int(bundle["n_features_in"])
    model.net = construct_official_network(config, model.n_features, device)
    model.net.load_state_dict(bundle["model_state_dict"], strict=True)
    model.net.eval()
    model.c = bundle["center_c"].to(device)
    scaler = scaler_from_state(bundle["scaler"])
    return model, scaler
