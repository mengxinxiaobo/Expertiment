"""Label-free PSM data and score adapter for the unmodified SimAD model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "psm_simad_frozen.json"
Phase = Literal["train", "score"]


def load_frozen_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("dataset") != "PSM":
        raise ValueError("SimAD adapter requires the frozen PSM configuration")
    evaluation = config["evaluation"]
    for key in (
        "score_search",
        "ratio_search",
        "threshold_search",
        "parameter_search",
        "test_label_parameter_selection",
    ):
        if evaluation.get(key):
            raise ValueError(f"Frozen SimAD protocol must disable {key}")
    if float(evaluation["anomaly_ratio"]) != 0.8:
        raise ValueError("Frozen SimAD PSM anomaly_ratio must be 0.8")
    if float(evaluation["percentile"]) != 99.2:
        raise ValueError("Frozen SimAD PSM percentile must be 99.2")
    return config


class PSMWindowDataset(Dataset):
    """Complete fixed-stride windows with label-free placeholders."""

    def __init__(self, values: np.ndarray, window: int, stride: int) -> None:
        if values.ndim != 2 or values.shape[1] != 25:
            raise ValueError(f"Expected PSM [N,25], got {values.shape}")
        if values.dtype != np.float32:
            raise TypeError(f"Expected float32, got {values.dtype}")
        self.values = values
        self.window = int(window)
        self.stride = int(stride)
        self.count = (len(values) - self.window) // self.stride + 1
        if self.count <= 0:
            raise ValueError("PSM split is shorter than the SimAD window")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        start = int(index) * self.stride
        window = self.values[start : start + self.window]
        placeholder = np.zeros(self.window, dtype=np.float32)
        return window, placeholder


@dataclass
class SimADPSMDataAdapter:
    phase: Phase
    root: Path = ROOT
    config_path: Path = CONFIG_PATH

    def __post_init__(self) -> None:
        if self.phase not in ("train", "score"):
            raise ValueError(f"Unsupported phase: {self.phase}")
        self.config = load_frozen_config(self.config_path)
        self.data_dir = self.root / "dataset" / "PSM"
        self.files_accessed: list[str] = []
        self.training_label_access = False
        self.test_label_access = False
        self._train: np.ndarray | None = None
        self._test: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

    def _load(self, filename: str) -> np.ndarray:
        if filename == "PSM_test_label.npy":
            raise RuntimeError("SimAD model adapter is forbidden from reading labels")
        if self.phase == "train" and filename != "PSM_train.npy":
            raise RuntimeError(
                f"Training may only access PSM_train.npy, got {filename}"
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
        train = self._load("PSM_train.npy")
        expected = tuple(self.config["expected_shapes"]["train"])
        if tuple(train.shape) != expected:
            raise ValueError(f"Unexpected PSM train shape: {train.shape}")
        if not np.isfinite(train).all():
            raise ValueError("PSM train contains NaN or Inf")
        self._scaler = StandardScaler()
        self._scaler.fit(train)
        self._train = np.asarray(
            self._scaler.transform(train), dtype=np.float32
        )

    def _prepare_test(self) -> None:
        if self.phase != "score":
            raise RuntimeError("PSM test features are unavailable during training")
        if self._test is not None:
            return
        self._prepare_train()
        test = self._load("PSM_test.npy")
        expected = tuple(self.config["expected_shapes"]["test"])
        if tuple(test.shape) != expected:
            raise ValueError(f"Unexpected PSM test shape: {test.shape}")
        if not np.isfinite(test).all():
            raise ValueError("PSM test contains NaN or Inf")
        assert self._scaler is not None
        self._test = np.asarray(
            self._scaler.transform(test), dtype=np.float32
        )

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
        split: Literal["train", "test"],
        *,
        stride: int,
        shuffle: bool,
        batch_size: int | None = None,
    ) -> DataLoader:
        if split == "train":
            values = self.train_values
        elif split == "test":
            values = self.test_values
        else:
            raise ValueError(split)
        model = self.config["model"]
        training = self.config["training"]
        dataset = PSMWindowDataset(values, int(model["win_size"]), stride)
        return DataLoader(
            dataset,
            batch_size=int(batch_size or training["batch_size"]),
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    def official_loader(
        self,
        index,
        data_path,
        batch_size,
        win_size=2048,
        step=1,
        mode="train",
        dataset="PSM",
        dist=False,
        **_kwargs,
    ) -> DataLoader:
        del index, data_path, dist
        frozen = self.config
        if dataset != "PSM":
            raise ValueError(f"Expected PSM, got {dataset}")
        if int(win_size) != int(frozen["model"]["win_size"]):
            raise ValueError("SimAD window differs from frozen PSM protocol")
        if int(batch_size) != int(frozen["training"]["batch_size"]):
            raise ValueError("SimAD batch size differs from frozen PSM protocol")
        if self.phase == "train":
            # SimAD Trainer creates a `thres` loader in __init__. It is replaced
            # with train-only placeholders, and trainer.test is disabled.
            values = self.train_values
        elif mode == "train":
            values = self.train_values
        else:
            values = self.test_values
        stride = int(step) if mode == "train" else int(win_size)
        view = PSMWindowDataset(values, int(win_size), stride)
        return DataLoader(
            view,
            batch_size=int(batch_size),
            shuffle=(mode == "train" and self.phase == "train"),
            num_workers=0,
            drop_last=False,
        )

    def assert_training_label_free(self) -> None:
        expected = {"dataset/PSM/PSM_train.npy"}
        if set(self.files_accessed) != expected:
            raise AssertionError(
                f"Unexpected training files accessed: {self.files_accessed}"
            )
        if self.training_label_access or self.test_label_access:
            raise AssertionError("Training adapter accessed labels")

    def assert_model_phase_label_free(self) -> None:
        if self.training_label_access or self.test_label_access:
            raise AssertionError("SimAD model phase accessed labels")
        if any(path.endswith("PSM_test_label.npy") for path in self.files_accessed):
            raise AssertionError("Access log contains PSM_test_label.npy")


class SimADWindowScoreAdapter(nn.Module):
    """Official SimAD rec_score_func2 exposed as a [B,T] score callable."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.window_size = 2048
        self.model_name = "SimAD"

    def window_scores(self, windows: torch.Tensor) -> torch.Tensor:
        x_out, _sim_score = self.model(windows)
        patch = int(self.model.patch_size)
        channels = int(self.model.c_dim)
        x_patch = windows.reshape(
            windows.shape[0], windows.shape[1] // patch, patch * channels
        )
        l2_score = (x_out - x_patch).square()
        l2_score = l2_score.reshape(
            windows.shape[0], -1, patch, channels
        ).mean(dim=-1).reshape(windows.shape[0], -1)
        cosine_score = 1.0 - F.cosine_similarity(x_out, x_patch, dim=-1)
        cosine_score = F.interpolate(
            cosine_score.unsqueeze(1),
            size=l2_score.shape[1],
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        score = l2_score + cosine_score
        expected = (windows.shape[0], windows.shape[1])
        if tuple(score.shape) != expected:
            raise RuntimeError(
                f"Unexpected SimAD score shape {tuple(score.shape)}; "
                f"expected {expected}"
            )
        return score

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.window_scores(windows)
