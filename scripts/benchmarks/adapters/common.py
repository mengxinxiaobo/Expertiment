"""Shared, label-free windowing and time-axis restoration utilities.

This module deliberately knows nothing about anomaly labels, thresholds, or
metrics.  It preserves the ``(window, label, start)`` batch contract expected
by the benchmark adapters, but the label is an all-zero placeholder created in
memory.  ``SKAB_test_label.npy`` is never opened by this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowScoreAdapter(Protocol):
    """Minimal interface implemented by every model-specific score adapter."""

    model_name: str
    window_size: int

    def eval(self) -> "WindowScoreAdapter": ...

    def window_scores(self, windows: torch.Tensor) -> torch.Tensor: ...


class LabelFreeWindowDataset(Dataset):
    """Create fully covering sliding windows without reading anomaly labels.

    The last possible start index, ``length - window_size``, is always present.
    With the frozen SKAB score stride of one it is naturally included; the
    explicit append also makes the full-coverage invariant hold for other
    diagnostic strides.
    """

    def __init__(self, data: np.ndarray, window_size: int, stride: int = 1) -> None:
        values = np.asarray(data, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(f"Expected [time, channels] data, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("Input data contains NaN or Inf")
        if window_size <= 0 or window_size > len(values):
            raise ValueError(
                f"Invalid window_size={window_size} for length={len(values)}"
            )
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        last_start = len(values) - int(window_size)
        starts = list(range(0, last_start + 1, int(stride)))
        if starts[-1] != last_start:
            starts.append(last_start)

        self.data = np.ascontiguousarray(values)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        stop = start + self.window_size
        window = torch.from_numpy(self.data[start:stop])
        # Preserve the official two-item data convention while carrying start.
        # This placeholder contains no information from SKAB_test_label.npy.
        placeholder = torch.zeros(self.window_size, dtype=torch.float32)
        return window, placeholder, start


@dataclass(frozen=True)
class ScoreCollection:
    """Point-level energy plus the exact windows used to produce it."""

    energy: np.ndarray
    window_starts: np.ndarray
    window_score_shape: tuple[int, ...]
    coverage_min: int
    coverage_max: int


def _validate_window_score_shape(
    scores: torch.Tensor,
    batch_size: int,
    window_size: int,
    model_name: str,
) -> None:
    expected = (int(batch_size), int(window_size))
    actual = tuple(int(value) for value in scores.shape)
    if actual != expected:
        raise RuntimeError(
            f"{model_name} must return one score per timestamp: expected "
            f"window_score.shape={expected}, got {actual}. Window-level [B] "
            "scores cannot be mapped by point-wise overlap averaging."
        )
    if not torch.isfinite(scores).all():
        raise RuntimeError(f"{model_name} window scores contain NaN or Inf")


def aggregate_point_scores(
    window_scores: np.ndarray,
    window_starts: np.ndarray,
    total_length: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore timestamp energy by averaging all covering window scores."""

    scores = np.asarray(window_scores, dtype=np.float64)
    starts = np.asarray(window_starts, dtype=np.int64).reshape(-1)
    expected = (starts.size, int(window_size))
    if scores.shape != expected:
        raise ValueError(f"Expected window scores {expected}, got {scores.shape}")
    if total_length <= 0:
        raise ValueError("total_length must be positive")
    if not np.isfinite(scores).all():
        raise ValueError("Window scores contain NaN or Inf")

    totals = np.zeros(int(total_length), dtype=np.float64)
    coverage = np.zeros(int(total_length), dtype=np.int64)
    for row, start in zip(scores, starts):
        stop = int(start) + int(window_size)
        if start < 0 or stop > total_length:
            raise ValueError(
                f"Window [{int(start)}, {stop}) is outside length {total_length}"
            )
        totals[int(start):stop] += row
        coverage[int(start):stop] += 1

    if np.any(coverage == 0):
        missing = np.flatnonzero(coverage == 0)
        raise RuntimeError(
            f"Time-axis restoration left {missing.size} timestamps uncovered; "
            f"first missing index={int(missing[0])}"
        )

    energy = totals / coverage
    if energy.shape != (int(total_length),):
        raise AssertionError(f"Unexpected restored energy shape: {energy.shape}")
    return energy, coverage


@torch.no_grad()
def collect_point_energy(
    adapter: WindowScoreAdapter,
    loader: Iterable,
    total_length: int,
    split: str,
    device: torch.device | str,
    print_diagnostics: bool = True,
) -> ScoreCollection:
    """Collect `[B,T]` scores and restore a complete point-level time axis."""

    adapter.eval()
    score_parts: list[np.ndarray] = []
    start_parts: list[np.ndarray] = []
    first_input_shape: tuple[int, ...] | None = None
    first_score_shape: tuple[int, ...] | None = None
    window_count = 0

    for windows, _placeholder, starts in loader:
        windows = windows.float().to(device, non_blocking=True)
        scores = adapter.window_scores(windows)
        _validate_window_score_shape(
            scores,
            batch_size=windows.shape[0],
            window_size=adapter.window_size,
            model_name=adapter.model_name,
        )

        starts_array = torch.as_tensor(starts).detach().cpu().numpy().reshape(-1)
        if starts_array.size != windows.shape[0]:
            raise RuntimeError(
                f"{adapter.model_name} start count {starts_array.size} does not "
                f"match batch size {windows.shape[0]}"
            )

        if first_input_shape is None:
            first_input_shape = tuple(int(value) for value in windows.shape)
            first_score_shape = tuple(int(value) for value in scores.shape)
        score_parts.append(scores.detach().cpu().numpy())
        start_parts.append(starts_array.astype(np.int64, copy=False))
        window_count += int(windows.shape[0])

    if not score_parts or first_input_shape is None or first_score_shape is None:
        raise RuntimeError(f"{adapter.model_name} {split} loader produced no windows")

    all_scores = np.concatenate(score_parts, axis=0)
    all_starts = np.concatenate(start_parts, axis=0)
    energy, coverage = aggregate_point_scores(
        all_scores,
        all_starts,
        total_length=total_length,
        window_size=adapter.window_size,
    )
    if len(energy) != int(total_length):
        raise AssertionError(f"energy length {len(energy)} != {total_length}")

    if print_diagnostics:
        print(f"[{adapter.model_name}][{split}] input_shape={first_input_shape}")
        print(f"[{adapter.model_name}][{split}] window_count={window_count}")
        print(f"[{adapter.model_name}][{split}] window_score_shape={first_score_shape}")
        print(f"[{adapter.model_name}][{split}] output_energy_length={len(energy)}")
        print(
            f"[{adapter.model_name}][{split}] coverage="
            f"{int(coverage.min())}..{int(coverage.max())}"
        )

    return ScoreCollection(
        energy=energy,
        window_starts=all_starts,
        window_score_shape=first_score_shape,
        coverage_min=int(coverage.min()),
        coverage_max=int(coverage.max()),
    )
