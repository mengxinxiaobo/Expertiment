"""Validate ASCA SKAB score artifacts without reading anomaly labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCORE_DIR = ROOT / "results" / "SKAB_BENCHMARK" / "scores"

EXPECTED = {
    "ASCA_train_energy.npy": (12450,),
    "ASCA_test_energy.npy": (5710,),
    "ASCA_train_window_starts.npy": (12351,),
    "ASCA_test_window_starts.npy": (5611,),
}


def load_exact(name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    path = SCORE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, allow_pickle=False)
    if values.shape != expected_shape:
        raise RuntimeError(f"{name} shape {values.shape} != {expected_shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    return values


def validate_starts(starts: np.ndarray, final_start: int, name: str) -> None:
    if starts.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integer indices, got {starts.dtype}")
    if int(starts[0]) != 0 or int(starts[-1]) != final_start:
        raise RuntimeError(
            f"{name} boundary starts must be 0..{final_start}, got "
            f"{int(starts[0])}..{int(starts[-1])}"
        )
    if not np.array_equal(np.diff(starts), np.ones(len(starts) - 1, dtype=np.int64)):
        raise RuntimeError(f"{name} must have frozen stride=1")


def main() -> None:
    arrays = {
        name: load_exact(name, shape) for name, shape in EXPECTED.items()
    }
    validate_starts(
        arrays["ASCA_train_window_starts.npy"],
        final_start=12450 - 100,
        name="ASCA_train_window_starts.npy",
    )
    validate_starts(
        arrays["ASCA_test_window_starts.npy"],
        final_start=5710 - 100,
        name="ASCA_test_window_starts.npy",
    )

    assert len(arrays["ASCA_train_energy.npy"]) == 12450
    assert len(arrays["ASCA_test_energy.npy"]) == 5710
    print("ASCA score artifact validation passed")
    for name in EXPECTED:
        print(f"{name}.shape={arrays[name].shape}")
    print("test_label_access=False")


if __name__ == "__main__":
    main()
