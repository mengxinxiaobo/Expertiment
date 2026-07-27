"""Validate all frozen SKAB score artifacts before unified evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCORE_DIR = ROOT / "results" / "SKAB_BENCHMARK" / "scores"

WINDOWS = {"ASCA": 100, "PPLAD": 60, "LTFAD": 90}
TRAIN_LENGTH = 12450
TEST_LENGTH = 5710


def load_array(name: str) -> np.ndarray:
    path = SCORE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, allow_pickle=False)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    return values


def validate_model(model: str, window: int) -> None:
    train_energy = load_array(f"{model}_train_energy.npy")
    test_energy = load_array(f"{model}_test_energy.npy")
    train_starts = load_array(f"{model}_train_window_starts.npy")
    test_starts = load_array(f"{model}_test_window_starts.npy")

    expected_train_starts = TRAIN_LENGTH - window + 1
    expected_test_starts = TEST_LENGTH - window + 1
    expected_shapes = {
        "train_energy": (TRAIN_LENGTH,),
        "test_energy": (TEST_LENGTH,),
        "train_starts": (expected_train_starts,),
        "test_starts": (expected_test_starts,),
    }
    actual = {
        "train_energy": train_energy.shape,
        "test_energy": test_energy.shape,
        "train_starts": train_starts.shape,
        "test_starts": test_starts.shape,
    }
    if actual != expected_shapes:
        raise RuntimeError(f"{model} score shape mismatch: {actual} != {expected_shapes}")
    for name, starts, final_start in (
        ("train", train_starts, TRAIN_LENGTH - window),
        ("test", test_starts, TEST_LENGTH - window),
    ):
        if starts.dtype.kind not in {"i", "u"}:
            raise TypeError(f"{model} {name} starts must be integer")
        expected = np.arange(final_start + 1, dtype=starts.dtype)
        if not np.array_equal(starts, expected):
            raise RuntimeError(f"{model} {name} starts do not implement stride=1")

    assert len(train_energy) == TRAIN_LENGTH
    assert len(test_energy) == TEST_LENGTH
    print(
        f"{model}: train_energy={train_energy.shape}, "
        f"test_energy={test_energy.shape}, window={window}, stride=1"
    )


def main() -> None:
    for model, window in WINDOWS.items():
        validate_model(model, window)
    print("Three-model score validation passed")
    print("all_test_energy_lengths_equal_5710=True")


if __name__ == "__main__":
    main()
