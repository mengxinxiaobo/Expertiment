"""Generate frozen ASCA-AD V4 scores for the SKAB PPLAD protocol.

This process never opens ``SKAB_test_label.npy`` and never calculates a
threshold or a metric.  It only performs the declared model forward pass and
restores window scores to the original train/test time axes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asca_ad.model import (  # noqa: E402
    AdaptiveSparseAnchorCompetitiveModelV4,
    AdaptiveSparseAnchorSolverV4,
)
from scripts.benchmarks.adapters import (  # noqa: E402
    ASCAV4ScoreAdapter,
    LabelFreeWindowDataset,
    collect_point_energy,
)


DATASET_DIR = ROOT / "dataset" / "SKAB"
CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "SKAB"
    / "SKAB_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
    "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
)
OUTPUT_ROOT = ROOT / "results" / "SKAB_BENCHMARK"
SCORE_DIR = OUTPUT_ROOT / "scores"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"

SEED = 42
WINDOW_SIZE = 100
STRIDE = 1
SCORE_MODE = "total"
ANOMALY_RATIO = 0.5
PERCENTILE = 99.5
EXPECTED_TRAIN_SHAPE = (12450, 8)
EXPECTED_TEST_SHAPE = (5710, 8)
EXPECTED_PARAMETERS = 146

MODEL_CONFIG: dict[str, Any] = {
    "local_candidate_lags": [1, 2, 3, 4, 5, 6, 7, 8],
    "global_candidate_lags": [12, 16, 20, 24, 28, 32, 40, 48],
    "local_topk": 2,
    "global_topk": 4,
    "selector_hidden": 8,
    "fitter_hidden": 8,
    "selector_temperature": 0.5,
    "similarity_tau": 1.0,
    "sigma_min": 0.03,
    "sigma_max": 1.5,
    "gap_weight": 1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {raw}")
    return device


def environment_record(device: torch.device) -> dict[str, Any]:
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu_name": gpu_name,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_feature_array(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, allow_pickle=False)
    if values.shape != expected_shape:
        raise ValueError(f"{path.name} shape {values.shape} != {expected_shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{path.name} must be numeric, got {values.dtype}")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains NaN or Inf")
    return values


def load_checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("ASCA checkpoint must be a dictionary")
    if "model" in checkpoint:
        state = checkpoint["model"]
        checkpoint_config = checkpoint.get("config", {})
    else:
        state = checkpoint
        checkpoint_config = {}
    if not isinstance(state, dict):
        raise TypeError("ASCA checkpoint model state must be a dictionary")
    if not isinstance(checkpoint_config, dict):
        raise TypeError("ASCA checkpoint config must be a dictionary")
    return state, checkpoint_config


def validate_checkpoint_config(checkpoint_config: dict[str, Any]) -> None:
    checks = {
        "dataset": "SKAB",
        "input_c": 8,
        "local_candidate_lags": MODEL_CONFIG["local_candidate_lags"],
        "global_candidate_lags": MODEL_CONFIG["global_candidate_lags"],
        "local_topk": MODEL_CONFIG["local_topk"],
        "global_topk": MODEL_CONFIG["global_topk"],
        "selector_hidden": MODEL_CONFIG["selector_hidden"],
        "fitter_hidden": MODEL_CONFIG["fitter_hidden"],
    }
    for key, expected in checks.items():
        if key in checkpoint_config and checkpoint_config[key] != expected:
            raise RuntimeError(
                f"Checkpoint config mismatch for {key}: "
                f"{checkpoint_config[key]!r} != {expected!r}"
            )


def build_inference_solver(
    model: AdaptiveSparseAnchorCompetitiveModelV4,
    device: torch.device,
) -> AdaptiveSparseAnchorSolverV4:
    # Deliberately bypass Solver.__init__: the parent initializer constructs the
    # official dataset loaders, which open SKAB_test_label.npy.  The scoring
    # methods below need only these frozen inference attributes.
    solver = object.__new__(AdaptiveSparseAnchorSolverV4)
    solver.model = model
    solver.device = device
    solver.win_size = WINDOW_SIZE
    solver.relation_input = "instance"
    solver.score_modes = [SCORE_MODE]
    solver.primary_score = SCORE_MODE
    solver.score_normalization = "official"
    return solver


def protocol_template(
    train_path: Path,
    test_path: Path,
    checkpoint_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "SKAB_ASCA_V4_PPLAD_PROTOCOL",
        "status": "score_generation_started",
        "dataset": {
            "name": "SKAB",
            "train": {
                "path": relative(train_path),
                "shape": list(EXPECTED_TRAIN_SHAPE),
                "sha256": sha256(train_path),
            },
            "test": {
                "path": relative(test_path),
                "shape": list(EXPECTED_TEST_SHAPE),
                "sha256": sha256(test_path),
            },
            # The evaluator, not this label-free generator, fills label metadata.
            "label": {
                "path": "dataset/SKAB/SKAB_test_label.npy",
                "shape": None,
                "sha256": None,
            },
        },
        "model": {
            "name": "ASCA-AD V4",
            "class": "asca_ad.model.AdaptiveSparseAnchorCompetitiveModelV4",
            "checkpoint": relative(CHECKPOINT),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_policy": "read_only_strict_load",
            "trainable_parameters": EXPECTED_PARAMETERS,
            **MODEL_CONFIG,
        },
        "scoring": {
            "seed": SEED,
            "input_dtype": "float32",
            "standard_scaler": "fit_train_transform_train_transform_test",
            "score_mode": SCORE_MODE,
            "score_selection": "predeclared_no_search",
            "score_normalization": "official_window_minmax_then_softmax",
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "overlap_aggregation": "mean",
            "energy_dtype": "float64",
            "uses_test_label": False,
        },
        "evaluation": {
            "protocol": "PPLAD_official_percentile_and_point_adjustment",
            "anomaly_ratio": ANOMALY_RATIO,
            "percentile": PERCENTILE,
            "threshold_source": "concat_train_energy_test_energy",
            "threshold_operator": "test_energy > threshold",
            "threshold": None,
            "parameter_search": False,
        },
        "environment": environment_record(device),
    }


def ensure_output_policy(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing benchmark artifacts. Use "
            f"--overwrite for an intentional full rerun:\n{names}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate label-free ASCA-AD V4 SKAB anomaly energy"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")

    device = resolve_device(args.device)
    set_seed(SEED)

    train_path = DATASET_DIR / "SKAB_train.npy"
    test_path = DATASET_DIR / "SKAB_test.npy"
    train_output = SCORE_DIR / "ASCA_train_energy.npy"
    test_output = SCORE_DIR / "ASCA_test_energy.npy"
    train_starts_output = SCORE_DIR / "ASCA_train_window_starts.npy"
    test_starts_output = SCORE_DIR / "ASCA_test_window_starts.npy"
    outputs = [
        train_output,
        test_output,
        train_starts_output,
        test_starts_output,
    ]
    ensure_output_policy(outputs, args.overwrite)

    checkpoint_hash = sha256(CHECKPOINT)
    protocol = protocol_template(train_path, test_path, checkpoint_hash, device)
    write_json(PROTOCOL_PATH, protocol)

    print("=" * 72)
    print("ASCA-AD V4 / SKAB score generation")
    print("=" * 72)
    print(f"checkpoint={CHECKPOINT}")
    print(f"checkpoint_sha256={checkpoint_hash}")
    print(f"device={device}")
    print(f"seed={SEED}")
    print(f"score_mode={SCORE_MODE}")
    print(f"window_size={WINDOW_SIZE}, stride={STRIDE}, aggregation=mean")
    print("test_label_access=False")

    train = load_feature_array(train_path, EXPECTED_TRAIN_SHAPE)
    test = load_feature_array(test_path, EXPECTED_TEST_SHAPE)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train).astype(np.float32, copy=False)
    test_scaled = scaler.transform(test).astype(np.float32, copy=False)
    if not np.isfinite(train_scaled).all() or not np.isfinite(test_scaled).all():
        raise RuntimeError("StandardScaler produced NaN or Inf")

    model = AdaptiveSparseAnchorCompetitiveModelV4(**MODEL_CONFIG)
    state, checkpoint_config = load_checkpoint_state(CHECKPOINT)
    validate_checkpoint_config(checkpoint_config)
    model.load_state_dict(state, strict=True)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"ASCA parameter count {parameter_count} != {EXPECTED_PARAMETERS}"
        )
    model = model.to(device)
    model.eval()

    runtime = build_inference_solver(model, device)
    adapter = ASCAV4ScoreAdapter(runtime)
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    train_dataset = LabelFreeWindowDataset(train_scaled, WINDOW_SIZE, STRIDE)
    test_dataset = LabelFreeWindowDataset(test_scaled, WINDOW_SIZE, STRIDE)
    train_loader = DataLoader(train_dataset, **loader_options)
    test_loader = DataLoader(test_dataset, **loader_options)

    train_scores = collect_point_energy(
        adapter,
        train_loader,
        total_length=EXPECTED_TRAIN_SHAPE[0],
        split="train",
        device=device,
    )
    test_scores = collect_point_energy(
        adapter,
        test_loader,
        total_length=EXPECTED_TEST_SHAPE[0],
        split="test",
        device=device,
    )

    assert len(train_scores.energy) == 12450
    assert len(test_scores.energy) == 5710
    assert train_scores.window_starts.size == len(train_dataset)
    assert test_scores.window_starts.size == len(test_dataset)

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(train_output, train_scores.energy, allow_pickle=False)
    np.save(test_output, test_scores.energy, allow_pickle=False)
    np.save(train_starts_output, train_scores.window_starts, allow_pickle=False)
    np.save(test_starts_output, test_scores.window_starts, allow_pickle=False)

    protocol["status"] = "scores_generated"
    protocol["scoring"]["batch_size"] = int(args.batch_size)
    protocol["scoring"]["num_workers"] = int(args.num_workers)
    protocol["scoring"]["train_window_count"] = int(len(train_dataset))
    protocol["scoring"]["test_window_count"] = int(len(test_dataset))
    protocol["scoring"]["train_window_score_shape_first_batch"] = list(
        train_scores.window_score_shape
    )
    protocol["scoring"]["test_window_score_shape_first_batch"] = list(
        test_scores.window_score_shape
    )
    protocol["scoring"]["train_energy_shape"] = list(train_scores.energy.shape)
    protocol["scoring"]["test_energy_shape"] = list(test_scores.energy.shape)
    protocol["scoring"]["train_coverage"] = [
        train_scores.coverage_min,
        train_scores.coverage_max,
    ]
    protocol["scoring"]["test_coverage"] = [
        test_scores.coverage_min,
        test_scores.coverage_max,
    ]
    protocol["artifacts"] = {
        "train_energy": relative(train_output),
        "test_energy": relative(test_output),
        "train_window_starts": relative(train_starts_output),
        "test_window_starts": relative(test_starts_output),
    }
    write_json(PROTOCOL_PATH, protocol)

    print(f"ASCA_train_energy.shape={train_scores.energy.shape}")
    print(f"ASCA_test_energy.shape={test_scores.energy.shape}")
    print(f"saved_scores={SCORE_DIR}")
    print(f"protocol={PROTOCOL_PATH}")


if __name__ == "__main__":
    main()
