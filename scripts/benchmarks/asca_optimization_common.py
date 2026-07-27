"""Shared, non-invasive utilities for the ASCA V4 inference optimization audit."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "ASCA_INFERENCE_OPTIMIZATION"
DATASETS = ("SKAB", "MSL", "PSM", "HAI", "PUMP", "SMD")
ANOMALY_RATIOS = {"SKAB": 0.5, "MSL": 0.83, "PSM": 0.8, "HAI": 0.98, "PUMP": 0.5, "SMD": 0.9}
EXPECTED_SHAPES = {
    "SKAB": ((12450, 8), (5710, 8), (5710,)),
    "MSL": ((58317, 55), (73729, 55), (73729,)),
    "PSM": ((132481, 25), (87841, 25), (87841,)),
    "HAI": ((896400, 86), (284400, 86), (284400,)),
    "PUMP": ((17155, 51), (203165, 51), (203165,)),
    "SMD": ((708405, 38), (708420, 38), (708420,)),
}
WINDOW = 100
BATCH_SIZE = 128
SEED = 42

PROTECTED_FILES = (
    ROOT / "asca_ad" / "model.py",
    ROOT / "scripts" / "benchmarks" / "adapters" / "asca_adapter.py",
)
PROTECTED_TREES = (
    ROOT / "results" / "UNIFIED_RAM_BENCHMARK",
    ROOT / "results" / "MEMORY_ROOT_CAUSE_AUDIT",
    ROOT / "results" / "UNIFIED_EFFICIENCY_COMPARISON",
)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "entries": 0, "digest": None}
    digest = hashlib.sha256()
    entries = 0
    for item in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: str(p)):
        stat = item.stat()
        relative = str(item.relative_to(path)).replace("\\", "/")
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        entries += 1
    return {"exists": True, "entries": entries, "digest": digest.hexdigest()}


def protected_snapshot() -> dict[str, Any]:
    historical_results = {
        str(path.relative_to(ROOT)).replace("\\", "/"): tree_signature(path)
        for path in sorted((ROOT / "results").iterdir(), key=lambda p: p.name)
        if path.is_dir() and path.resolve() != OUT.resolve()
    } if (ROOT / "results").is_dir() else {}
    return {
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
            }
            for path in PROTECTED_FILES
        },
        "trees": {
            str(path.relative_to(ROOT)).replace("\\", "/"): tree_signature(path)
            for path in PROTECTED_TREES
        },
        "all_historical_result_trees": historical_results,
    }


def git_snapshot() -> dict[str, Any]:
    commands = {
        "status_short": ["git", "status", "--short"],
        "diff_stat": ["git", "diff", "--stat"],
        "existing_model_diff": ["git", "diff", "--", "asca_ad/model.py"],
        "existing_adapter_diff": ["git", "diff", "--", "scripts/benchmarks/adapters/asca_adapter.py"],
    }
    result: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        result[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; automatic CPU fallback is forbidden")
    return torch.device("cuda:0")


def import_protocol(dataset: str):
    if dataset == "SKAB":
        return importlib.import_module("scripts.benchmarks.benchmark_skab_efficiency")
    wrapper = importlib.import_module(f"scripts.benchmarks.benchmark_{dataset.lower()}_efficiency")
    wrapper.configure_protocol()
    return wrapper.protocol


def load_old_model(dataset: str, device: torch.device):
    protocol = import_protocol(dataset)
    display, window, model, score_call, checkpoint = protocol.load_model("asca", device)
    if display != "ASCA-AD V4" or int(window) != WINDOW:
        raise RuntimeError(f"unexpected formal ASCA identity/window: {display}/{window}")
    model.eval()
    return protocol, model, score_call, Path(checkpoint)


def optimized_config_from_old(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "local_candidate_lags": [int(v) for v in model.local_lags.detach().cpu().tolist()],
        "global_candidate_lags": [int(v) for v in model.global_lags.detach().cpu().tolist()],
        "local_topk": int(model.local_topk),
        "global_topk": int(model.global_topk),
        "selector_hidden": int(model.selector.network[0].out_features),
        "fitter_hidden": int(model.fitter.network[0].out_features),
        "selector_temperature": float(model.selector_temperature),
        "similarity_tau": float(model.similarity_tau),
        "sigma_min": float(model.fitter.sigma_min),
        "sigma_max": float(model.fitter.sigma_max),
        "gap_weight": float(model.gap_weight),
        "window_size": WINDOW,
    }


def extract_checkpoint_state(checkpoint: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        metadata = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        return payload["model"], metadata
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        metadata = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        return payload["state_dict"], metadata
    if isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload, metadata
    raise RuntimeError(f"unsupported ASCA checkpoint payload: {checkpoint}")


def build_optimized_from_old(old_model: torch.nn.Module, checkpoint: Path, device: torch.device):
    from asca_ad_optimized import ASCAInferenceOptimized, ASCASolverInferenceOptimized
    from scripts.benchmarks.adapters.asca_optimized_adapter import ASCAOptimizedScoreAdapter

    model = ASCAInferenceOptimized(**optimized_config_from_old(old_model))
    state, _metadata = extract_checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict optimized load failed: {incompatible}")
    model = model.to(device).eval()
    solver = ASCASolverInferenceOptimized(model, device, WINDOW, "instance", "official")
    adapter = ASCAOptimizedScoreAdapter(solver).eval()
    return model, adapter.window_scores


def state_dict_kib(model: torch.nn.Module) -> float:
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return len(stream.getvalue()) / 1024.0


def checkpoint_compatibility(dataset: str, device: torch.device) -> dict[str, Any]:
    _protocol, old_model, _old_score, checkpoint = load_old_model(dataset, device)
    state, metadata = extract_checkpoint_state(checkpoint)
    old_load = old_model.load_state_dict(state, strict=True)
    optimized, _new_score = build_optimized_from_old(old_model, checkpoint, device)
    old_state, new_state = old_model.state_dict(), optimized.state_dict()
    old_keys, new_keys = list(old_state), list(new_state)
    shapes_equal = old_keys == new_keys and all(old_state[k].shape == new_state[k].shape for k in old_keys)
    old_params = sum(p.numel() for p in old_model.parameters())
    new_params = sum(p.numel() for p in optimized.parameters())
    persistent_names = set(new_state)
    cache_names = [name for name, _ in optimized.named_buffers() if name.startswith("_")]
    cache_nonpersistent = all(name not in persistent_names for name in cache_names)
    result = {
        "dataset": dataset,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_metadata": metadata,
        "old_strict_missing_keys": list(old_load.missing_keys),
        "old_strict_unexpected_keys": list(old_load.unexpected_keys),
        "optimized_strict_missing_keys": [],
        "optimized_strict_unexpected_keys": [],
        "state_dict_keys_equal": old_keys == new_keys,
        "state_dict_shapes_equal": shapes_equal,
        "old_parameters": int(old_params),
        "optimized_parameters": int(new_params),
        "old_state_dict_kib": state_dict_kib(old_model),
        "optimized_state_dict_kib": state_dict_kib(optimized),
        "nonpersistent_cache_names": cache_names,
        "nonpersistent_caches_excluded_from_state_dict": cache_nonpersistent,
    }
    result["status"] = "PASS" if (
        shapes_equal and old_params == new_params == 146 and cache_nonpersistent
        and not old_load.missing_keys and not old_load.unexpected_keys
    ) else "FAIL"
    return result


def dataset_paths(dataset: str) -> tuple[Path, Path, Path]:
    base = ROOT / "dataset" / dataset
    return base / f"{dataset}_train.npy", base / f"{dataset}_test.npy", base / f"{dataset}_test_label.npy"


def load_scaled_data(dataset: str, include_label: bool = False):
    train_path, test_path, label_path = dataset_paths(dataset)
    train = np.asarray(np.load(train_path, allow_pickle=False), dtype=np.float32)
    test = np.asarray(np.load(test_path, allow_pickle=False), dtype=np.float32)
    expected_train, expected_test, expected_label = EXPECTED_SHAPES[dataset]
    if train.shape != expected_train or test.shape != expected_test:
        raise RuntimeError(f"unexpected {dataset} shape: {train.shape}/{test.shape}")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError(f"{dataset} contains NaN/Inf")
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train).astype(np.float32, copy=False)
    test_scaled = scaler.transform(test).astype(np.float32, copy=False)
    if not include_label:
        return train_scaled, test_scaled, None
    label = np.asarray(np.load(label_path, allow_pickle=False)).reshape(-1).astype(np.int64)
    if label.shape != expected_label or not set(np.unique(label)).issubset({0, 1}):
        raise RuntimeError(f"invalid {dataset} label")
    return train_scaled, test_scaled, label


def formal_starts(length: int, window: int, split: str) -> list[int]:
    step = 1 if split == "train" else window
    if length < window:
        raise RuntimeError("sequence shorter than ASCA window")
    return list(range(0, length - window + 1, step))


def full_coverage_starts(length: int, window: int = WINDOW) -> list[int]:
    starts = list(range(0, length - window + 1, window))
    final_start = length - window
    if starts[-1] != final_start:
        starts.append(final_start)
    coverage = np.zeros(length, dtype=np.bool_)
    for start in starts:
        coverage[start : start + window] = True
    if not coverage.all():
        raise RuntimeError("full-test tail policy did not cover the complete timeline")
    return starts


def cpu_batches(data: np.ndarray, starts: list[int], batch_size: int = BATCH_SIZE) -> Iterator[torch.Tensor]:
    for offset in range(0, len(starts), batch_size):
        current = starts[offset : offset + batch_size]
        values = np.stack([data[start : start + WINDOW] for start in current]).astype(np.float32, copy=False)
        yield torch.from_numpy(np.ascontiguousarray(values))


def point_adjust(prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
    prediction = prediction.copy()
    anomaly_state = False
    for index in range(len(labels)):
        if labels[index] == 1 and prediction[index] == 1 and not anomaly_state:
            anomaly_state = True
            for backward in range(index, 0, -1):
                if labels[backward] == 0:
                    break
                prediction[backward] = 1
            for forward in range(index, len(labels)):
                if labels[forward] == 0:
                    break
                prediction[forward] = 1
        elif labels[index] == 0:
            anomaly_state = False
        if anomaly_state:
            prediction[index] = 1
    return prediction


def array_equivalence(old_path: Path, new_path: Path, diff_path: Path, chunk: int = 2_000_000) -> dict[str, Any]:
    old = np.load(old_path, mmap_mode="r")
    new = np.load(new_path, mmap_mode="r")
    if old.shape != new.shape or old.dtype != new.dtype:
        return {"status": "FAIL", "shape_equal": False, "dtype_equal": old.dtype == new.dtype}
    diff = np.lib.format.open_memmap(diff_path, mode="w+", dtype=np.float32, shape=old.shape)
    max_abs = mean_abs_sum = max_rel = mean_rel_sum = 0.0
    exact = finite = nan_count = inf_count = 0
    total = int(old.size)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        a = np.asarray(old[start:stop], dtype=np.float64)
        b = np.asarray(new[start:stop], dtype=np.float64)
        delta = np.abs(a - b)
        diff[start:stop] = delta.astype(np.float32)
        denominator = np.maximum(np.abs(a), 1e-12)
        relative = delta / denominator
        max_abs = max(max_abs, float(delta.max(initial=0.0)))
        mean_abs_sum += float(delta.sum())
        max_rel = max(max_rel, float(relative.max(initial=0.0)))
        mean_rel_sum += float(relative.sum())
        exact += int(np.count_nonzero(a == b))
        finite += int(np.count_nonzero(np.isfinite(a) & np.isfinite(b)))
        nan_count += int(np.count_nonzero(np.isnan(a)) + np.count_nonzero(np.isnan(b)))
        inf_count += int(np.count_nonzero(np.isinf(a)) + np.count_nonzero(np.isinf(b)))
    diff.flush()
    allclose = bool(max_abs <= 1e-7)
    return {
        "status": "PASS" if allclose and nan_count == 0 and inf_count == 0 else "FAIL",
        "shape": list(old.shape), "shape_equal": True, "dtype": str(old.dtype), "dtype_equal": True,
        "finite_pair_count": finite, "nan_count_both_arrays": nan_count,
        "inf_count_both_arrays": inf_count, "max_absolute_error": max_abs,
        "mean_absolute_error": mean_abs_sum / max(total, 1), "max_relative_error": max_rel,
        "mean_relative_error": mean_rel_sum / max(total, 1), "exact_equality_count": exact,
        "array_equal": exact == total, "allclose_atol_1e-7_rtol_1e-6": allclose,
    }
