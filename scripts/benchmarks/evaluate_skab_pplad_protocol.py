"""Unified PPLAD-protocol evaluation for ASCA-AD V4, PPLAD, and LTFAD."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "results" / "SKAB_BENCHMARK"
SCORE_DIR = OUTPUT_ROOT / "scores"
PREDICTION_DIR = OUTPUT_ROOT / "three_model_predictions"
CSV_PATH = OUTPUT_ROOT / "three_model_detection_metrics.csv"
JSON_PATH = OUTPUT_ROOT / "three_model_detection_metrics.json"
SUMMARY_PATH = OUTPUT_ROOT / "three_model_summary.md"
PROTOCOL_PATH = OUTPUT_ROOT / "three_model_protocol.json"
ASCA_METRICS_PATH = OUTPUT_ROOT / "detection_metrics.json"
ASCA_PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PPLAD_METADATA_PATH = OUTPUT_ROOT / "PPLAD_score_metadata.json"
LTFAD_METADATA_PATH = OUTPUT_ROOT / "LTFAD_score_metadata.json"
LABEL_PATH = ROOT / "dataset" / "SKAB" / "SKAB_test_label.npy"
PA_PATH = (
    ROOT / "BaselineModels" / "PPLAD-main" / "metrics" / "f1_score_f1_pa.py"
)

MODELS = (
    ("ASCA-AD V4", "ASCA"),
    ("PPLAD", "PPLAD"),
    ("LTFAD", "LTFAD"),
)
TRAIN_LENGTH = 12450
TEST_LENGTH = 5710
ANOMALY_RATIO = 0.5
PERCENTILE = 99.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_pa_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("unified_pplad_official_pa", PA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official PPLAD PA: {PA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_energy(path: Path, length: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, allow_pickle=False).reshape(-1)
    if values.shape != (length,):
        raise RuntimeError(f"{path.name} shape {values.shape} != {(length,)}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path.name} contains NaN or Inf")
    return np.asarray(values, dtype=np.float64)


def load_labels() -> tuple[np.ndarray, tuple[int, ...]]:
    source = np.load(LABEL_PATH, allow_pickle=False)
    source_shape = tuple(int(value) for value in source.shape)
    labels = source.reshape(-1)
    if labels.shape != (TEST_LENGTH,):
        raise RuntimeError(f"Label shape {labels.shape} != {(TEST_LENGTH,)}")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise RuntimeError("SKAB labels must be binary")
    return labels.astype(np.int64, copy=False), source_shape


def binary_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_model(
    display_name: str,
    file_prefix: str,
    labels: np.ndarray,
    pa_module: ModuleType,
) -> dict[str, Any]:
    train_path = SCORE_DIR / f"{file_prefix}_train_energy.npy"
    test_path = SCORE_DIR / f"{file_prefix}_test_energy.npy"
    train_energy = load_energy(train_path, TRAIN_LENGTH)
    test_energy = load_energy(test_path, TEST_LENGTH)
    assert len(test_energy) == len(labels) == 5710

    threshold = float(
        np.percentile(np.concatenate([train_energy, test_energy]), PERCENTILE)
    )
    raw_prediction = (test_energy > threshold).astype(np.int64)
    raw = binary_metrics(labels, raw_prediction)
    adjusted_prediction = raw_prediction.copy()
    official_values = pa_module.get_adjust_F1PA(adjusted_prediction, labels)
    pa = binary_metrics(labels, adjusted_prediction)
    for key, official_value in zip(
        ("accuracy", "precision", "recall", "f1"), official_values
    ):
        if not np.isclose(pa[key], float(official_value), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{display_name} official PA parity failed for {key}")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = PREDICTION_DIR / f"{file_prefix}_pred_raw.npy"
    pa_path = PREDICTION_DIR / f"{file_prefix}_pred_pa.npy"
    np.save(raw_path, raw_prediction, allow_pickle=False)
    np.save(pa_path, adjusted_prediction, allow_pickle=False)
    return {
        "model": display_name,
        "threshold": threshold,
        "accuracy": raw["accuracy"],
        "precision": raw["precision"],
        "recall": raw["recall"],
        "f1": raw["f1"],
        "pa_accuracy": pa["accuracy"],
        "pa_precision": pa["precision"],
        "pa_recall": pa["recall"],
        "pa_f1": pa["f1"],
        "score_shapes": {"train": [TRAIN_LENGTH], "test": [TEST_LENGTH]},
        "score_sha256": {
            "train": sha256(train_path),
            "test": sha256(test_path),
        },
        "predictions": {
            "raw": relative(raw_path),
            "pa": relative(pa_path),
        },
    }


def assert_asca_unchanged(result: dict[str, Any]) -> None:
    previous = load_json(ASCA_METRICS_PATH)
    pairs = [(result["threshold"], previous["threshold"])]
    for key in ("accuracy", "precision", "recall", "f1"):
        pairs.append((result[key], previous["raw"][key]))
        pairs.append((result[f"pa_{key}"], previous["pa"][key]))
    if not all(np.isclose(left, right, rtol=0.0, atol=1e-12) for left, right in pairs):
        raise RuntimeError("Unified evaluator does not reproduce frozen ASCA result")


def write_csv(results: list[dict[str, Any]]) -> None:
    fields = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "PA-Accuracy",
        "PA-Precision",
        "PA-Recall",
        "PA-F1",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "Model": item["model"],
                    "Accuracy": item["accuracy"],
                    "Precision": item["precision"],
                    "Recall": item["recall"],
                    "F1": item["f1"],
                    "PA-Accuracy": item["pa_accuracy"],
                    "PA-Precision": item["pa_precision"],
                    "PA-Recall": item["pa_recall"],
                    "PA-F1": item["pa_f1"],
                }
            )


def write_summary(results: list[dict[str, Any]]) -> None:
    lines = [
        "# SKAB Three-Model Detection - Unified PPLAD Protocol",
        "",
        "- Anomaly ratio: `0.5`",
        "- Per-model threshold: `percentile(concat(train_energy, test_energy), 99.5)`",
        "- Point adjustment: official PPLAD `get_adjust_F1PA`",
        "- Threshold, anomaly ratio, and score search: disabled",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | PA-Accuracy | PA-Precision | PA-Recall | PA-F1 | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['model']} | {item['accuracy']:.6f} | "
            f"{item['precision']:.6f} | {item['recall']:.6f} | "
            f"{item['f1']:.6f} | {item['pa_accuracy']:.6f} | "
            f"{item['pa_precision']:.6f} | {item['pa_recall']:.6f} | "
            f"{item['pa_f1']:.6f} | {item['threshold']:.12g} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    asca_protocol = load_json(ASCA_PROTOCOL_PATH)
    pplad_metadata = load_json(PPLAD_METADATA_PATH)
    ltfad_metadata = load_json(LTFAD_METADATA_PATH)
    if asca_protocol.get("status") != "evaluation_completed":
        raise RuntimeError("Frozen ASCA protocol is not complete")
    for name, metadata in (("PPLAD", pplad_metadata), ("LTFAD", ltfad_metadata)):
        if metadata.get("status") != "scores_generated":
            raise RuntimeError(f"{name} score metadata is incomplete")
        if metadata.get("parameter_search") is not False:
            raise RuntimeError(f"{name} metadata must disable parameter search")
        if metadata.get("threshold_calculated") is not False:
            raise RuntimeError(f"{name} generator must not calculate thresholds")

    labels, label_file_shape = load_labels()
    pa_module = load_pa_module()
    results = [
        evaluate_model(display, prefix, labels, pa_module)
        for display, prefix in MODELS
    ]
    assert_asca_unchanged(results[0])
    assert all(item["score_shapes"]["test"] == [5710] for item in results)

    payload = {
        "dataset": "SKAB",
        "anomaly_ratio": ANOMALY_RATIO,
        "percentile": PERCENTILE,
        "threshold_source": "per_model_concat(train_energy,test_energy)",
        "prediction_rule": "test_energy > model_threshold",
        "point_adjustment": {
            "implementation": relative(PA_PATH),
            "implementation_sha256": sha256(PA_PATH),
            "function": "get_adjust_F1PA",
        },
        "label": {
            "path": relative(LABEL_PATH),
            "shape": [TEST_LENGTH],
            "file_shape": list(label_file_shape),
            "sha256": sha256(LABEL_PATH),
        },
        "results": results,
        "all_test_energy_lengths_equal_5710": True,
        "parameter_search": False,
    }
    write_json(JSON_PATH, payload)
    write_csv(results)
    write_summary(results)

    protocol = {
        "schema_version": 1,
        "experiment": "SKAB_THREE_MODEL_PPLAD_PROTOCOL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": asca_protocol["dataset"],
        "models": {
            "ASCA-AD V4": {
                "source_protocol": relative(ASCA_PROTOCOL_PATH),
                "source_protocol_sha256": sha256(ASCA_PROTOCOL_PATH),
                "reused_existing_scores": True,
                "configuration": asca_protocol["model"],
                "scoring": asca_protocol["scoring"],
            },
            "PPLAD": pplad_metadata,
            "LTFAD": ltfad_metadata,
        },
        "evaluation": {
            "anomaly_ratio": ANOMALY_RATIO,
            "percentile": PERCENTILE,
            "threshold_source": "per_model_concat(train_energy,test_energy)",
            "thresholds": {item["model"]: item["threshold"] for item in results},
            "point_adjustment": payload["point_adjustment"],
            "parameter_search": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {
            "metrics_csv": relative(CSV_PATH),
            "metrics_json": relative(JSON_PATH),
            "summary": relative(SUMMARY_PATH),
        },
    }
    write_json(PROTOCOL_PATH, protocol)

    print("=" * 72)
    print("SKAB three-model unified PPLAD evaluation")
    print("=" * 72)
    for item in results:
        print(
            f"{item['model']}: threshold={item['threshold']:.12g}, "
            f"RAW_F1={item['f1']:.12g}, PA_F1={item['pa_f1']:.12g}, "
            "test_energy_length=5710"
        )
    print("all_test_energy_lengths_equal_5710=True")
    print(f"metrics_csv={CSV_PATH}")
    print(f"metrics_json={JSON_PATH}")
    print(f"protocol={PROTOCOL_PATH}")


if __name__ == "__main__":
    main()
