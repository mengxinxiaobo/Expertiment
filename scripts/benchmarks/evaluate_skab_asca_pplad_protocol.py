"""Evaluate frozen ASCA SKAB energy with PPLAD's fixed protocol."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "results" / "SKAB_BENCHMARK"
SCORE_DIR = OUTPUT_ROOT / "scores"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
METRICS_JSON = OUTPUT_ROOT / "detection_metrics.json"
METRICS_CSV = OUTPUT_ROOT / "detection_metrics.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.md"
LABEL_PATH = ROOT / "dataset" / "SKAB" / "SKAB_test_label.npy"
PPLAD_PA_PATH = (
    ROOT
    / "BaselineModels"
    / "PPLAD-main"
    / "metrics"
    / "f1_score_f1_pa.py"
)

ANOMALY_RATIO = 0.5
PERCENTILE = 99.5
EXPECTED_TRAIN_LENGTH = 12450
EXPECTED_TEST_LENGTH = 5710


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
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_official_pa_module() -> ModuleType:
    if not PPLAD_PA_PATH.is_file():
        raise FileNotFoundError(PPLAD_PA_PATH)
    spec = importlib.util.spec_from_file_location(
        "skab_benchmark_pplad_official_f1_pa", PPLAD_PA_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load PPLAD PA module: {PPLAD_PA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_energy(path: Path, expected_length: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, allow_pickle=False).reshape(-1)
    if len(values) != expected_length:
        raise RuntimeError(f"{path.name} length {len(values)} != {expected_length}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path.name} contains NaN or Inf")
    return np.asarray(values, dtype=np.float64)


def load_labels(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    source = np.load(path, allow_pickle=False)
    source_shape = tuple(int(value) for value in source.shape)
    labels = source.reshape(-1)
    if len(labels) != EXPECTED_TEST_LENGTH:
        raise RuntimeError(
            f"SKAB_test_label length {len(labels)} != {EXPECTED_TEST_LENGTH}"
        )
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"SKAB labels must be binary, got {sorted(unique)}")
    return labels.astype(np.int64, copy=False), source_shape


def binary_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _support = precision_recall_fscore_support(
        labels,
        prediction,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment") != "SKAB_ASCA_V4_PPLAD_PROTOCOL":
        raise RuntimeError("Unexpected or missing experiment in protocol.json")
    scoring = protocol.get("scoring", {})
    evaluation = protocol.get("evaluation", {})
    if scoring.get("score_mode") != "total":
        raise RuntimeError("protocol score_mode must be frozen to 'total'")
    if scoring.get("window_size") != 100 or scoring.get("stride") != 1:
        raise RuntimeError("protocol window_size/stride must be 100/1")
    if evaluation.get("anomaly_ratio") != ANOMALY_RATIO:
        raise RuntimeError("protocol anomaly_ratio must be frozen to 0.5")
    if evaluation.get("percentile") != PERCENTILE:
        raise RuntimeError("protocol percentile must be frozen to 99.5")
    if evaluation.get("parameter_search") is not False:
        raise RuntimeError("protocol must explicitly disable parameter search")


def write_csv(metrics: dict[str, Any]) -> None:
    row = {
        "Model": metrics["model"],
        "Threshold": metrics["threshold"],
        "Accuracy": metrics["raw"]["accuracy"],
        "Precision": metrics["raw"]["precision"],
        "Recall": metrics["raw"]["recall"],
        "F1": metrics["raw"]["f1"],
        "PA-Accuracy": metrics["pa"]["accuracy"],
        "PA-Precision": metrics["pa"]["precision"],
        "PA-Recall": metrics["pa"]["recall"],
        "PA-F1": metrics["pa"]["f1"],
    }
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_summary(metrics: dict[str, Any]) -> None:
    raw = metrics["raw"]
    pa = metrics["pa"]
    lines = [
        "# ASCA-AD V4 — SKAB — PPLAD Protocol",
        "",
        "## Frozen protocol",
        "",
        "- Score mode: `total`",
        "- Window / stride: `100 / 1`",
        "- Overlap aggregation: `mean`",
        "- Anomaly ratio: `0.5`",
        "- Threshold: `percentile(concat(train_energy, test_energy), 99.5)`",
        "- Parameter, threshold, ratio and score search: `disabled`",
        "",
        "## Score validation",
        "",
        f"- Train energy shape: `{metrics['score_shapes']['train']}`",
        f"- Test energy shape: `{metrics['score_shapes']['test']}`",
        f"- Label shape: `{metrics['score_shapes']['label']}`",
        f"- Test length equals label length: `{metrics['test_length_matches_label']}`",
        f"- Threshold: `{metrics['threshold']:.12g}`",
        "",
        "## Detection metrics",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | PA-Accuracy | PA-Precision | PA-Recall | PA-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| ASCA-AD V4 | {raw['accuracy']:.6f} | {raw['precision']:.6f} | "
            f"{raw['recall']:.6f} | {raw['f1']:.6f} | {pa['accuracy']:.6f} | "
            f"{pa['precision']:.6f} | {pa['recall']:.6f} | {pa['f1']:.6f} |"
        ),
        "",
        "PA is produced by `BaselineModels/PPLAD-main/metrics/f1_score_f1_pa.py`.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    protocol = load_json(PROTOCOL_PATH)
    validate_protocol(protocol)

    train_energy = load_energy(
        SCORE_DIR / "ASCA_train_energy.npy", EXPECTED_TRAIN_LENGTH
    )
    test_energy = load_energy(
        SCORE_DIR / "ASCA_test_energy.npy", EXPECTED_TEST_LENGTH
    )
    labels, label_file_shape = load_labels(LABEL_PATH)

    assert len(train_energy) == 12450
    assert len(test_energy) == 5710
    assert len(labels) == 5710
    if not (len(test_energy) == len(labels) == EXPECTED_TEST_LENGTH):
        raise RuntimeError("ASCA test energy and SKAB label lengths do not match")

    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = float(np.percentile(combined_energy, PERCENTILE))
    raw_prediction = (test_energy > threshold).astype(np.int64)
    raw = binary_metrics(labels, raw_prediction)

    official_pa = load_official_pa_module()
    adjusted_prediction = raw_prediction.copy()
    official_values = official_pa.get_adjust_F1PA(adjusted_prediction, labels)
    pa = binary_metrics(labels, adjusted_prediction)
    official_names = ("accuracy", "precision", "recall", "f1")
    for name, official_value in zip(official_names, official_values):
        if not np.isclose(pa[name], float(official_value), rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"PPLAD PA parity check failed for {name}: "
                f"{pa[name]} != {float(official_value)}"
            )

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = PREDICTION_DIR / "ASCA_pred_raw.npy"
    pa_path = PREDICTION_DIR / "ASCA_pred_pa.npy"
    np.save(raw_path, raw_prediction, allow_pickle=False)
    np.save(pa_path, adjusted_prediction, allow_pickle=False)

    metrics = {
        "model": "ASCA-AD V4",
        "dataset": "SKAB",
        "score_mode": "total",
        "anomaly_ratio": ANOMALY_RATIO,
        "percentile": PERCENTILE,
        "threshold": threshold,
        "threshold_source": "concat(train_energy,test_energy)",
        "prediction_rule": "test_energy > threshold",
        "raw": raw,
        "pa": pa,
        "point_adjustment": {
            "implementation": relative(PPLAD_PA_PATH),
            "implementation_sha256": sha256(PPLAD_PA_PATH),
            "function": "get_adjust_F1PA",
        },
        "score_shapes": {
            "train": list(train_energy.shape),
            "test": list(test_energy.shape),
            "label": list(labels.shape),
            "label_file": list(label_file_shape),
        },
        "test_length_matches_label": bool(len(test_energy) == len(labels) == 5710),
        "parameter_search": False,
        "artifacts": {
            "raw_prediction": relative(raw_path),
            "pa_prediction": relative(pa_path),
        },
    }
    write_json(METRICS_JSON, metrics)
    write_csv(metrics)
    write_summary(metrics)

    protocol["status"] = "evaluation_completed"
    protocol["dataset"]["label"] = {
        "path": relative(LABEL_PATH),
        "shape": list(labels.shape),
        "file_shape": list(label_file_shape),
        "sha256": sha256(LABEL_PATH),
    }
    protocol["evaluation"]["threshold"] = threshold
    protocol["evaluation"]["raw_metrics"] = raw
    protocol["evaluation"]["pa_metrics"] = pa
    protocol["evaluation"]["point_adjustment"] = {
        "implementation": relative(PPLAD_PA_PATH),
        "implementation_sha256": sha256(PPLAD_PA_PATH),
        "function": "get_adjust_F1PA",
    }
    protocol["artifacts"].update(metrics["artifacts"])
    protocol["artifacts"].update(
        {
            "detection_metrics_json": relative(METRICS_JSON),
            "detection_metrics_csv": relative(METRICS_CSV),
            "summary": relative(SUMMARY_PATH),
        }
    )
    write_json(PROTOCOL_PATH, protocol)

    print("=" * 72)
    print("ASCA-AD V4 / SKAB / PPLAD evaluation")
    print("=" * 72)
    print(f"train_energy.shape={train_energy.shape}")
    print(f"test_energy.shape={test_energy.shape}")
    print(f"label.shape={labels.shape}")
    print(f"test_energy_length_equals_5710={len(test_energy) == 5710}")
    print(f"anomaly_ratio={ANOMALY_RATIO}")
    print(f"threshold={threshold:.12g}")
    print("RAW " + json.dumps(raw, ensure_ascii=False, sort_keys=True))
    print("PA  " + json.dumps(pa, ensure_ascii=False, sort_keys=True))
    print(f"metrics_json={METRICS_JSON}")
    print(f"metrics_csv={METRICS_CSV}")
    print(f"summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
