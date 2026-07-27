#!/usr/bin/env python3
"""PUMP binding for the audited generic DeepOD COUTA formal runner."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmarks.run_psm_couta_detection as runner
from scripts.benchmarks.adapters.couta_dataset_adapter import (
    PUMPCOUTADataAdapter, load_pump_couta_config,
)

runner.DATASET_NAME = "PUMP"
runner.PSMCOUTADataAdapter = PUMPCOUTADataAdapter
runner.load_couta_config = load_pump_couta_config
runner.OUT = ROOT / "results" / "PUMP_COUTA_RESULTS"
runner.DET = runner.OUT / "Detection"
runner.SCORES = runner.OUT / "scores"
runner.CKPTS = runner.OUT / "checkpoints"
runner.BUNDLE = runner.CKPTS / "COUTA_PUMP_bundle.pt"
runner.STATE = runner.CKPTS / "COUTA_PUMP_state_dict.pt"
runner.TRAIN_SCORE = runner.SCORES / "train_score.npy"
runner.TEST_SCORE = runner.SCORES / "test_score.npy"
runner.TRAIN_AUDIT = runner.OUT / "training_audit.json"
runner.SCORE_AUDIT = runner.OUT / "score_audit.json"

DETECTION_FIELDS = [
    "Model", "RAW Accuracy", "RAW Precision", "RAW Recall", "RAW F1",
    "PA Accuracy", "PA Precision", "PA Recall", "PA F1", "Threshold",
    "Anomaly Ratio", "Percentile",
]


def verify_protocol(path: Path) -> None:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("dataset") != "PUMP":
        raise RuntimeError(f"Non-PUMP protocol cannot be aggregated: {path}")
    evaluation = protocol.get("evaluation", protocol.get("config", {}).get("evaluation", {}))
    if float(evaluation.get("anomaly_ratio", -1)) != 0.5:
        raise RuntimeError(f"PUMP anomaly_ratio mismatch: {path}")
    if float(evaluation.get("percentile", -1)) != 99.5:
        raise RuntimeError(f"PUMP percentile mismatch: {path}")


def aggregate_detection() -> None:
    sources = [
        (ROOT / "results" / "PUMP_PAPER_RESULTS" / "protocol.json",
         ROOT / "results" / "PUMP_PAPER_RESULTS" / "Detection" / "comparison_detection.csv"),
        (ROOT / "results" / "PUMP_TRANAD_RESULTS" / "protocol.json",
         ROOT / "results" / "PUMP_TRANAD_RESULTS" / "Detection" / "comparison_detection.csv"),
        (runner.OUT / "protocol.json", runner.DET / "comparison_detection.csv"),
    ]
    rows: list[dict[str, object]] = []
    for protocol_path, metrics_path in sources:
        verify_protocol(protocol_path)
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "Model": row["Model"], "RAW Accuracy": row["Accuracy"],
                    "RAW Precision": row["Precision"], "RAW Recall": row["Recall"],
                    "RAW F1": row["F1"], "PA Accuracy": row["PA-Accuracy"],
                    "PA Precision": row["PA-Precision"], "PA Recall": row["PA-Recall"],
                    "PA F1": row["PA-F1"], "Threshold": row["Threshold"],
                    "Anomaly Ratio": 0.5, "Percentile": 99.5,
                })
    expected = {"ASCA-AD V4", "PPLAD", "LTFAD", "TranAD", "COUTA"}
    if {str(row["Model"]) for row in rows} != expected:
        raise RuntimeError(f"Unexpected PUMP model set: {[row['Model'] for row in rows]}")
    path = runner.OUT / "five_model_detection_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"five_model_detection={path}", flush=True)


def finalize_protocol() -> None:
    path = runner.OUT / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["audit"].update({
        "deepod_core_modified": False, "psm_results_modified": False,
        "skab_results_modified": False, "msl_results_modified": False,
        "hai_results_modified": False, "ray_installed": False,
        "ray_stub_used": True, "fit_auto_hyper_used": False,
        "deepod_testbed_used": False, "ts_metrics_used": False,
        "internal_threshold_used_for_paper": False,
        "internal_prediction_used": False, "contamination_prediction_used": False,
    })
    path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    runner.main()
    finalize_protocol()
    aggregate_detection()
