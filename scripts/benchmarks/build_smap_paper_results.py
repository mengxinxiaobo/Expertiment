#!/usr/bin/env python3
"""Frozen SMAP three-model detection experiment without parameter search."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmarks.build_psm_paper_results as experiment  # noqa: E402


experiment.DATASET_NAME = "SMAP"
experiment.ENTRY_SCRIPT = Path(__file__).resolve()
experiment.OUTPUT_ROOT = ROOT / "results" / "SMAP_PAPER_RESULTS"
experiment.DETECTION_ROOT = experiment.OUTPUT_ROOT / "Detection"
experiment.PROTOCOL_PATH = experiment.DETECTION_ROOT / "protocol.json"
experiment.ASCA_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "SMAP"
    / "SMAP_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
    "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
)
experiment.ANOMALY_RATIO = 0.8
experiment.PERCENTILE = 99.2
experiment.EXPECTED_SHAPES = {
    "train": (135183, 25),
    "test": (427617, 25),
    "label": (427617,),
}
experiment.PPLAD_CONFIG_SOURCE = "BaselineModels/PPLAD-main/scripts/SMAP.sh"
experiment.LTFAD_CONFIG_SOURCE = (
    "pre-registered fixed SMAP configuration; official configuration unavailable"
)
base_asca_config = experiment.asca_config


def asca_config() -> dict[str, Any]:
    config = base_asca_config()
    config.update(
        {
            "dataset": "SMAP",
            "data_path": "SMAP",
            "input_c": 25,
            "output_c": 25,
            "win_size": 100,
            "batch_size": 128,
            "num_epochs": 10,
            "lr": 1e-3,
            "anormly_ratio": 0.8,
            "score_modes": ["total"],
            "primary_score": "total",
            "local_size": 3,
            "global_size": [5],
            "r": 0.9,
            "model_save_path": str(experiment.ASCA_CHECKPOINT.parent),
            "result_path": str(experiment.DETECTION_ROOT / "ASCA"),
        }
    )
    return config


def baseline_config(model: str) -> dict[str, Any]:
    name = "PPLAD" if model == "pplad" else "LTFAD"
    common = {
        "index": 137,
        "dataset": "SMAP",
        "data_path": "SMAP",
        "input_c": 25,
        "output_c": 25,
        "d_model": 128,
        "lr": 1e-4,
        "loss_fuc": "MSE",
        "anormly_ratio": 0.8,
        "model_save_path": str(experiment.DETECTION_ROOT / name),
    }
    if model == "pplad":
        return {
            **common,
            "win_size": 90,
            "batch_size": 128,
            "num_epochs": 3,
            "local_size": [3],
            "global_size": [5],
            "r": 0.9,
            "similar": "MSE",
        }
    return {
        **common,
        "win_size": 90,
        "batch_size": 128,
        "num_epochs": 2,
        "local_size": [5],
        "global_size": [13],
        "r": 0.1,
    }


experiment.asca_config = asca_config
experiment.baseline_config = baseline_config


if __name__ == "__main__":
    experiment.main()
