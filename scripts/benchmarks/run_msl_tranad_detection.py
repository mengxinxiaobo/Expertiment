#!/usr/bin/env python3
"""MSL binding for the shared frozen TranAD detection implementation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmarks.run_psm_tranad_detection as runner
from scripts.benchmarks.adapters.tranad_dataset_adapter import (
    TranADMSLDataAdapter,
    load_tranad_msl_config,
)


OUTPUT_ROOT = ROOT / "results" / "MSL_TRANAD_RESULTS"


def configure() -> None:
    runner.OUTPUT_ROOT = OUTPUT_ROOT
    runner.DETECTION_ROOT = OUTPUT_ROOT / "Detection"
    runner.SCORES_ROOT = OUTPUT_ROOT / "scores"
    runner.CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
    runner.CHECKPOINT_PATH = (
        runner.CHECKPOINT_ROOT / "TranAD_MSL_state_dict.pt"
    )
    runner.TRAIN_SCORE_PATH = runner.SCORES_ROOT / "train_score.npy"
    runner.TEST_SCORE_PATH = runner.SCORES_ROOT / "test_score.npy"
    runner.COMPARISON_CSV = (
        runner.DETECTION_ROOT / "comparison_detection.csv"
    )
    runner.COMPARISON_JSON = (
        runner.DETECTION_ROOT / "comparison_detection.json"
    )
    runner.PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
    runner.LABEL_PATH = ROOT / "dataset" / "MSL" / "MSL_test_label.npy"
    runner.DATASET_NAME = "MSL"
    runner.EXPECTED_TRAIN_POINTS = 58317
    runner.EXPECTED_TEST_POINTS = 73729
    runner.TranADPSMDataAdapter = TranADMSLDataAdapter
    runner.load_tranad_psm_config = load_tranad_msl_config


if __name__ == "__main__":
    configure()
    runner.main()
