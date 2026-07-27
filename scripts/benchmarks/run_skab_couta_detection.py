#!/usr/bin/env python3
"""SKAB binding for the validated generic COUTA formal runner."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import scripts.benchmarks.run_psm_couta_detection as runner
from scripts.benchmarks.adapters.couta_dataset_adapter import (
    SKABCOUTADataAdapter, load_skab_couta_config,
)

runner.DATASET_NAME = "SKAB"
runner.PSMCOUTADataAdapter = SKABCOUTADataAdapter
runner.load_couta_config = load_skab_couta_config
runner.OUT = ROOT / "results" / "SKAB_COUTA_RESULTS"
runner.DET = runner.OUT / "Detection"
runner.SCORES = runner.OUT / "scores"
runner.CKPTS = runner.OUT / "checkpoints"
runner.BUNDLE = runner.CKPTS / "COUTA_SKAB_bundle.pt"
runner.STATE = runner.CKPTS / "COUTA_SKAB_state_dict.pt"
runner.TRAIN_SCORE = runner.SCORES / "train_score.npy"
runner.TEST_SCORE = runner.SCORES / "test_score.npy"
runner.TRAIN_AUDIT = runner.OUT / "training_audit.json"
runner.SCORE_AUDIT = runner.OUT / "score_audit.json"

if __name__ == "__main__": runner.main()
