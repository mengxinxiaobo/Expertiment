#!/usr/bin/env python3
"""SKAB binding for the COUTA inference-only efficiency benchmark."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import scripts.benchmarks.benchmark_psm_couta_efficiency as benchmark
from scripts.benchmarks.adapters.couta_dataset_adapter import (
    SKABCOUTADataAdapter, load_skab_couta_config,
)

benchmark.DATASET_NAME = "SKAB"
benchmark.CSV_INCLUDE_INCREMENTAL = False
benchmark.PSMCOUTADataAdapter = SKABCOUTADataAdapter
benchmark.load_couta_config = load_skab_couta_config
benchmark.OUT = ROOT / "results" / "SKAB_COUTA_RESULTS"
benchmark.EFF = benchmark.OUT / "Efficiency"
benchmark.BUNDLE = benchmark.OUT / "checkpoints" / "COUTA_SKAB_bundle.pt"

if __name__ == "__main__": benchmark.main()
