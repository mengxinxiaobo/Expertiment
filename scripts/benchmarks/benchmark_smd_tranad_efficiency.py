#!/usr/bin/env python3
"""SMD binding for the shared TranAD inference-efficiency benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmarks.benchmark_psm_tranad_efficiency as benchmark
from scripts.benchmarks.adapters.tranad_dataset_adapter import (
    TranADSMDDataAdapter,
    load_tranad_smd_config,
)


RESULT_ROOT = ROOT / "results" / "SMD_TRANAD_RESULTS"


def configure() -> None:
    detection_protocol_path = RESULT_ROOT / "protocol.json"
    if not detection_protocol_path.is_file():
        raise FileNotFoundError(
            "Run the formal SMD TranAD detection experiment before efficiency"
        )
    detection_protocol = json.loads(
        detection_protocol_path.read_text(encoding="utf-8")
    )
    checkpoint = detection_protocol["checkpoint"]
    benchmark.OUTPUT_ROOT = RESULT_ROOT / "Efficiency"
    benchmark.COMPARISON_CSV = benchmark.OUTPUT_ROOT / "comparison_efficiency.csv"
    benchmark.COMPARISON_JSON = benchmark.OUTPUT_ROOT / "comparison_efficiency.json"
    benchmark.PROTOCOL_PATH = benchmark.OUTPUT_ROOT / "protocol_efficiency.json"
    benchmark.CHECKPOINT_PATH = ROOT / checkpoint["path"]
    benchmark.EXPECTED_CHECKPOINT_SHA256 = checkpoint["sha256"]
    benchmark.EXPECTED_TEST_POINTS = 708420
    benchmark.DATASET_NAME = "SMD"
    benchmark.WINDOW = 10
    benchmark.INPUT_C = 38
    benchmark.EXPECTED_PARAMETERS = 127538
    benchmark.TranADPSMDataAdapter = TranADSMDDataAdapter
    benchmark.load_tranad_psm_config = load_tranad_smd_config


if __name__ == "__main__":
    if not any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        configure()
    benchmark.main()
