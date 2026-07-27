#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/PSM_PAPER_RESULTS/Efficiency"
LOG_PATH="${OUTPUT_ROOT}/run.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

{
  echo "PSM three-model inference efficiency benchmark"
  echo "detection_protocol_anomaly_ratio=0.8 (not used by timing)"
  echo "training=disabled"
  echo "labels=not_loaded"
  echo "evaluator=not_called"
  echo "warmup=30"
  echo "latency_repeat=200"
  echo "full_test_scope=window input -> model forward -> window score"
  echo "data_loading=excluded"
  echo "host_to_device_transfer=excluded"

  "${PYTHON_BIN}" scripts/benchmarks/benchmark_psm_efficiency.py
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
