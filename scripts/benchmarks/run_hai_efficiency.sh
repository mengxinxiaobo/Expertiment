#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/HAI_PAPER_RESULTS/Efficiency"
LOG_PATH="${OUTPUT_ROOT}/run.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

{
  echo "HAI three-model inference efficiency benchmark"
  echo "training=disabled"
  echo "labels=not_loaded"
  echo "evaluator=not_called"
  echo "dtype=float32"
  echo "warmup=30"
  echo "latency_repeat=200"
  echo "full_test_repeat=20"
  echo "full_test_scope=window input -> model forward -> anomaly window score"
  echo "data_loading=excluded"
  echo "host_to_device_transfer=excluded"

  "${PYTHON_BIN}" scripts/benchmarks/benchmark_hai_efficiency.py
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
