#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/PSM_SIMAD_RESULTS/Efficiency"
LOG_PATH="${OUTPUT_ROOT}/run.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

{
  echo "PSM SimAD inference efficiency benchmark"
  echo "training=disabled"
  echo "labels=not_loaded"
  echo "evaluator=not_called"
  echo "dtype=float32"
  echo "window=2048"
  echo "warmup=30"
  echo "latency_repeat=200"
  echo "full_test_repeat=20"
  echo "data_loading=excluded"
  echo "host_to_device_transfer=excluded"
  echo "threshold=not_computed"
  echo "point_adjustment=not_computed"

  "${PYTHON_BIN}" scripts/benchmarks/benchmark_psm_simad_efficiency.py "${ARGS[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
