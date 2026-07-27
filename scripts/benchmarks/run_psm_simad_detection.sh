#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/PSM_SIMAD_RESULTS"
LOG_PATH="${OUTPUT_ROOT}/run_detection.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

{
  echo "PSM SimAD fixed-protocol detection"
  echo "seed=42"
  echo "scaler=train_fit_only"
  echo "dtype=float32"
  echo "anomaly_ratio=0.8"
  echo "percentile=99.2"
  echo "score_search=disabled"
  echo "ratio_search=disabled"
  echo "threshold_search=disabled"
  echo "parameter_search=disabled"
  echo "training_label_access=forbidden"
  echo "training_test_access=forbidden"
  echo "SimAD_core_source=unmodified"
  echo "overwrite=${OVERWRITE}"

  "${PYTHON_BIN}" scripts/benchmarks/run_psm_simad_detection.py "${ARGS[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
