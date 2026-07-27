#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/PUMP_PAPER_RESULTS"
LOG_PATH="${OUTPUT_ROOT}/run_detection.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

{
  echo "PUMP ASCA-AD V4 vs PPLAD vs LTFAD fixed-protocol detection"
  echo "seed=42"
  echo "standard_scaler=train_fit_only"
  echo "dtype=float32"
  echo "anomaly_ratio=0.5"
  echo "percentile=99.5"
  echo "threshold=independent_per_model"
  echo "score_search=disabled"
  echo "ratio_search=disabled"
  echo "threshold_search=disabled"
  echo "parameter_search=disabled"
  echo "oracle_search=disabled"
  echo "training_test_label_access=forbidden"
  echo "score_generation_test_label_access=forbidden"
  echo "ASCA_training=disabled"
  echo "PPLAD=fixed_fallback_configuration"
  echo "LTFAD=fixed_fallback_configuration"
  echo "overwrite=${OVERWRITE}"

  "${PYTHON_BIN}" scripts/benchmarks/build_pump_paper_results.py "${ARGS[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
