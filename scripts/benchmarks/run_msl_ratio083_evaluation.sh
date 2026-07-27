#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/MSL_PAPER_RESULTS"
DETECTION_NAME="Detection_ratio083"
LOG_PATH="${OUTPUT_ROOT}/run_msl_ratio083_evaluation.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

{
  echo "MSL three-model evaluation with official PPLAD MSL anomaly ratio"
  echo "python=${PYTHON_BIN}"
  echo "anomaly_ratio=0.83"
  echo "percentile=99.17"
  echo "training=skipped"
  echo "baseline_checkpoints=results/MSL_PAPER_RESULTS/Detection"
  echo "output=${OUTPUT_ROOT}/${DETECTION_NAME}"

  "${PYTHON_BIN}" scripts/benchmarks/build_msl_paper_results.py \
    --evaluate-existing \
    --output-name "${DETECTION_NAME}"
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
