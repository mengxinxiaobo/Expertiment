#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/PSM_PAPER_RESULTS"
LOG_PATH="${OUTPUT_ROOT}/run_detection.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

{
  echo "PSM ASCA-AD V4 vs PPLAD vs LTFAD formal detection experiment"
  echo "python=${PYTHON_BIN}"
  echo "seed=42"
  echo "anomaly_ratio=0.8"
  echo "percentile=99.2"
  echo "threshold=independent per model"
  echo "score_search=disabled"
  echo "ratio_search=disabled"
  echo "test_label_parameter_selection=disabled"
  echo "efficiency_benchmark=disabled"

  "${PYTHON_BIN}" scripts/benchmarks/build_psm_paper_results.py
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
