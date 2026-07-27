#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/MSL_PAPER_RESULTS"
LOG_PATH="${OUTPUT_ROOT}/run_msl_three_models.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

{
  echo "MSL ASCA-AD V4 vs PPLAD vs LTFAD paper experiment"
  echo "python=${PYTHON_BIN}"
  echo "anomaly_ratio=0.83"
  echo "threshold=per-model percentile(concat(train_energy,test_energy),99.17)"
  echo "parameter_search=disabled"

  "${PYTHON_BIN}" scripts/benchmarks/build_msl_paper_results.py
  "${PYTHON_BIN}" scripts/benchmarks/benchmark_msl_efficiency.py
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
