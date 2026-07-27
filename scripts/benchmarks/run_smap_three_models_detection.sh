#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/SMAP_PAPER_RESULTS/Detection"
LOG_PATH="${OUTPUT_ROOT}/run_detection.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

if [[ -f "${LOG_PATH}" ]]; then
  mv "${LOG_PATH}" "${LOG_PATH}.failed_previous"
fi

{
  echo "SMAP ASCA-AD V4 vs PPLAD vs LTFAD formal detection experiment"
  echo "python=${PYTHON_BIN}"
  echo "seed=42"
  echo "input_dtype=float32"
  echo "standard_scaler=train_fit_only"
  echo "anomaly_ratio=0.8"
  echo "percentile=99.2"
  echo "threshold=independent per model"
  echo "score_search=disabled"
  echo "ratio_search=disabled"
  echo "test_label_parameter_selection=disabled"
  echo "old_runner=not_used"
  echo "efficiency_benchmark=disabled"

  "${PYTHON_BIN}" scripts/benchmarks/build_smap_paper_results.py
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
