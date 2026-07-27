#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/PSM_TRANAD_RESULTS/Efficiency"
LOG_PATH="${OUTPUT_ROOT}/efficiency.log"

mkdir -p "${OUTPUT_ROOT}"
if [[ -f "${LOG_PATH}" && "${OVERWRITE}" != "1" ]]; then
  echo "Existing TranAD efficiency log found: ${LOG_PATH}" >&2
  echo "Use OVERWRITE=1 only for an explicitly authorized rerun." >&2
  exit 1
fi

ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

{
  echo "========================================================================"
  echo "PSM TranAD inference-only Efficiency Benchmark"
  echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "========================================================================"
  echo "training=False"
  echo "label_access=False"
  echo "evaluator_called=False"
  echo "threshold_computed=False prediction_generated=False"
  echo "POT=False SPOT=False bf_search=False"
  echo "dtype=float32 window=10 input_c=25"
  echo "warmup=30 latency_repeat=200 full_test_repeat=20"
  echo "data_loading=excluded"
  echo "host_to_device_transfer=excluded"
  "${PYTHON_BIN}" scripts/benchmarks/benchmark_psm_tranad_efficiency.py "${ARGS[@]}"
  echo "========================================================================"
  echo "Efficiency Benchmark Finished"
  echo "End Time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "========================================================================"
} 2>&1 | tee "${LOG_PATH}"
