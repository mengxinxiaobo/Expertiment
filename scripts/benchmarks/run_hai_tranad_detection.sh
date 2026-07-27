#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/HAI_TRANAD_RESULTS"
LOG_PATH="${OUTPUT_ROOT}/run_detection.log"

mkdir -p "${OUTPUT_ROOT}"
if [[ -f "${LOG_PATH}" && -f "${OUTPUT_ROOT}/protocol.json" && "${OVERWRITE}" != "1" ]]; then
  echo "Existing formal HAI TranAD result found; refusing overwrite." >&2
  exit 1
fi

ARGS=()
TEE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
elif [[ -f "${LOG_PATH}" ]]; then
  TEE_ARGS+=(-a)
fi

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
{
  echo "========================================================================"
  echo "HAI TranAD fixed-protocol training and detection"
  echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "========================================================================"
  echo "seed=42 window=10 input_c=86 batch=128 epochs=5"
  echo "optimizer=Adam learning_rate=1e-4 dtype=float32"
  echo "standard_scaler=train_fit_only"
  echo "anomaly_ratio=0.98 percentile=99.02"
  echo "training_test_access=False training_label_access=False"
  echo "POT=False SPOT=False bf_search=False searches=False"
  "${PYTHON_BIN}" scripts/benchmarks/run_hai_tranad_detection.py "${ARGS[@]}"
  echo "========================================================================"
  echo "HAI TranAD Detection Finished"
  echo "End Time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "========================================================================"
} 2>&1 | tee "${TEE_ARGS[@]}" "${LOG_PATH}"
