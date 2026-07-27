#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/PSM_TRANAD_RESULTS"
LOG_PATH="${OUTPUT_ROOT}/run_detection.log"

mkdir -p "${OUTPUT_ROOT}"
if [[ -f "${LOG_PATH}" && -f "${OUTPUT_ROOT}/protocol.json" && "${OVERWRITE}" != "1" ]]; then
  echo "Existing formal log found: ${LOG_PATH}" >&2
  echo "Refusing to overwrite it. Use OVERWRITE=1 only for an authorized rerun." >&2
  exit 1
fi

ARGS=()
TEE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
elif [[ -f "${LOG_PATH}" ]]; then
  # Preserve the failed compatibility trace and append the clean retry.
  TEE_ARGS+=(-a)
fi

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

{
  echo "========================================================================"
  echo "Stage 3: PSM TranAD formal training and detection"
  echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "========================================================================"
  echo "dataset=PSM"
  echo "seed=42"
  echo "window=10 input_c=25 batch=128 epochs=5"
  echo "optimizer=Adam learning_rate=1e-4 dtype=float32"
  echo "standard_scaler=train_fit_only"
  echo "anomaly_ratio=0.8 percentile=99.2"
  echo "training_test_access=False"
  echo "training_label_access=False"
  echo "POT=disabled SPOT=disabled bf_search=disabled"
  echo "searches=disabled"
  "${PYTHON_BIN}" scripts/benchmarks/run_psm_tranad_detection.py "${ARGS[@]}"
  echo "========================================================================"
  echo "Stage 3 Finished"
  echo "End Time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "========================================================================"
} 2>&1 | tee "${TEE_ARGS[@]}" "${LOG_PATH}"
