#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OVERWRITE="${OVERWRITE:-0}"
RETRAIN="${RETRAIN:-0}"
OUTPUT_DIR="${PROJECT_ROOT}/results/SKAB_BENCHMARK"
LOG_PATH="${OUTPUT_DIR}/three_model_run.log"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

BASELINE_ARGS=(--device "${DEVICE}" --batch-size 128 --num-workers "${NUM_WORKERS}")
if [[ "${OVERWRITE}" == "1" ]]; then
  BASELINE_ARGS+=(--overwrite-scores)
fi
if [[ "${RETRAIN}" == "1" ]]; then
  BASELINE_ARGS+=(--retrain --overwrite-scores)
fi

{
  echo "SKAB ASCA-AD V4 vs PPLAD vs LTFAD - unified PPLAD protocol"
  echo "ASCA score generation is intentionally skipped"
  echo "project_root=${PROJECT_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "num_workers=${NUM_WORKERS}"
  echo "overwrite=${OVERWRITE}"
  echo "retrain=${RETRAIN}"

  "${PYTHON_BIN}" scripts/benchmarks/generate_pplad_skab_score.py \
    "${BASELINE_ARGS[@]}"

  "${PYTHON_BIN}" scripts/benchmarks/generate_ltfad_skab_score.py \
    "${BASELINE_ARGS[@]}"

  "${PYTHON_BIN}" scripts/benchmarks/validate_skab_three_model_scores.py

  "${PYTHON_BIN}" scripts/benchmarks/evaluate_skab_pplad_protocol.py
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
