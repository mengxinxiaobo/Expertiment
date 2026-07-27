#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_DIR="${PROJECT_ROOT}/results/SKAB_BENCHMARK"
LOG_PATH="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

GENERATOR_ARGS=(
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  GENERATOR_ARGS+=(--overwrite)
fi

{
  echo "SKAB ASCA-AD V4 PPLAD protocol"
  echo "project_root=${PROJECT_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "batch_size=${BATCH_SIZE}"
  echo "num_workers=${NUM_WORKERS}"
  echo "overwrite=${OVERWRITE}"

  "${PYTHON_BIN}" \
    scripts/benchmarks/generate_asca_v4_skab_score.py \
    "${GENERATOR_ARGS[@]}"

  "${PYTHON_BIN}" \
    scripts/benchmarks/validate_skab_asca_scores.py

  "${PYTHON_BIN}" \
    scripts/benchmarks/evaluate_skab_asca_pplad_protocol.py
} 2>&1 | tee "${LOG_PATH}"

echo "Completed. Log: ${LOG_PATH}"
