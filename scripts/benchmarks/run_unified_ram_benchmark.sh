#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/results/UNIFIED_RAM_BENCHMARK"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LC_ALL=C

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

echo "Unified independent-process Peak RAM benchmark"
echo "peak_source=/usr/bin/time -v Maximum resident set size"
echo "warmup=3 full_test_repeat=1 num_workers=0"
echo "training=false labels=false threshold=false prediction=false evaluator=false"
echo "planned_cases=30 resume=true"

if [[ "${1:-}" == "--validate-only" ]]; then
  "${PYTHON_BIN}" scripts/benchmarks/run_unified_ram_benchmark.py --validate-only \
    2>&1 | tee "${OUTPUT_ROOT}/validation.log"
else
  "${PYTHON_BIN}" scripts/benchmarks/run_unified_ram_benchmark.py --run-all --resume "$@" \
    2>&1 | tee "${OUTPUT_ROOT}/run.log"
fi
