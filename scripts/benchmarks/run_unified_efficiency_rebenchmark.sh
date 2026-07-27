#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/results/UNIFIED_EFFICIENCY_REBENCHMARK"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

echo "Unified audit-driven efficiency rebenchmark"
echo "training=false labels=false evaluator=false threshold=false detection=false"
echo "required_cases=28 retained_cases=2"
echo "full_test_repeat=20 warmup=30"
echo "all_windows_on_gpu=false"
echo "legacy_results_overwritten=false"
echo "output=${OUTPUT_ROOT}"

"${PYTHON_BIN}" scripts/benchmarks/run_unified_efficiency_rebenchmark.py --validate-only
"${PYTHON_BIN}" scripts/benchmarks/run_unified_efficiency_rebenchmark.py "$@" 2>&1 | tee "${OUTPUT_ROOT}/run.log"
