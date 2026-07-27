#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p results/MEMORY_ROOT_CAUSE_AUDIT/logs
LOG=results/MEMORY_ROOT_CAUSE_AUDIT/logs/run.log

echo "Memory root-cause audit: independent formal-peak and object-audit processes" | tee -a "$LOG"
echo "training=false labels=false threshold=false prediction=false evaluator=false" | tee -a "$LOG"
python3 scripts/benchmarks/run_memory_root_cause_audit.py --run-all --resume "$@" 2>&1 | tee -a "$LOG"
