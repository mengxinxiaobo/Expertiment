#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT="$ROOT/results/ASCA_INFERENCE_OPTIMIZATION"
mkdir -p "$OUT/logs" "$OUT/summary"
LOG="$OUT/summary/run.log"
echo "ASCA-AD V4-IO Phase-1 gated optimization audit" | tee -a "$LOG"
echo "training=false labels_in_score=false historical_results_overwrite=false" | tee -a "$LOG"
"$PYTHON_BIN" scripts/benchmarks/run_asca_inference_optimization.py "$@" 2>&1 | tee -a "$LOG"
