#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/results/ASCA_INFERENCE_OPTIMIZATION_V2"
mkdir -p "$OUT/summary" "$OUT/logs"
echo "ASCA-AD V4-IO2 gated optimization audit"
echo "training=false historical_results_overwrite=false"
python3 "$ROOT/scripts/benchmarks/run_asca_v4_io2_optimization.py" "$@" \
  2>&1 | tee -a "$OUT/summary/run.log"
