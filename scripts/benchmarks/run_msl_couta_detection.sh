#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"; OUT="$ROOT/results/MSL_COUTA_RESULTS"; mkdir -p "$OUT"
echo "MSL COUTA official fixed-protocol detection"; echo "epochs=20 ratio=0.83 percentile=99.17 searches=false"
python3 scripts/benchmarks/run_msl_couta_detection.py "$@" 2>&1 | tee "$OUT/run_detection.log"
