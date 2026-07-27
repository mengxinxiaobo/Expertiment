#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"; OUT="$ROOT/results/HAI_COUTA_RESULTS"; mkdir -p "$OUT"
echo "HAI COUTA official fixed-protocol detection"; echo "epochs=20 ratio=0.98 percentile=99.02 searches=false"
python3 scripts/benchmarks/run_hai_couta_detection.py "$@" 2>&1 | tee "$OUT/run_detection.log"
