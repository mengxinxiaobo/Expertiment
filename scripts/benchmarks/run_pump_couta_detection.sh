#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/results/PUMP_COUTA_RESULTS"
mkdir -p "$OUT"
echo "PUMP COUTA official fixed-protocol detection"
echo "epochs=20 ratio=0.5 percentile=99.5 score_chunk_points=20000 searches=false"
python3 scripts/benchmarks/run_pump_couta_detection.py "$@" 2>&1 | tee "$OUT/run_detection.log"
