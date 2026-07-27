#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
OUT="$ROOT/results/SMD_COUTA_RESULTS"; mkdir -p "$OUT"
echo "SMD COUTA official fixed-protocol detection"
echo "epochs=20 ratio=0.9 percentile=99.1 chunk=20000 searches=false"
python3 scripts/benchmarks/run_smd_couta_detection.py "$@" 2>&1 | tee "$OUT/run_detection.log"
