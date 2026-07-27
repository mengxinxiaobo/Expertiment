#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
OUT="$ROOT/results/SKAB_COUTA_RESULTS"; mkdir -p "$OUT"
echo "SKAB COUTA official fixed-protocol detection"
echo "formal_epochs=20 anomaly_ratio=0.5 percentile=99.5 searches=false"
python3 scripts/benchmarks/run_skab_couta_detection.py "$@" 2>&1 | tee "$OUT/run_detection.log"
