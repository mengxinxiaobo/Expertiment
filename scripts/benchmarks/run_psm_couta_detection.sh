#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/results/PSM_COUTA_RESULTS"
mkdir -p "$OUT"
echo "PSM COUTA official fixed-protocol detection"
echo "official_fit=true official_decision_function=true"
echo "training_test_access=false training_test_label_access=false"
echo "ray_tune=false searches=false anomaly_ratio=0.8 percentile=99.2"
python3 scripts/benchmarks/run_psm_couta_detection.py "$@" 2>&1 | tee "$OUT/run_detection.log"
