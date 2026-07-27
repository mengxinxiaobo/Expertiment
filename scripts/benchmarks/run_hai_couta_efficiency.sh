#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"; OUT="$ROOT/results/HAI_COUTA_RESULTS/Efficiency"; mkdir -p "$OUT"
echo "HAI COUTA inference-only efficiency benchmark"; echo "training=false labels=false evaluator=false threshold=false"
echo "warmup=30 latency_repeat=200 full_test_repeat=20"
python3 scripts/benchmarks/benchmark_hai_couta_efficiency.py "$@" 2>&1 | tee "$OUT/efficiency.log"
