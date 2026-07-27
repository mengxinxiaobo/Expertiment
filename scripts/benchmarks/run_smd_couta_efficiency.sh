#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
OUT="$ROOT/results/SMD_COUTA_RESULTS/Efficiency"; mkdir -p "$OUT"
echo "SMD COUTA streaming inference-only efficiency benchmark"
echo "training=false labels=false evaluator=false threshold=false"
echo "warmup=30 latency_repeat=200 full_test_repeat=20 batch=64 all_windows_on_gpu=false"
python3 scripts/benchmarks/benchmark_smd_couta_efficiency.py "$@" 2>&1 | tee "$OUT/efficiency.log"
