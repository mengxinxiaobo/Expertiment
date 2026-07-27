#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/results/ASCA_V4_IO_FINAL_EFFICIENCY"
mkdir -p "$OUT/summary"
echo "ASCA-AD V4-IO final paper efficiency benchmark"
echo "training=false labels=false evaluator=false detection=false"
echo "gpu_memory=false process_memory=false warmup=30 latency_repeat=200 full_repeat=20"
python3 "$ROOT/scripts/benchmarks/run_asca_v4_io_final_efficiency.py" "$@" \
  2>&1 | tee -a "$OUT/summary/run.log"
