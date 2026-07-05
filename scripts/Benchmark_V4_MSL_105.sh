#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/benchmark_model.py \
  --model v4 \
  --dataset MSL \
  --device cuda \
  --batch-size 128 \
  --win-size 105 \
  --warmup 10 \
  --repeats 50 \
  --checkpoint "$ROOT/checkpoints_compare/MSL/MSL_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_g12-16-20-24-28-32-40-48_kl2_kg4.pt" \
  --output-dir results_compare/benchmark
