#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
cd "$ROOT"

mkdir -p logs results/SKAB_COMBINED_FIXED

python -m py_compile run_skab_v4_combined_fixed.py

python -u run_skab_v4_combined_fixed.py \
  --root "$ROOT" \
  --anormly-ratio 0.50 \
  --output-dir results/SKAB_COMBINED_FIXED \
  2>&1 | tee logs/skab_v4_combined_fixed.log

echo
echo "结果："
cat results/SKAB_COMBINED_FIXED/summary.md
