#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
cd "$ROOT"

mkdir -p logs results/ALL_COMBINED_FIXED

python -m py_compile run_all_v4_combined_fixed.py

python -u run_all_v4_combined_fixed.py \
  --root "$ROOT" \
  --datasets all \
  --output-dir results/ALL_COMBINED_FIXED \
  2>&1 | tee logs/all_v4_combined_fixed.log

echo
echo "汇总结果："
cat results/ALL_COMBINED_FIXED/summary.md
