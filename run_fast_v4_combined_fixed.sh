#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
cd "$ROOT"

mkdir -p logs results/FAST_COMBINED_FIXED

python -m py_compile run_all_v4_combined_fixed.py

# 先跑较快数据集：SKAB、PUMP、PSM
python -u run_all_v4_combined_fixed.py \
  --root "$ROOT" \
  --datasets SKAB,PUMP,PSM \
  --output-dir results/FAST_COMBINED_FIXED \
  2>&1 | tee logs/fast_v4_combined_fixed.log

echo
echo "快速数据集汇总结果："
cat results/FAST_COMBINED_FIXED/summary.md
