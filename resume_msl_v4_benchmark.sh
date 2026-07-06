#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
cd "$ROOT"

BENCH="$ROOT/benchmark_msl_official_vs_v4_best.py"
OUT="results/MSL_OFFICIAL_VS_V4_BEST/BENCHMARK"
CHECKPOINT="checkpoints/MSL_OFFICIAL_VS_V4_BEST/V4/MSL_adaptive_anchor_v4_l1-2-3_g4-5-6-7-8-9-10-11-12-13-14-15-16-17-18_kl2_kg6.pt"

for FILE in \
  "$BENCH" \
  "$OUT/original.json" \
  "results/MSL_OFFICIAL_VS_V4_BEST/original_official_metrics.json" \
  "results/MSL_OFFICIAL_VS_V4_BEST/v4_best_run.json" \
  "$CHECKPOINT"
do
  if [[ ! -f "$FILE" ]]; then
    echo "ERROR: 缺少继续运行所需文件：$FILE" >&2
    exit 1
  fi
done

mkdir -p logs/MSL_OFFICIAL_VS_V4_BEST "$OUT"

echo "========== 继续 3/3：V4 轻量化基准 =========="
python -u "$BENCH" \
  --model v4_best \
  --root "$ROOT" \
  --output-dir "$OUT" \
  --device cuda \
  --batch-size 256 \
  --win-size 90 \
  --warmup 30 \
  --repeats 200 \
  --full-test-repeats 5 \
  --seed 42 \
  --v4-checkpoint "$CHECKPOINT" \
  2>&1 | tee logs/MSL_OFFICIAL_VS_V4_BEST/benchmark_v4.log

echo
echo "========== 汇总 Original + V4 =========="
python -u "$BENCH" \
  --model aggregate \
  --root "$ROOT" \
  --output-dir "$OUT" \
  --device cuda \
  --batch-size 256 \
  --win-size 90 \
  --warmup 30 \
  --repeats 200 \
  --full-test-repeats 5 \
  --seed 42 \
  --v4-checkpoint "$CHECKPOINT" \
  2>&1 | tee logs/MSL_OFFICIAL_VS_V4_BEST/benchmark_aggregate.log

echo
echo "完成："
cat "$OUT/summary.md"
