#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
cd "$ROOT"

RUNNER="$ROOT/run_msl_official_vs_v4_best.py"
BENCH="$ROOT/benchmark_msl_official_vs_v4_best.py"

for FILE in "$RUNNER" "$BENCH"; do
  if [[ ! -f "$FILE" ]]; then
    echo "ERROR: 缺少脚本：$FILE" >&2
    exit 1
  fi
done

for FILE in \
  dataset/MSL/MSL_train.npy \
  dataset/MSL/MSL_test.npy \
  dataset/MSL/MSL_test_label.npy \
  BaselineModels/PPLAD-main/model/PPLAD.py \
  main.py \
  solver.py
do
  if [[ ! -f "$FILE" ]]; then
    echo "ERROR: 缺少依赖：$FILE" >&2
    exit 1
  fi
done

CHANNELS="$(
python - <<'PY'
import numpy as np
x = np.load("dataset/MSL/MSL_train.npy", mmap_mode="r")
print(x.shape[1])
PY
)"
if [[ "$CHANNELS" != "55" ]]; then
  echo "ERROR: MSL 应为 55 通道，实际为 $CHANNELS" >&2
  exit 1
fi

mkdir -p \
  logs/MSL_OFFICIAL_VS_V4_BEST \
  checkpoints/MSL_OFFICIAL_VS_V4_BEST/ORIGINAL \
  checkpoints/MSL_OFFICIAL_VS_V4_BEST/V4 \
  results/MSL_OFFICIAL_VS_V4_BEST/V4 \
  results/MSL_OFFICIAL_VS_V4_BEST/BENCHMARK

echo
echo "================================================================"
echo "MSL：Original 官方配置 vs V4 各自最优结果"
echo "Original：官方 win=90, batch=256, epochs=3, ratio=0.83"
echo "          local=7, global=30, d_model=128, lr=1e-4"
echo "V4：      win=90, batch=256, epochs=10"
echo "          local candidates=1,2,3 / top2"
echo "          global candidates=4..18 / top6"
echo "          score mode=gap,total,combined 联合搜索"
echo "          ratio=0.10..3.00, step=0.01，按最高 PA-F1 选择"
echo "说明：V4 阈值搜索为 oracle best / best-over-grid 口径。"
echo "================================================================"

echo
echo "========== 1/3 Original PPLAD：官方配置训练与测试 =========="
python -u "$RUNNER" \
  --model original \
  --root "$ROOT" \
  --seed 42 \
  2>&1 | tee logs/MSL_OFFICIAL_VS_V4_BEST/original.log

echo
echo "========== 2/3 V4：训练一次并搜索最优阈值/分数 =========="
python -u "$RUNNER" \
  --model v4 \
  --root "$ROOT" \
  --seed 42 \
  --v4-epochs 10 \
  --ratio-min 0.10 \
  --ratio-max 3.00 \
  --ratio-step 0.01 \
  2>&1 | tee logs/MSL_OFFICIAL_VS_V4_BEST/v4_best.log

echo
echo "========== 3/3 参数、时延、吞吐率和内存基准 =========="
python -u "$BENCH" \
  --model all \
  --root "$ROOT" \
  --output-dir results/MSL_OFFICIAL_VS_V4_BEST/BENCHMARK \
  --device cuda \
  --batch-size 256 \
  --win-size 90 \
  --warmup 30 \
  --repeats 200 \
  --full-test-repeats 5 \
  --seed 42 \
  --v4-checkpoint \
    checkpoints/MSL_OFFICIAL_VS_V4_BEST/V4/MSL_adaptive_anchor_v4_l1-2-3_g4-5-6-7-8-9-10-11-12-13-14-15-16-17-18_kl2_kg6.pt \
  2>&1 | tee logs/MSL_OFFICIAL_VS_V4_BEST/benchmark.log

echo
echo "实验完成。核心结果："
cat results/MSL_OFFICIAL_VS_V4_BEST/BENCHMARK/summary.md
