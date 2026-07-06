#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
cd "$ROOT"

RUNNER="$ROOT/run_skab_official_default_vs_v4_best.py"
BENCH="$ROOT/benchmark_skab_official_default_vs_v4_best.py"

for FILE in "$RUNNER" "$BENCH"; do
  if [[ ! -f "$FILE" ]]; then
    echo "ERROR: 缺少脚本：$FILE" >&2
    exit 1
  fi
done

for FILE in \
  dataset/SKAB/SKAB_train.npy \
  dataset/SKAB/SKAB_test.npy \
  dataset/SKAB/SKAB_test_label.npy \
  BaselineModels/PPLAD-main/model/PPLAD.py \
  main.py \
  solver.py
do
  if [[ ! -f "$FILE" ]]; then
    echo "ERROR: 缺少依赖：$FILE" >&2
    exit 1
  fi
done

python - <<'PY'
import numpy as np

train = np.load("dataset/SKAB/SKAB_train.npy", mmap_mode="r")
test = np.load("dataset/SKAB/SKAB_test.npy", mmap_mode="r")
label = np.load("dataset/SKAB/SKAB_test_label.npy", mmap_mode="r").reshape(-1)

if train.ndim != 2 or test.ndim != 2:
    raise ValueError(f"SKAB train/test 应为二维：{train.shape}, {test.shape}")
if train.shape[1] != test.shape[1]:
    raise ValueError("SKAB train/test 通道数不一致。")
if test.shape[0] != label.shape[0]:
    raise ValueError("SKAB test/label 长度不一致。")

print("SKAB data validation passed.")
print("train shape:", train.shape)
print("test shape :", test.shape)
print("label shape:", label.shape)
print("channels   :", train.shape[1])
print("anomaly ratio: {:.6f}%".format(float(label.mean()) * 100.0))
PY

mkdir -p \
  result \
  logs/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST \
  checkpoints/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/ORIGINAL \
  checkpoints/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/V4 \
  results/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/V4 \
  results/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/BENCHMARK

echo
echo "================================================================"
echo "SKAB：Original PPLAD vs ASCA-AD / V4"
echo
echo "Original：官方 main.py 默认 SKAB 配置"
echo "  win=60, batch=128, epochs=3, ratio=0.50"
echo "  local=3, global=20, d_model=128, r=0.5, lr=1e-4"
echo
echo "ASCA-AD / V4："
echo "  win=100, batch=128, epochs=10, lr=1e-3"
echo "  local candidates=1..8 / top2"
echo "  global candidates=12,16,20,24,28,32,40,48 / top4"
echo "  score mode=gap,total,combined 联合搜索"
echo "  ratio=0.10..3.00, step=0.01，按最高 PA-F1 选择"
echo
echo "轻量化受控基准：两种模型统一 win=60、batch=128。"
echo "说明：V4 阈值搜索属于 oracle best / best-over-grid 口径。"
echo "================================================================"

echo
echo "========== 1/3 Original PPLAD：官方默认配置训练与测试 =========="
python -u "$RUNNER" \
  --model original \
  --root "$ROOT" \
  --seed 42 \
  2>&1 | tee logs/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/original.log

echo
echo "========== 2/3 ASCA-AD / V4：训练与最优阈值搜索 =========="
python -u "$RUNNER" \
  --model v4 \
  --root "$ROOT" \
  --seed 42 \
  --v4-epochs 10 \
  --ratio-min 0.10 \
  --ratio-max 3.00 \
  --ratio-step 0.01 \
  2>&1 | tee logs/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/v4_best.log

echo
echo "========== 3/3 参数、时延、吞吐率和内存基准 =========="
python -u "$BENCH" \
  --model all \
  --root "$ROOT" \
  --output-dir results/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/BENCHMARK \
  --device cuda \
  --batch-size 128 \
  --win-size 60 \
  --warmup 30 \
  --repeats 200 \
  --full-test-repeats 5 \
  --seed 42 \
  --v4-checkpoint \
    checkpoints/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/V4/SKAB_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_g12-16-20-24-28-32-40-48_kl2_kg4.pt \
  2>&1 | tee logs/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/benchmark.log

echo
echo "实验完成。核心结果："
cat results/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/BENCHMARK/summary.md
