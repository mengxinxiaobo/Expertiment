#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
cd "$ROOT"

RUNNER="$ROOT/run_wadi_official_default_vs_v4_best.py"
BENCH="$ROOT/benchmark_wadi_official_default_vs_v4_best.py"

for FILE in "$RUNNER" "$BENCH"; do
  if [[ ! -f "$FILE" ]]; then
    echo "ERROR: 缺少脚本：$FILE" >&2
    exit 1
  fi
done

for FILE in \
  BaselineModels/PPLAD-main/model/PPLAD.py \
  main.py \
  solver.py
do
  if [[ ! -f "$FILE" ]]; then
    echo "ERROR: 缺少依赖：$FILE" >&2
    exit 1
  fi
done

if [[ ! -d dataset/WADI ]]; then
  echo "ERROR: 缺少 dataset/WADI 目录。" >&2
  exit 1
fi

python - <<'PY'
from pathlib import Path
import numpy as np

data_dir = Path("dataset/WADI")
groups = {
    "train": ["WaDi_train.npy", "WADI_train.npy", "wadi_train.npy"],
    "test": ["WaDi_test.npy", "WADI_test.npy", "wadi_test.npy"],
    "label": ["WaDi_test_label.npy", "WADI_test_label.npy", "wadi_test_label.npy"],
}
paths = {}
for key, names in groups.items():
    for name in names:
        path = data_dir / name
        if path.exists():
            paths[key] = path
            break
    if key not in paths:
        raise FileNotFoundError(f"缺少 WADI {key} 文件：{names}")

train = np.load(paths["train"], mmap_mode="r")
test = np.load(paths["test"], mmap_mode="r")
label = np.load(paths["label"], mmap_mode="r")

if train.ndim != 2 or test.ndim != 2:
    raise ValueError(f"WADI train/test 应为二维：{train.shape}, {test.shape}")
if train.shape[1] != test.shape[1]:
    raise ValueError("WADI train/test 通道数不一致。")
if test.shape[0] != label.reshape(-1).shape[0]:
    raise ValueError("WADI test/label 长度不一致。")

print("WADI data validation passed.")
print("train file :", paths["train"])
print("test file  :", paths["test"])
print("label file :", paths["label"])
print("train shape:", train.shape)
print("test shape :", test.shape)
print("label shape:", label.shape)
print("channels   :", train.shape[1])
PY

mkdir -p \
  result \
  logs/WADI_OFFICIAL_DEFAULT_VS_V4_BEST \
  checkpoints/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/ORIGINAL \
  checkpoints/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/V4 \
  results/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/V4 \
  results/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/BENCHMARK

echo
echo "================================================================"
echo "WADI：Original 官方仓库默认配置 vs V4 最优结果"
echo
echo "Original：官方仓库 main.py 默认值应用于 WADI"
echo "  win=60, batch=128, epochs=3, ratio=0.50"
echo "  local=3, global=20, d_model=128, r=0.5, lr=1e-4"
echo "  注意：官方仓库不存在独立 scripts/WADI.sh。"
echo
echo "V4："
echo "  win=100, batch=128, epochs=10, lr=1e-3"
echo "  local candidates=1..8 / top2"
echo "  global candidates=12,16,20,24,28,32,40,48 / top4"
echo "  score mode=gap,total,combined 联合搜索"
echo "  ratio=0.10..8.00, step=0.01，按最高 PA-F1 选择"
echo
echo "轻量化受控基准：两种模型统一 win=60、batch=128。"
echo "说明：V4 阈值搜索属于 oracle best / best-over-grid 口径。"
echo "================================================================"

echo
echo "========== 1/3 Original PPLAD：官方仓库默认配置 =========="
python -u "$RUNNER" \
  --model original \
  --root "$ROOT" \
  --seed 42 \
  2>&1 | tee logs/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/original.log

echo
echo "========== 2/3 V4：训练及最优阈值/分数搜索 =========="
python -u "$RUNNER" \
  --model v4 \
  --root "$ROOT" \
  --seed 42 \
  --v4-epochs 10 \
  --ratio-min 0.10 \
  --ratio-max 8.00 \
  --ratio-step 0.01 \
  2>&1 | tee logs/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/v4_best.log

echo
echo "========== 3/3 参数、时延、吞吐率和内存基准 =========="
python -u "$BENCH" \
  --model all \
  --root "$ROOT" \
  --output-dir results/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/BENCHMARK \
  --device cuda \
  --batch-size 128 \
  --win-size 60 \
  --warmup 30 \
  --repeats 200 \
  --full-test-repeats 5 \
  --seed 42 \
  --v4-checkpoint \
    checkpoints/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/V4/WADI_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_g12-16-20-24-28-32-40-48_kl2_kg4.pt \
  2>&1 | tee logs/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/benchmark.log

echo
echo "实验完成。核心结果："
cat results/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/BENCHMARK/summary.md
