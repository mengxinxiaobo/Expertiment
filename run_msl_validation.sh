#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
MODE="${1:-all}"

cd "$ROOT"

mkdir -p \
  logs/MSL_V5 \
  logs/MSL_COMPARE \
  checkpoints/MSL_V5 \
  checkpoints/MSL_PPLAD_OFFICIAL \
  results/MSL_V5 \
  results/v5_benchmark \
  BaselineModels/PPLAD-main/result

CHANNELS="$(
python - <<'PY'
import numpy as np
x = np.load("dataset/MSL/MSL_train.npy", mmap_mode="r")
print(x.shape[1])
PY
)"

echo "ROOT=$ROOT"
echo "MSL channels=$CHANNELS"

if [[ "$CHANNELS" != "55" ]]; then
  echo "ERROR: MSL 应为 55 通道，但检测到 $CHANNELS" >&2
  exit 1
fi

run_v5_train() {
  echo
  echo "========== 训练 V5 / MSL =========="
  rm -f checkpoints/MSL_V5/MSL_adaptive_anchor_v5_rd8_*.pt

  python -u main_v5_fused_projection_v2.py \
    --mode train \
    --dataset MSL \
    --data_path MSL \
    --input_c "$CHANNELS" \
    --output_c "$CHANNELS" \
    --relation_dim 8 \
    --win_size 90 \
    --batch_size 256 \
    --num_epochs 10 \
    --lr 0.001 \
    --anormly_ratio 0.83 \
    --index 137 \
    --seed 42 \
    --local_candidate_lags 1 2 3 4 5 6 7 8 \
    --global_candidate_lags 12 16 20 24 28 32 40 48 \
    --local_topk 2 \
    --global_topk 4 \
    --selector_hidden 8 \
    --fitter_hidden 8 \
    --selector_temperature 0.5 \
    --similarity_tau 1.0 \
    --sigma_min 0.03 \
    --sigma_max 1.50 \
    --area_weight 0.1 \
    --selector_balance_weight 0.05 \
    --gap_weight 1.0 \
    --relation_input instance \
    --score_modes total \
    --primary_score total \
    --score_normalization official \
    --threshold_source original \
    --quantile_method exact \
    --model_save_path checkpoints/MSL_V5 \
    --result_path results/MSL_V5 \
    2>&1 | tee logs/MSL_V5/msl_v5_rd8_seed42.log
}

run_v5_benchmark() {
  echo
  echo "========== Benchmark V5 / MSL =========="

  python -u main_v5_fused_projection_v2.py \
    --mode benchmark \
    --dataset MSL \
    --data_path MSL \
    --input_c "$CHANNELS" \
    --output_c "$CHANNELS" \
    --relation_dim 8 \
    --win_size 90 \
    --batch_size 256 \
    --anormly_ratio 0.83 \
    --index 137 \
    --seed 42 \
    --local_candidate_lags 1 2 3 4 5 6 7 8 \
    --global_candidate_lags 12 16 20 24 28 32 40 48 \
    --local_topk 2 \
    --global_topk 4 \
    --selector_hidden 8 \
    --fitter_hidden 8 \
    --selector_temperature 0.5 \
    --similarity_tau 1.0 \
    --sigma_min 0.03 \
    --sigma_max 1.50 \
    --area_weight 0.1 \
    --selector_balance_weight 0.05 \
    --gap_weight 1.0 \
    --relation_input instance \
    --score_modes total \
    --primary_score total \
    --score_normalization official \
    --threshold_source original \
    --quantile_method exact \
    --model_save_path checkpoints/MSL_V5 \
    --result_path results/MSL_V5 \
    --benchmark_warmup 30 \
    --benchmark_repeats 200 \
    --benchmark_threshold \
    --benchmark_output results/v5_benchmark/MSL_v5_rd8_w90_b256.json \
    2>&1 | tee logs/MSL_V5/msl_v5_benchmark.log
}

run_pplad() {
  echo
  echo "========== 训练并测试官方 PPLAD / MSL =========="

  cd "$ROOT/BaselineModels/PPLAD-main"
  mkdir -p result

  if [[ ! -e dataset ]]; then
    ln -s ../../dataset dataset
  fi

  if [[ ! -f dataset/MSL/MSL_train.npy ]]; then
    echo "ERROR: 找不到 BaselineModels/PPLAD-main/dataset/MSL/MSL_train.npy" >&2
    echo "当前 dataset 指向：$(readlink -f dataset || true)" >&2
    exit 1
  fi

  python -u main.py \
    --mode train \
    --dataset MSL \
    --data_path MSL \
    --input_c 55 \
    --output_c 55 \
    --win_size 90 \
    --batch_size 256 \
    --num_epochs 3 \
    --lr 0.0001 \
    --anormly_ratio 0.83 \
    --local_size 7 \
    --global_size 30 \
    --d_model 128 \
    --r 0.5 \
    --similar MSE \
    --model_save_path "$ROOT/checkpoints/MSL_PPLAD_OFFICIAL" \
    2>&1 | tee "$ROOT/logs/MSL_COMPARE/pplad_msl_official.log"

  cd "$ROOT"
}

case "$MODE" in
  v5-train)
    run_v5_train
    ;;
  v5-benchmark)
    run_v5_benchmark
    ;;
  pplad)
    run_pplad
    ;;
  all)
    run_pplad
    run_v5_train
    run_v5_benchmark
    ;;
  *)
    echo "用法：$0 {pplad|v5-train|v5-benchmark|all}" >&2
    exit 2
    ;;
esac

echo
echo "========== 输出文件 =========="
echo "PPLAD 日志 : $ROOT/logs/MSL_COMPARE/pplad_msl_official.log"
echo "PPLAD 内部日志: $ROOT/BaselineModels/PPLAD-main/result/MSL.log"
echo "V5 训练日志: $ROOT/logs/MSL_V5/msl_v5_rd8_seed42.log"
echo "V5 benchmark: $ROOT/results/v5_benchmark/MSL_v5_rd8_w90_b256.json"
