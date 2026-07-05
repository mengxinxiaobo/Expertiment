#!/usr/bin/env bash
set -euo pipefail

CHANNELS=$(python -c "import numpy as np; print(np.load('dataset/PUMP/PUMP_train.npy', mmap_mode='r').shape[1])")

mkdir -p checkpoints/PUMP results/PUMP logs/PUMP

python main.py \
  --mode train \
  --dataset PUMP \
  --data_path PUMP \
  --input_c "$CHANNELS" \
  --output_c "$CHANNELS" \
  --win_size 100 \
  --batch_size 128 \
  --num_epochs 10 \
  --lr 0.001 \
  --anormly_ratio 0.5 \
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
  --model_save_path checkpoints/PUMP \
  --result_path results/PUMP \
  2>&1 | tee logs/PUMP/pump_v4_seed42.log
