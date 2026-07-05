#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p checkpoints_compare/MSL results_compare/MSL logs_compare/MSL

python main.py \
  --dataset MSL \
  --data_path MSL \
  --input_c 55 \
  --output_c 55 \
  --anormly_ratio 0.75 \
  --mode train \
  --batch_size 128 \
  --win_size 105 \
  --num_epochs 3 \
  --lr 0.001 \
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
  --quantile_method exact
  --model_save_path checkpoints_compare/MSL \
  --result_path results_compare/MSL \
  2>&1 | tee logs_compare/MSL/v4_msl_win105_seed42.log
