#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHAD="$ROOT/BaselineModels/PatchAD-main"
mkdir -p "$PATCHAD/checkpoints_compare" "$PATCHAD/results_compare"
cd "$PATCHAD"

python main_ad.py \
  --mode train \
  --data_name SMAP \
  --data_path "$ROOT/dataset" \
  --device cuda \
  --win_size 105 \
  --stride 1 \
  --batch_size 128 \
  --epochs 3 \
  --anormly_ratio 2.0 \
  --learning_rate 0.0001 \
  --patch_sizes '[3,5,7]' \
  --patch_mx 0.1 \
  --d_model 60 \
  --e_layer 3 \
  --seed 42 \
  --cont_beta 0.0 \
  --save_model 1 \
  --full_res 1 \
  --model_save_path "$PATCHAD/checkpoints_compare/" \
  --res_pth "$PATCHAD/results_compare/"
