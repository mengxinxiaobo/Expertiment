#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/BaselineModels/PatchAD-main"

python benchmark_patchad.py \
  --dataset SMAP \
  --data-root "$ROOT/dataset" \
  --checkpoint "$ROOT/BaselineModels/PatchAD-main/checkpoints_compare/SMAP/SMAP_checkpoint.pth" \
  --device cuda \
  --batch-size 128 \
  --win-size 105 \
  --patch-sizes 3 5 7 \
  --d-model 60 \
  --e-layer 3 \
  --warmup 10 \
  --repeats 50 \
  --output-dir results_compare/benchmark
