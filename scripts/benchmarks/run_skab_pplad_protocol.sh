#!/usr/bin/env bash
set -e

ROOT="${1:-/mnt/c/Users/DING/Desktop/Experiment/CODE}"
cd "$ROOT"

mkdir -p results/SKAB_scores
mkdir -p results/SKAB_PPLAD_PROTOCOL

echo "Step 1: generate PPLAD scores"
python scripts/benchmarks/generate_pplad_skab_score.py

echo "Step 2: generate ASCA-AD V4 scores"
python scripts/benchmarks/generate_asca_v4_skab_score.py

echo "Step 3: generate LTFAD scores"
python scripts/benchmarks/generate_ltfad_skab_score.py

echo "Step 4: PPLAD protocol evaluation"
python scripts/benchmarks/evaluate_pplad_protocol.py \
 --score-dir results/SKAB_scores \
 --label dataset/SKAB/SKAB_test_label.npy \
 --anomaly-ratio 0.3 \
 --output results/SKAB_PPLAD_PROTOCOL/results.json
