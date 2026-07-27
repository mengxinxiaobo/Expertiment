#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_DIR="$ROOT_DIR/results/ASCA_RETRAIN_VERIFY"
LOG_PATH="$OUTPUT_DIR/run.log"
mkdir -p "$OUTPUT_DIR"

ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "ASCA-AD V4 retrain reproducibility verification"
echo "datasets=SKAB,MSL,PSM"
echo "baselines=disabled"
echo "paper_results=unchanged"
echo "seed=42"
echo "epochs=10"
echo "batch_size=128"
echo "learning_rate=0.001"
echo "optimizer=Adam"
echo "lr_schedule=initial_lr*0.5^(epoch-1)"
echo "window=100"
echo "score_mode=total"
echo "training_test_label_access=blocked"
echo "existing_checkpoint_metrics=recomputed"
echo "anomaly_ratio: SKAB=0.5, MSL=0.8, PSM=0.8"
echo "overwrite_verification_checkpoint=$OVERWRITE"

"$PYTHON_BIN" scripts/benchmarks/retrain_asca_v4_verify.py "${ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo "Completed. Log: $LOG_PATH"
