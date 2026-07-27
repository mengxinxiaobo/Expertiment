#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONUNBUFFERED=1
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_DIR="$ROOT_DIR/results/HAI_PAPER_RESULTS"
LOG_PATH="$OUTPUT_DIR/run_detection.log"
mkdir -p "$OUTPUT_DIR"

ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "HAI ASCA-AD V4 vs PPLAD vs LTFAD fixed-protocol detection"
echo "seed=42"
echo "standard_scaler=train_fit_only"
echo "input_dtype=float32"
echo "anomaly_ratio=0.98"
echo "percentile=99.02"
echo "threshold=independent_per_model"
echo "score_search=disabled"
echo "ratio_search=disabled"
echo "threshold_search=disabled"
echo "test_label_parameter_selection=disabled"
echo "worker_test_label_access=forbidden"
echo "ASCA_training=disabled"
echo "PPLAD=official_HAI_config"
echo "LTFAD=pre_registered_fixed_config_with_external_adapter"
echo "overwrite=$OVERWRITE"

"$PYTHON_BIN" scripts/benchmarks/build_hai_paper_results.py "${ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo "Completed. Log: $LOG_PATH"
