#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-comparison/config.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

DATASETS=("$@")
DETECT_ARGS=(--config "$CONFIG")
BENCH_ARGS=(--config "$CONFIG")

if [[ ${#DATASETS[@]} -gt 0 ]]; then
  DETECT_ARGS+=(--datasets "${DATASETS[@]}")
  BENCH_ARGS+=(--datasets "${DATASETS[@]}")
fi

python comparison/run_detection_compare.py "${DETECT_ARGS[@]}"

RUN_ROOT="$(cat comparison_runs/LATEST)"

python comparison/run_benchmark_compare.py \
  "${BENCH_ARGS[@]}" \
  --run-root "$RUN_ROOT"

python comparison/make_report.py \
  --config "$CONFIG" \
  --run-root "$RUN_ROOT"

echo
echo "全部实验完成：$RUN_ROOT"
echo "检测结果：$RUN_ROOT/detection_comparison.csv"
echo "轻量化结果：$RUN_ROOT/lightweight_comparison.csv"
echo "Markdown报告：$RUN_ROOT/summary.md"
