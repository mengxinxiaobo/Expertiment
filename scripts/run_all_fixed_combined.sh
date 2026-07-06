#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs results/ALL_COMBINED_FIXED

python -m py_compile scripts/run_all_fixed_combined.py

python -u scripts/run_all_fixed_combined.py \
  --root "$ROOT" \
  --datasets all \
  --output-dir results/ALL_COMBINED_FIXED \
  2>&1 | tee logs/all_v4_combined_fixed.log

echo
echo "Summary:"
cat results/ALL_COMBINED_FIXED/summary.md
