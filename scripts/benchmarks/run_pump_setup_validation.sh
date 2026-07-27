#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"
echo "PUMP three-model compatibility validation only"
echo "training=disabled"
echo "forward=disabled"
echo "metrics=disabled"
echo "test_label_access=forbidden"
"${PYTHON_BIN}" scripts/benchmarks/validate_pump_three_models_setup.py
