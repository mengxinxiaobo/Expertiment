#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_psm_simad_detection.sh"
bash "${SCRIPT_DIR}/run_psm_simad_efficiency.sh"
