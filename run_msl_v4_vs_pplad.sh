#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
MODE="${1:-all}"

cd "$ROOT"

mkdir -p \
  logs/MSL_V4_COMPARE \
  logs/MSL_COMPARE \
  checkpoints/MSL_V4_ALIGNED \
  checkpoints/MSL_PPLAD_OFFICIAL \
  results/MSL_V4_ALIGNED \
  results/MSL_COMPARISON \
  BaselineModels/PPLAD-main/result

CHANNELS="$(
python - <<'PY'
import numpy as np
x = np.load("dataset/MSL/MSL_train.npy", mmap_mode="r")
print(x.shape[1])
PY
)"

if [[ "$CHANNELS" != "55" ]]; then
  echo "ERROR: MSL 应为 55 通道，实际检测到 $CHANNELS" >&2
  exit 1
fi

run_pplad() {
  echo
  echo "========== Original PPLAD / MSL =========="

  cd "$ROOT/BaselineModels/PPLAD-main"
  mkdir -p result

  if [[ ! -e dataset ]]; then
    ln -s ../../dataset dataset
  fi

  if [[ ! -f dataset/MSL/MSL_train.npy ]]; then
    echo "ERROR: 找不到 dataset/MSL/MSL_train.npy" >&2
    echo "dataset 实际指向：$(readlink -f dataset || true)" >&2
    exit 1
  fi

  python -u main.py \
    --mode train \
    --dataset MSL \
    --data_path MSL \
    --input_c 55 \
    --output_c 55 \
    --win_size 90 \
    --batch_size 256 \
    --num_epochs 3 \
    --lr 0.0001 \
    --anormly_ratio 0.83 \
    --local_size 7 \
    --global_size 30 \
    --d_model 128 \
    --r 0.5 \
    --similar MSE \
    --model_save_path "$ROOT/checkpoints/MSL_PPLAD_OFFICIAL" \
    2>&1 | tee "$ROOT/logs/MSL_COMPARE/pplad_msl_official.log"

  cd "$ROOT"
}

run_v4() {
  echo
  echo "========== V4 aligned / MSL =========="
  echo "V4 与当前 V5-ADAPTED 使用相同候选范围与 Top-k，仅去掉低维投影。"

  rm -f checkpoints/MSL_V4_ALIGNED/MSL_adaptive_anchor_v4_*.pt

  python -u main.py \
    --mode train \
    --dataset MSL \
    --data_path MSL \
    --input_c "$CHANNELS" \
    --output_c "$CHANNELS" \
    --win_size 90 \
    --batch_size 256 \
    --num_epochs 10 \
    --lr 0.001 \
    --anormly_ratio 0.83 \
    --index 137 \
    --seed 42 \
    --local_candidate_lags 1 2 3 \
    --global_candidate_lags 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
    --local_topk 3 \
    --global_topk 12 \
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
    --quantile_method exact \
    --model_save_path checkpoints/MSL_V4_ALIGNED \
    --result_path results/MSL_V4_ALIGNED \
    2>&1 | tee logs/MSL_V4_COMPARE/msl_v4_aligned_seed42.log
}

compare_logs() {
  echo
  echo "========== 指标对比 =========="

  python - <<'PY'
from pathlib import Path
import re
import csv

root = Path("/mnt/c/Users/DING/Desktop/Experiment/CODE")
pplad_log = root / "logs/MSL_COMPARE/pplad_msl_official.log"
v4_log = root / "logs/MSL_V4_COMPARE/msl_v4_aligned_seed42.log"

for p in (pplad_log, v4_log):
    if not p.exists():
        raise SystemExit(f"缺少日志：{p}")

def last_float(patterns, text):
    values = []
    for pattern in patterns:
        values.extend(re.findall(pattern, text, flags=re.I))
    if not values:
        return None
    return float(values[-1])

def parse(path, kind):
    text = path.read_text(encoding="utf-8", errors="ignore")
    data = {}
    # Project/PA fields are printed by both implementations.
    data["PA-Accuracy"] = last_float(
        [r"pa_accuracy\s*:\s*([0-9.]+)", r"PA\s+Accuracy=([0-9.]+)"], text
    )
    data["PA-Precision"] = last_float(
        [r"pa_precision\s*:\s*([0-9.]+)", r"PA\s+Accuracy=[0-9.]+,\s*Precision=([0-9.]+)"], text
    )
    data["PA-Recall"] = last_float(
        [r"pa_recall\s*:\s*([0-9.]+)", r"PA\s+Accuracy=[0-9.]+,\s*Precision=[0-9.]+,\s*Recall=([0-9.]+)"], text
    )
    data["PA-F1"] = last_float(
        [r"pa_f_score\s*:\s*([0-9.]+)", r"PA\s+Accuracy=[0-9.]+,\s*Precision=[0-9.]+,\s*Recall=[0-9.]+,\s*F1=([0-9.]+)"], text
    )
    data["R-AUC-PR"] = last_float([r"R_AUC_PR\s*:\s*([0-9.]+)"], text)
    data["VUS-PR"] = last_float([r"VUS_PR\s*:\s*([0-9.]+)"], text)
    return data

p = parse(pplad_log, "pplad")
v = parse(v4_log, "v4")
metrics = ["PA-Accuracy", "PA-Precision", "PA-Recall", "PA-F1", "R-AUC-PR", "VUS-PR"]

print(f"{'Metric':<16}{'Original':>12}{'V4':>12}{'V4-Original':>16}")
print("-" * 56)
rows = []
for m in metrics:
    pv, vv = p[m], v[m]
    if pv is None or vv is None:
        print(f"{m:<16}{str(pv):>12}{str(vv):>12}{'N/A':>16}")
        diff = None
    else:
        diff = vv - pv
        print(f"{m:<16}{pv:>12.4f}{vv:>12.4f}{diff:>+16.4f}")
    rows.append([m, pv, vv, diff])

out_dir = root / "results/MSL_COMPARISON"
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "MSL_v4_vs_original_metrics.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Original_PPLAD", "V4", "V4_minus_Original"])
    w.writerows(rows)

txt_path = out_dir / "MSL_v4_vs_original_metrics.txt"
with txt_path.open("w", encoding="utf-8") as f:
    f.write(f"{'Metric':<16}{'Original':>12}{'V4':>12}{'V4-Original':>16}\n")
    f.write("-" * 56 + "\n")
    for m, pv, vv, diff in rows:
        if pv is None or vv is None:
            f.write(f"{m:<16}{str(pv):>12}{str(vv):>12}{'N/A':>16}\n")
        else:
            f.write(f"{m:<16}{pv:>12.4f}{vv:>12.4f}{diff:>+16.4f}\n")

print()
print("CSV :", csv_path)
print("TXT :", txt_path)
PY
}

case "$MODE" in
  pplad)
    run_pplad
    ;;
  v4)
    run_v4
    ;;
  compare)
    compare_logs
    ;;
  all)
    run_pplad
    run_v4
    compare_logs
    ;;
  *)
    echo "用法：$0 {pplad|v4|compare|all}" >&2
    exit 2
    ;;
esac
