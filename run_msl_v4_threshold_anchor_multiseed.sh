#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/DING/Desktop/Experiment/CODE"
MODE="${1:-threshold}"

cd "$ROOT"

mkdir -p \
  logs/MSL_V4_THRESHOLD \
  logs/MSL_V4_ANCHORS \
  logs/MSL_V4_MULTI_SEED \
  checkpoints/MSL_V4_ANCHORS \
  checkpoints/MSL_V4_MULTI_SEED \
  results/MSL_V4_THRESHOLD \
  results/MSL_V4_ANCHORS \
  results/MSL_V4_MULTI_SEED \
  results/MSL_V4_ANALYSIS

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

COMMON_ARGS=(
  --dataset MSL
  --data_path MSL
  --input_c "$CHANNELS"
  --output_c "$CHANNELS"
  --win_size 90
  --batch_size 256
  --lr 0.001
  --index 137
  --selector_hidden 8
  --fitter_hidden 8
  --selector_temperature 0.5
  --similarity_tau 1.0
  --sigma_min 0.03
  --sigma_max 1.50
  --area_weight 0.1
  --selector_balance_weight 0.05
  --gap_weight 1.0
  --relation_input instance
  --score_normalization official
  --threshold_source original
  --quantile_method exact
)

run_threshold_sweep() {
  echo
  echo "========== V4 阈值敏感性测试（不重新训练） =========="

  CKPT="checkpoints/MSL_V4_ALIGNED/MSL_adaptive_anchor_v4_l1-2-3_g4-5-6-7-8-9-10-11-12-13-14-15-16-17-18_kl3_kg12.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: 找不到现有 V4 checkpoint：$CKPT" >&2
    exit 1
  fi

  # anormly_ratio 越大，百分位阈值越低，预测异常点越多，通常 Recall 上升、Precision 下降。
  RATIOS=(0.40 0.55 0.70 0.83 1.00 1.20 1.50 2.00)

  for RATIO in "${RATIOS[@]}"; do
    TAG="${RATIO/./p}"
    echo
    echo "----- anormly_ratio=$RATIO -----"

    python -u main.py \
      --mode test \
      "${COMMON_ARGS[@]}" \
      --anormly_ratio "$RATIO" \
      --seed 42 \
      --local_candidate_lags 1 2 3 \
      --global_candidate_lags 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
      --local_topk 3 \
      --global_topk 12 \
      --score_modes gap total combined \
      --primary_score combined \
      --model_save_path checkpoints/MSL_V4_ALIGNED \
      --result_path "results/MSL_V4_THRESHOLD/ratio_${TAG}" \
      2>&1 | tee "logs/MSL_V4_THRESHOLD/ratio_${TAG}.log"
  done

  parse_threshold_logs
}

run_anchor_sweep() {
  echo
  echo "========== V4 锚点与稀疏度对比 =========="
  echo "所有配置使用官方 MSL 时间范围作为基础，并额外测试一个多尺度范围。"

  # 格式：tag|local_lags|global_lags|local_topk|global_topk
  CONFIGS=(
    "official_dense|1 2 3|4 5 6 7 8 9 10 11 12 13 14 15 16 17 18|3|15"
    "official_k12|1 2 3|4 5 6 7 8 9 10 11 12 13 14 15 16 17 18|3|12"
    "official_k8|1 2 3|4 5 6 7 8 9 10 11 12 13 14 15 16 17 18|2|8"
    "official_k6|1 2 3|4 5 6 7 8 9 10 11 12 13 14 15 16 17 18|2|6"
    "multiscale_wide|1 2 3 4 5|6 8 10 12 14 16 18 22 27 32 40|3|6"
  )

  for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r TAG LOCAL_LAGS GLOBAL_LAGS LOCAL_TOPK GLOBAL_TOPK <<< "$CFG"

    CKPT_DIR="checkpoints/MSL_V4_ANCHORS/${TAG}"
    RESULT_DIR="results/MSL_V4_ANCHORS/${TAG}"
    LOG="logs/MSL_V4_ANCHORS/${TAG}.log"

    mkdir -p "$CKPT_DIR" "$RESULT_DIR"

    echo
    echo "----- $TAG -----"
    echo "local=[$LOCAL_LAGS], topk=$LOCAL_TOPK"
    echo "global=[$GLOBAL_LAGS], topk=$GLOBAL_TOPK"

    # shellcheck disable=SC2206
    LOCAL_ARRAY=($LOCAL_LAGS)
    # shellcheck disable=SC2206
    GLOBAL_ARRAY=($GLOBAL_LAGS)

    python -u main.py \
      --mode train \
      "${COMMON_ARGS[@]}" \
      --num_epochs 10 \
      --anormly_ratio 0.83 \
      --seed 42 \
      --local_candidate_lags "${LOCAL_ARRAY[@]}" \
      --global_candidate_lags "${GLOBAL_ARRAY[@]}" \
      --local_topk "$LOCAL_TOPK" \
      --global_topk "$GLOBAL_TOPK" \
      --score_modes gap total combined \
      --primary_score combined \
      --model_save_path "$CKPT_DIR" \
      --result_path "$RESULT_DIR" \
      2>&1 | tee "$LOG"
  done

  parse_anchor_logs
}

run_multi_seed() {
  echo
  echo "========== V4 多随机种子重复训练 =========="
  echo "默认使用 official_k12；每个 seed 使用独立 checkpoint 目录，避免互相覆盖。"

  SEEDS=(42 43 44 45 46)

  for SEED in "${SEEDS[@]}"; do
    CKPT_DIR="checkpoints/MSL_V4_MULTI_SEED/seed_${SEED}"
    RESULT_DIR="results/MSL_V4_MULTI_SEED/seed_${SEED}"
    LOG="logs/MSL_V4_MULTI_SEED/seed_${SEED}.log"

    mkdir -p "$CKPT_DIR" "$RESULT_DIR"

    echo
    echo "----- seed=$SEED -----"

    python -u main.py \
      --mode train \
      "${COMMON_ARGS[@]}" \
      --num_epochs 10 \
      --anormly_ratio 0.83 \
      --seed "$SEED" \
      --local_candidate_lags 1 2 3 \
      --global_candidate_lags 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
      --local_topk 3 \
      --global_topk 12 \
      --score_modes gap total combined \
      --primary_score combined \
      --model_save_path "$CKPT_DIR" \
      --result_path "$RESULT_DIR" \
      2>&1 | tee "$LOG"
  done

  parse_seed_logs
}

parse_threshold_logs() {
  python - <<'PY'
from pathlib import Path
import csv, re

root = Path("/mnt/c/Users/DING/Desktop/Experiment/CODE")
log_dir = root / "logs/MSL_V4_THRESHOLD"
out = root / "results/MSL_V4_ANALYSIS/MSL_v4_threshold_sweep.csv"

def blocks(text):
    pattern = re.compile(
        r"Score mode\s*:\s*(\w+)(.*?)(?=\n={10,}\nScore mode|\Z)",
        re.S,
    )
    return pattern.findall(text)

rows = []
for path in sorted(log_dir.glob("ratio_*.log")):
    text = path.read_text(encoding="utf-8", errors="ignore")
    ratio_m = re.search(r"anormly_ratio\s*:\s*([0-9.]+)", text)
    ratio = float(ratio_m.group(1)) if ratio_m else None

    for mode, body in blocks(text):
        def get(pattern):
            m = re.search(pattern, body, re.I)
            return float(m.group(1)) if m else None

        rows.append({
            "ratio": ratio,
            "score_mode": mode,
            "threshold": get(r"Threshold\s*:\s*([0-9.eE+-]+)"),
            "pa_precision": get(r"pa_precision\s*:\s*([0-9.]+)"),
            "pa_recall": get(r"pa_recall\s*:\s*([0-9.]+)"),
            "pa_f1": get(r"pa_f_score\s*:\s*([0-9.]+)"),
            "r_auc_pr": get(r"R_AUC_PR\s*:\s*([0-9.]+)"),
            "vus_pr": get(r"VUS_PR\s*:\s*([0-9.]+)"),
            "raw_f1": get(r"RAW\s+Accuracy=[0-9.]+,\s*Precision=[0-9.]+,\s*Recall=[0-9.]+,\s*F1=([0-9.]+)"),
        })

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\n阈值敏感性结果：", out)
print(f"{'ratio':>7} {'mode':>9} {'threshold':>12} {'P':>8} {'R':>8} {'F1':>8} {'R-AUC-PR':>10} {'VUS-PR':>9}")
print("-" * 82)
for r in sorted(rows, key=lambda x: (x["ratio"], x["score_mode"])):
    print(
        f"{r['ratio']:7.2f} {r['score_mode']:>9} "
        f"{(r['threshold'] or 0):12.6f} {(r['pa_precision'] or 0):8.4f} "
        f"{(r['pa_recall'] or 0):8.4f} {(r['pa_f1'] or 0):8.4f} "
        f"{(r['r_auc_pr'] or 0):10.4f} {(r['vus_pr'] or 0):9.4f}"
    )
PY
}

parse_anchor_logs() {
  python - <<'PY'
from pathlib import Path
import csv, re

root = Path("/mnt/c/Users/DING/Desktop/Experiment/CODE")
log_dir = root / "logs/MSL_V4_ANCHORS"
out = root / "results/MSL_V4_ANALYSIS/MSL_v4_anchor_sweep.csv"

rows = []
for path in sorted(log_dir.glob("*.log")):
    text = path.read_text(encoding="utf-8", errors="ignore")
    params = re.search(r"Trainable parameters\s*:\s*([\d,]+)", text)
    for mode, body in re.findall(
        r"Score mode\s*:\s*(\w+)(.*?)(?=\n={10,}\nScore mode|\Z)",
        text,
        flags=re.S,
    ):
        def get(pattern):
            m = re.search(pattern, body, re.I)
            return float(m.group(1)) if m else None
        rows.append({
            "config": path.stem,
            "score_mode": mode,
            "params": int(params.group(1).replace(",", "")) if params else None,
            "pa_precision": get(r"pa_precision\s*:\s*([0-9.]+)"),
            "pa_recall": get(r"pa_recall\s*:\s*([0-9.]+)"),
            "pa_f1": get(r"pa_f_score\s*:\s*([0-9.]+)"),
            "r_auc_pr": get(r"R_AUC_PR\s*:\s*([0-9.]+)"),
            "vus_pr": get(r"VUS_PR\s*:\s*([0-9.]+)"),
        })

with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\n锚点配置结果：", out)
print(f"{'config':<20} {'mode':>9} {'P':>8} {'R':>8} {'F1':>8} {'R-AUC-PR':>10} {'VUS-PR':>9}")
print("-" * 78)
for r in rows:
    print(
        f"{r['config']:<20} {r['score_mode']:>9} "
        f"{(r['pa_precision'] or 0):8.4f} {(r['pa_recall'] or 0):8.4f} "
        f"{(r['pa_f1'] or 0):8.4f} {(r['r_auc_pr'] or 0):10.4f} "
        f"{(r['vus_pr'] or 0):9.4f}"
    )
PY
}

parse_seed_logs() {
  python - <<'PY'
from pathlib import Path
import csv, re, statistics

root = Path("/mnt/c/Users/DING/Desktop/Experiment/CODE")
log_dir = root / "logs/MSL_V4_MULTI_SEED"
out = root / "results/MSL_V4_ANALYSIS/MSL_v4_multi_seed.csv"

rows = []
for path in sorted(log_dir.glob("seed_*.log")):
    seed = int(path.stem.split("_")[-1])
    text = path.read_text(encoding="utf-8", errors="ignore")
    for mode, body in re.findall(
        r"Score mode\s*:\s*(\w+)(.*?)(?=\n={10,}\nScore mode|\Z)",
        text,
        flags=re.S,
    ):
        def get(pattern):
            m = re.search(pattern, body, re.I)
            return float(m.group(1)) if m else None
        rows.append({
            "seed": seed,
            "score_mode": mode,
            "pa_precision": get(r"pa_precision\s*:\s*([0-9.]+)"),
            "pa_recall": get(r"pa_recall\s*:\s*([0-9.]+)"),
            "pa_f1": get(r"pa_f_score\s*:\s*([0-9.]+)"),
            "r_auc_pr": get(r"R_AUC_PR\s*:\s*([0-9.]+)"),
            "vus_pr": get(r"VUS_PR\s*:\s*([0-9.]+)"),
        })

with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\n多随机种子结果：", out)
for mode in sorted(set(r["score_mode"] for r in rows)):
    subset = [r for r in rows if r["score_mode"] == mode]
    print(f"\n[{mode}]")
    for metric in ("pa_precision", "pa_recall", "pa_f1", "r_auc_pr", "vus_pr"):
        vals = [r[metric] for r in subset if r[metric] is not None]
        if vals:
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{metric:14} = {statistics.mean(vals):.4f} ± {std:.4f}")
PY
}

case "$MODE" in
  threshold)
    run_threshold_sweep
    ;;
  anchors)
    run_anchor_sweep
    ;;
  multiseed)
    run_multi_seed
    ;;
  parse-threshold)
    parse_threshold_logs
    ;;
  parse-anchors)
    parse_anchor_logs
    ;;
  parse-seeds)
    parse_seed_logs
    ;;
  all)
    run_threshold_sweep
    run_anchor_sweep
    run_multi_seed
    ;;
  *)
    echo "用法：$0 {threshold|anchors|multiseed|parse-threshold|parse-anchors|parse-seeds|all}" >&2
    exit 2
    ;;
esac
