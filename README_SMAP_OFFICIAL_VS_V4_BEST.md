# SMAP：Original 官方配置 vs V4 最优结果

## 实验口径

### Original PPLAD

严格使用官方 `scripts/SMAP.sh`：

- `anormly_ratio=0.80`
- `win_size=90`
- `batch_size=128`
- `num_epochs=3`
- `local_size=3`
- `global_size=5`
- `d_model=128`
- `r=0.9`
- `lr=1e-4`

### V4

使用当前已经验证的 SMAP V4 结构：

- `win_size=100`
- `batch_size=128`
- `num_epochs=10`
- `lr=1e-3`
- local candidates: `1..8`, Top-k=2
- global candidates: `12,16,20,24,28,32,40,48`, Top-k=4
- 搜索 `gap`、`total`、`combined`
- 搜索 `anormly_ratio=0.10..5.00`，步长 `0.01`
- 选择最高 PA-F1

### 轻量化基准

为了隔离模型结构开销，两种模型统一使用：

- `win_size=90`
- `batch_size=128`
- 相同设备、预热次数、重复次数与测试集

## 运行

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/CODE

chmod +x \
  run_smap_official_vs_v4_best.py \
  benchmark_smap_official_vs_v4_best.py \
  run_smap_official_vs_v4_best_full.sh

bash run_smap_official_vs_v4_best_full.sh
```

## 主要输出

```text
results/SMAP_OFFICIAL_VS_V4_BEST/
├── original_official_metrics.json
├── v4_threshold_sweep.csv
├── v4_best_metrics.json
├── v4_best_run.json
├── v4_best_score.txt
├── v4_best_pred_raw.txt
├── v4_best_pred_pa.txt
├── v4_label.txt
└── BENCHMARK/
    ├── original.json
    ├── v4_best.json
    ├── comparison.json
    ├── comparison.csv
    └── summary.md
```
