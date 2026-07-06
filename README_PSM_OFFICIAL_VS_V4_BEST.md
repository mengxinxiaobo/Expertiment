# PSM：Original 官方配置 vs V4 最优结果

## Original PPLAD 官方配置

官方 `scripts/PSM.sh`：

- `anormly_ratio=0.80`
- `win_size=60`
- `batch_size=256`
- `num_epochs=1`
- `local_size=1`
- `global_size=20`
- `d_model=128`
- `r=0.5`
- `lr=1e-4`

## V4 最优结果协议

使用当前已经验证的 PSM V4 结构：

- `win_size=100`
- `batch_size=128`
- `num_epochs=10`
- `lr=1e-3`
- local candidates: `1..8`, Top-k=2
- global candidates: `12,16,20,24,28,32,40,48`, Top-k=4
- 联合搜索 `gap`、`total`、`combined`
- 搜索 `anormly_ratio=0.10..3.00`，步长 `0.01`
- 选择最高 PA-F1

历史 PSM V4 结果使用 `total + anormly_ratio=1.0` 获得过
PA-F1≈0.9853；本程序会重新搜索，避免把阈值固定在历史值。

## 轻量化受控基准

两种模型统一：

- `win_size=60`
- `batch_size=256`
- 相同硬件、预热次数、重复次数和测试集

## 一键运行

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/CODE

chmod +x \
  run_psm_official_vs_v4_best.py \
  benchmark_psm_official_vs_v4_best.py \
  run_psm_official_vs_v4_best_full.sh

python -m py_compile \
  run_psm_official_vs_v4_best.py \
  benchmark_psm_official_vs_v4_best.py

bash run_psm_official_vs_v4_best_full.sh
```

## 主要输出

```text
results/PSM_OFFICIAL_VS_V4_BEST/
├── original_official_metrics.json
├── v4_threshold_sweep.csv
├── v4_best_metrics.json
├── v4_best_run.json
└── BENCHMARK/
    ├── original.json
    ├── v4_best.json
    ├── comparison.json
    ├── comparison.csv
    └── summary.md
```
