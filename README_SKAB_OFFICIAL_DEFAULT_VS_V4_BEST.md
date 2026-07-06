# SKAB：Original PPLAD vs ASCA-AD / V4

## 口径

SKAB 是官方 PPLAD 仓库 `main.py` 中的默认数据集。Original 使用官方
默认参数，并把 `input_c/output_c` 设置为本地 `SKAB_train.npy` 的实际通道数。

Original 默认配置：

- `win_size=60`
- `batch_size=128`
- `num_epochs=3`
- `anormly_ratio=0.50`
- `local_size=[3]`
- `global_size=[20]`
- `d_model=128`
- `r=0.5`
- `lr=1e-4`

## ASCA-AD / V4

- `win_size=100`
- `batch_size=128`
- `num_epochs=10`
- `lr=1e-3`
- local candidates: `1..8`, Top-k=2
- global candidates: `12,16,20,24,28,32,40,48`, Top-k=4
- 搜索 `gap,total,combined`
- 搜索 `anormly_ratio=0.10..3.00`，步长 `0.01`
- 按最高 PA-F1 选择结果

V4 阈值选择属于 oracle best / best-over-grid，论文中必须说明。

## 数据文件

```text
dataset/SKAB/
├── SKAB_train.npy
├── SKAB_test.npy
└── SKAB_test_label.npy
```

## 轻量化基准

两种模型统一：

- `win_size=60`
- `batch_size=128`
- 相同测试集、设备、预热次数和重复次数

## 运行

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/CODE

chmod +x \
  run_skab_official_default_vs_v4_best.py \
  benchmark_skab_official_default_vs_v4_best.py \
  run_skab_official_default_vs_v4_best_full.sh

python -m py_compile \
  run_skab_official_default_vs_v4_best.py \
  benchmark_skab_official_default_vs_v4_best.py

bash run_skab_official_default_vs_v4_best_full.sh
```

## 输出

```text
results/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/
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
