# PUMP：Original PPLAD vs ASCA-AD / V4

## 口径

官方 PPLAD 代码中支持 `PUMP` 数据加载器，但公开仓库不一定包含独立
`PUMP.sh`。因此脚本采用以下策略：

1. 若本地存在 `BaselineModels/PPLAD-main/scripts/PUMP.sh`，自动解析并使用；
2. 若不存在，则使用官方 `main.py` 默认参数，并替换为 PUMP 的实际通道数。

默认回退配置为：

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
dataset/PUMP/
├── PUMP_train.npy
├── PUMP_test.npy
└── PUMP_test_label.npy
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
  run_pump_official_default_vs_v4_best.py \
  benchmark_pump_official_default_vs_v4_best.py \
  run_pump_official_default_vs_v4_best_full.sh

python -m py_compile \
  run_pump_official_default_vs_v4_best.py \
  benchmark_pump_official_default_vs_v4_best.py

bash run_pump_official_default_vs_v4_best_full.sh
```

## 输出

```text
results/PUMP_OFFICIAL_DEFAULT_VS_V4_BEST/
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
