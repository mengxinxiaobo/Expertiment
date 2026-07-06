# WADI：Original 官方仓库默认配置 vs V4 最优结果

## 重要口径

官方 PPLAD 仓库没有独立的 `scripts/WADI.sh`。因此，本实验不能写成
“Original 使用官方 WADI 专用配置”。

Original 使用官方仓库 `main.py` 中的默认参数，并把数据集名称与输入/
输出通道数替换为 WADI：

- `win_size=60`
- `local_size=[3]`
- `global_size=[20]`
- `lr=1e-4`
- `d_model=128`
- `num_epochs=3`
- `batch_size=128`
- `anormly_ratio=0.5`
- `r=0.5`
- 通道数由 `dataset/WADI` 中的训练数据自动读取

## WADI 文件名兼容

官方数据加载器使用：

```text
WaDi_train.npy
WaDi_test.npy
WaDi_test_label.npy
```

本实验脚本也兼容：

```text
WADI_train.npy / WADI_test.npy / WADI_test_label.npy
wadi_train.npy / wadi_test.npy / wadi_test_label.npy
```

无需重命名或复制大型数据文件。

## V4

- `win_size=100`
- `batch_size=128`
- `num_epochs=10`
- `lr=1e-3`
- local candidates `1..8`, Top-k `2`
- global candidates `12,16,20,24,28,32,40,48`, Top-k `4`
- 搜索 `gap,total,combined`
- 搜索 `anormly_ratio=0.10..8.00`，步长 `0.01`
- 按最高 PA-F1 选择结果

V4 的选择属于 oracle best / best-over-grid，即使用测试标签从预定义
网格中选择最高 PA-F1，论文中必须明确披露。

## 轻量化基准

为隔离模型结构开销，两种模型统一使用：

- `win_size=60`
- `batch_size=128`
- 相同测试集、设备、预热次数和重复次数

## 运行

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/CODE

chmod +x \
  run_wadi_official_default_vs_v4_best.py \
  benchmark_wadi_official_default_vs_v4_best.py \
  run_wadi_official_default_vs_v4_best_full.sh

python -m py_compile \
  run_wadi_official_default_vs_v4_best.py \
  benchmark_wadi_official_default_vs_v4_best.py

bash run_wadi_official_default_vs_v4_best_full.sh
```

## 输出

```text
results/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/
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
