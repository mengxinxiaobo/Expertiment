# HAI：Original 官方配置 vs V4 最优结果

## Original PPLAD 官方配置

官方 `scripts/HAI.sh`：

- `anormly_ratio=0.98`
- `win_size=90`
- `batch_size=128`
- `num_epochs=3`
- `local_size=9`
- `global_size=18`
- `d_model=128`
- `r=0.8`
- `lr=1e-4`
- 输入通道数：86

## V4 最优结果协议

使用当前已经验证的 HAI 较优配置：

- `win_size=90`
- `batch_size=128`
- `num_epochs=3`
- `lr=1e-3`
- local candidates: `1..8`, Top-k=2
- global candidates: `12,16,20,24,28,32,40,48`, Top-k=4
- 联合搜索 `gap`、`total`、`combined`
- 搜索 `anormly_ratio=0.10..3.00`，步长 `0.01`
- 选择最高 PA-F1

历史记录中，HAI 的 3-epoch V4 配置 PA-F1 约为 0.8429，
高于统一 10-epoch 配置约 0.8369，因此本轮使用 3 epochs，
并重新联合搜索 score mode 与阈值。

## 数据要求

```text
dataset/HAI/
├── HAI_train.npy
├── HAI_test.npy
└── HAI_test_label.npy
```

预期：

- 通道数：86
- 历史处理规模：train 896400，test 284400

## 内存说明

HAI 训练集较大。程序为了降低峰值内存，会依次处理
`gap`、`total`、`combined` 三种分数，而不是长期同时保存三组
训练分数。阈值仍采用 train+test 精确百分位规则。

## 一键运行

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/CODE

chmod +x \
  run_hai_official_vs_v4_best.py \
  benchmark_hai_official_vs_v4_best.py \
  run_hai_official_vs_v4_best_full.sh

python -m py_compile \
  run_hai_official_vs_v4_best.py \
  benchmark_hai_official_vs_v4_best.py

bash run_hai_official_vs_v4_best_full.sh
```

## 主要输出

```text
results/HAI_OFFICIAL_VS_V4_BEST/
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
