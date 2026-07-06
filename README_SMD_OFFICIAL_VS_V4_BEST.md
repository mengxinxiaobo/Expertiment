# SMD：Original 官方配置 vs V4 最优结果

## Original PPLAD 官方配置

严格使用官方 `scripts/SMD.sh`：

- `anormly_ratio=0.90`
- `win_size=105`
- `batch_size=128`
- `num_epochs=3`
- `local_size=7`
- `global_size=11`
- `d_model=128`
- `r=0.9`
- `lr=1e-4`
- 输入通道数：38

## V4 最优结果协议

采用此前已验证的较优 SMD 训练预算：

- `win_size=100`
- `batch_size=128`
- `num_epochs=10`
- `lr=1e-3`
- local candidates: `1..8`, Top-k=2
- global candidates: `12,16,20,24,28,32,40,48`, Top-k=4
- 联合搜索 `gap`、`total`、`combined`
- 搜索 `anormly_ratio=0.10..3.00`，步长 `0.01`
- 选择最高 PA-F1

此前记录中：

- `win=60 / 3 epochs / ratio=0.50` 的 V4 PA-F1 约为 `0.8659`
- `win=100 / 10 epochs` 的 V4 PA-F1 约为 `0.8928`

因此本轮使用后者，并重新搜索分数模式和阈值。

## 高效阈值搜索

SMD 的窗口化分数样本数量很大。脚本不对每个阈值重复执行完整
Point Adjustment（点调整）扫描，而是使用与其数学等价的方式：

1. 正常点分数排序后用二分查找计算 FP/TN；
2. 每个连续异常区间保存最大分数和区间长度；
3. 根据“区间内任一点触发即整段命中”的 PA 规则计算 TP/FN。

最终最优阈值仍会显式生成 RAW prediction（原始预测）和
PA prediction（点调整预测）文件。

## 轻量化受控基准

两种模型统一：

- `win_size=105`
- `batch_size=128`
- 相同硬件、预热次数、重复次数与测试集

## 数据要求

```text
dataset/SMD/
├── SMD_train.npy
├── SMD_test.npy
└── SMD_test_label.npy
```

预期通道数为 38。

## 一键运行

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/CODE

chmod +x \
  run_smd_official_vs_v4_best.py \
  benchmark_smd_official_vs_v4_best.py \
  run_smd_official_vs_v4_best_full.sh

python -m py_compile \
  run_smd_official_vs_v4_best.py \
  benchmark_smd_official_vs_v4_best.py

bash run_smd_official_vs_v4_best_full.sh
```

## 主要输出

```text
results/SMD_OFFICIAL_VS_V4_BEST/
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
