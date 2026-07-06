# SKAB：ASCA-AD / V4 固定 combined score 快速验证

## 目的

先用最快的 SKAB 数据集验证固定评分协议。

这个脚本不再从 `total / gap / combined` 中按测试集 PA-F1 选最优，而是固定：

```text
score_mode = combined
anormly_ratio = 0.50
```

其中 `anormly_ratio=0.50` 与 SKAB 的 Original PPLAD 官方默认配置对齐。

阈值仍按照 PPLAD 官方风格使用：

```text
threshold = percentile(train_score + test_score, 100 - anormly_ratio)
```

测试标签只用于最后计算 Precision / Recall / F1 / PA-F1。

## 运行

把下面两个文件放到项目根目录：

```text
run_skab_v4_combined_fixed.py
run_skab_v4_combined_fixed.sh
```

运行：

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/CODE
chmod +x run_skab_v4_combined_fixed.py run_skab_v4_combined_fixed.sh
bash run_skab_v4_combined_fixed.sh
```

默认读取已经训练好的 V4 checkpoint。

如果 checkpoint 不存在，可以训练一次：

```bash
python -u run_skab_v4_combined_fixed.py --train-if-missing
```

## 输出

```text
results/SKAB_COMBINED_FIXED/
├── skab_v4_combined_fixed_metrics.json
├── summary.md
├── score_combined.txt
├── label.txt
├── pred_raw.txt
└── pred_pa.txt
```
