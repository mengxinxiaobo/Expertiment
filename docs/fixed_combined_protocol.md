# ASCA-AD / V4：所有数据集固定 combined score 重新评估

## 功能

这个脚本直接调用已经训练好的 V4 checkpoint，不重新训练模型。

固定：

```text
score_mode = combined
```

不再进行：

```text
total / gap / combined 中按测试集 PA-F1 选最优
```

默认 `anormly_ratio` 对齐 Original PPLAD 官方/默认配置：

```text
MSL  = 0.83
SMAP = 0.80
PSM  = 0.80
HAI  = 0.98
SMD  = 0.90
WADI = 0.50
PUMP = 0.50
SKAB = 0.50
```

阈值仍使用 PPLAD 风格协议：

```text
threshold = percentile(train_combined_score + test_combined_score, 100 - anormly_ratio)
```

## 运行

把以下文件放到项目根目录：

```text
run_all_v4_combined_fixed.py
run_all_v4_combined_fixed.sh
run_fast_v4_combined_fixed.sh
```

### 先跑快的数据集

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/CODE
chmod +x run_all_v4_combined_fixed.py run_all_v4_combined_fixed.sh run_fast_v4_combined_fixed.sh
bash run_fast_v4_combined_fixed.sh
```

### 跑全部数据集

```bash
bash run_all_v4_combined_fixed.sh
```

### 指定单个或多个数据集

```bash
python -u run_all_v4_combined_fixed.py \
  --datasets SKAB,PUMP,PSM \
  --output-dir results/FAST_COMBINED_FIXED
```

### 临时覆盖 ratio

```bash
python -u run_all_v4_combined_fixed.py \
  --datasets SKAB \
  --ratio-overrides SKAB=0.5
```

如果某个 checkpoint 不存在，先运行原来的 V4 训练脚本，或者加：

```bash
--train-if-missing
```

## 输出

```text
results/ALL_COMBINED_FIXED/
├── summary.md
├── all_combined_fixed_metrics.csv
├── all_combined_fixed_metrics.json
└── SKAB/
    ├── summary.md
    ├── combined_fixed_metrics.json
    ├── score_combined.txt
    ├── label.txt
    ├── pred_raw.txt
    └── pred_pa.txt
```


## 兼容性修复

新版脚本内置 Precision / Recall / F1 / PA-F1 和 Point Adjustment 计算函数，因此即使某些旧数据集 runner 没有 `fast_threshold_metrics`，也可以正常运行。
