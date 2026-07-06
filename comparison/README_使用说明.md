# Original PPLAD 与 V4 对比实验

把整个 `comparison/` 目录放到项目根目录。预期结构：

```text
Expertiment/
├── main.py
├── solver.py
├── dataset/
│   ├── SMAP/
│   ├── MSL/
│   ├── PSM/
│   ├── HAI/
│   └── SMD/
├── Baseline/
│   └── pplad-main/
│       ├── main.py
│       ├── solver.py
│       ├── model/PPLAD.py
│       └── metrics/
└── comparison/
    ├── config.json
    ├── run_detection_compare.py
    ├── benchmark_one.py
    ├── run_benchmark_compare.py
    ├── make_report.py
    └── run_all.sh
```

## 1. 安装额外依赖

在原来能运行 V4 的 Conda 环境中：

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/Expertiment
conda activate ML
pip install -r comparison/requirements-comparison.txt
```

原版与 V4 共用当前环境。不要同时运行两个模型，脚本会顺序执行。

## 2. 检查路径

`comparison/config.json` 默认：

```json
{
  "project_root": "..",
  "v4_root": ".",
  "original_root": "Baseline/pplad-main",
  "dataset_root": "dataset"
}
```

若本地根目录名称或层级不同，只修改这三项即可。

原版 PPLAD 内部固定读取：

```text
Baseline/pplad-main/dataset/<DATASET>/
```

检测脚本会自动在该位置创建软链接，指向根目录的：

```text
dataset/<DATASET>/
```

所以不需要复制大型 `.npy` 文件。

## 3. 先只跑 SMAP

```bash
python comparison/run_detection_compare.py \
  --config comparison/config.json \
  --datasets SMAP
```

完成后运行轻量化基准：

```bash
RUN_ROOT="$(cat comparison_runs/LATEST)"

python comparison/run_benchmark_compare.py \
  --config comparison/config.json \
  --run-root "$RUN_ROOT" \
  --datasets SMAP
```

生成报告：

```bash
python comparison/make_report.py \
  --config comparison/config.json \
  --run-root "$RUN_ROOT"
```

也可一条命令完成：

```bash
bash comparison/run_all.sh comparison/config.json SMAP
```

## 4. 批量运行已经核验的五个数据集

```bash
bash comparison/run_all.sh comparison/config.json \
  SMAP MSL PSM HAI SMD
```

注意：HAI、SMD 的 exact threshold（精确阈值）阶段可能耗时较长并占用较大内存。

## 5. 阈值协议

两个模型不强制使用同一个比例。

Original PPLAD 使用官方仓库各数据集脚本配置：

| Dataset | Ratio | Win | Batch | Epoch |
|---|---:|---:|---:|---:|
| SMAP | 0.80 | 90 | 128 | 3 |
| MSL | 0.83 | 90 | 256 | 3 |
| PSM | 0.80 | 60 | 256 | 1 |
| HAI | 0.98 | 90 | 128 | 3 |
| SMD | 0.90 | 105 | 128 | 3 |
| WADI | 0.50 | 90 | 128 | 3 |

V4 使用当前项目脚本及已有实验记录：

| Dataset | Ratio |
|---|---:|
| SMAP | 2.00 |
| MSL | 0.75 |
| PSM | 1.00 |
| HAI | 0.98 |
| SMD | 0.50 |
| WADI | 0.50 |
| SKAB | 2.00 |
| PUMP | 0.50 |

SKAB 和 PUMP 在官方 PPLAD 仓库中没有对应 shell 配置，默认关闭。核验本地旧脚本后，再填写 `config.json` 中的 Original 配置并设置 `"enabled": true`。

## 6. 保存的检测指标

每个模型都会保存：

- RAW Accuracy、Precision、Recall、F1、MCC
- PA Accuracy、Precision、Recall、F1、MCC
- 标准 ROC-AUC
- Average Precision / PR-AUC
- 当前项目已有的 Affiliation Precision / Recall
- R-AUC-ROC、R-AUC-PR
- VUS-ROC、VUS-PR
- TP、FP、TN、FN
- 阈值、异常比例、分数统计
- 总运行时间、epoch 时间
- 运行进程树 CPU RSS 峰值
- 运行进程树 GPU 显存峰值

## 7. 保存的轻量化指标

`native` 协议：各模型使用各自正式窗口和 batch。

`controlled` 协议：两个模型统一使用：

```text
win_size=100
batch_size=128
```

保存：

- Trainable parameters（可训练参数量）
- Total parameters（总参数量）
- 参数张量字节数
- 序列化 state_dict 大小
- 实际 checkpoint 文件大小
- 单 batch 平均、标准差、P50、P95、P99 时延
- 单窗口平均时延
- 完整测试集运行时间
- windows/s 与 points/s
- CPU RSS 峰值与增量
- GPU allocated/reserved 峰值与增量
- exact threshold 时间、样本数和峰值内存

原版官方代码不保存训练 checkpoint。因此原版轻量化基准重新实例化同结构模型。参数量、state_dict 大小和前向计算复杂度不依赖训练后的具体权重值。

## 8. 结果目录

```text
comparison_runs/<时间戳>/
├── environment.json
├── config_snapshot.json
├── dataset_metadata.json
├── detection_metrics.csv
├── detection_comparison.csv
├── benchmark_metrics.csv
├── lightweight_comparison.csv
├── comparison_summary.json
├── summary.md
└── SMAP/
    ├── original/
    │   ├── command.txt
    │   ├── stdout.log
    │   ├── resource.json
    │   ├── metrics.json
    │   ├── original_score.txt
    │   ├── original_fact.txt
    │   ├── pred_raw_unified.txt
    │   └── pred_pa_unified.txt
    └── v4/
        ├── command.txt
        ├── stdout.log
        ├── resource.json
        ├── metrics.json
        ├── checkpoints/
        ├── model_outputs/
        └── benchmark/
```

## 9. 只重新生成报告

```bash
RUN_ROOT="$(cat comparison_runs/LATEST)"

python comparison/make_report.py \
  --config comparison/config.json \
  --run-root "$RUN_ROOT"
```

## 10. 常见问题

### 原版提示找不到数据

检查：

```bash
ls -l Baseline/pplad-main/dataset/
ls -lh dataset/SMAP/
```

软链接目标中必须存在：

```text
SMAP_train.npy
SMAP_test.npy
SMAP_test_label.npy
```

### `psutil` 不存在

```bash
pip install psutil
```

即使没有 `psutil`，脚本仍会尝试读取主进程 RSS，但无法完整统计 DataLoader 子进程，所以正式实验应安装。

### HAI/SMD 在 Saved checkpoint 后很久

这是 exact threshold 阶段重新遍历训练窗口和测试窗口、收集完整异常分数并计算百分位造成的，不是程序卡死。
