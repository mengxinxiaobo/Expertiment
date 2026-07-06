# Original PPLAD 与 V4 实验对比

实验目录：`/mnt/c/Users/DING/Desktop/Experiment/CODE/comparison_runs/20260705_173046`

## 检测性能

| dataset   |   original_pa_precision |   v4_pa_precision |   original_pa_recall |   v4_pa_recall |   original_pa_f1 |   v4_pa_f1 |   original_roc_auc |   v4_roc_auc |   original_project_r_auc_pr |   v4_project_r_auc_pr |   original_project_vus_pr |   v4_project_vus_pr |
|:----------|------------------------:|------------------:|---------------------:|---------------:|-----------------:|-----------:|-------------------:|-------------:|----------------------------:|----------------------:|--------------------------:|--------------------:|
| SMAP      |                0.944174 |          0.897852 |             0.988244 |       0.927892 |         0.965706 |   0.912625 |            0.50753 |     0.492228 |                    0.928822 |              0.854048 |                  0.926339 |            0.850931 |

## 轻量化结果

| dataset   | protocol   |   original_trainable_params |   v4_trainable_params |   original_single_batch_latency_ms_mean |   v4_single_batch_latency_ms_mean |   original_full_test_seconds |   v4_full_test_seconds |   original_full_test_points_per_second |   v4_full_test_points_per_second |   original_full_test_gpu_peak_allocated_mib |   v4_full_test_gpu_peak_allocated_mib |
|:----------|:-----------|----------------------------:|----------------------:|----------------------------------------:|----------------------------------:|-----------------------------:|-----------------------:|---------------------------------------:|---------------------------------:|--------------------------------------------:|--------------------------------------:|
| SMAP      | controlled |                        1281 |                   146 |                                 3.44895 |                           5.31004 |                     0.428056 |               0.468787 |                       998935           |                           912140 |                                     42.812  |                                56.064 |
| SMAP      | native     |                        1281 |                   146 |                                 3.81776 |                           3.68838 |                     0.411971 |               0.505819 |                            1.03791e+06 |                           845361 |                                     38.4272 |                                56.064 |

## 解释规则

- 检测指标的 `delta` 为 V4 − Original。
- 资源指标的 `reduction_percent` 为 `(Original − V4) / Original × 100%`。
- 延迟和内存越低越好；吞吐率越高越好。
- `native` 使用各模型自己的正式配置；`controlled` 使用相同窗口和 batch。