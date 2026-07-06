#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from common import load_config, resolve_from, write_json


DETECTION_METRICS = [
    "raw_accuracy", "raw_precision", "raw_recall", "raw_f1", "raw_mcc",
    "pa_accuracy", "pa_precision", "pa_recall", "pa_f1", "pa_mcc",
    "roc_auc", "pr_auc_average_precision",
    "project_r_auc_roc", "project_r_auc_pr",
    "project_vus_roc", "project_vus_pr",
    "project_affiliation_precision", "project_affiliation_recall",
]

LIGHTWEIGHT_METRICS = [
    "trainable_params", "serialized_state_dict_bytes",
    "checkpoint_disk_bytes", "single_batch_latency_ms_mean",
    "single_batch_latency_ms_p95", "single_window_latency_ms_mean",
    "full_test_seconds", "full_test_windows_per_second",
    "full_test_points_per_second", "full_test_cpu_peak_rss_mib",
    "full_test_gpu_peak_allocated_mib",
    "exact_threshold_seconds", "exact_threshold_cpu_peak_rss_mib",
    "exact_threshold_gpu_peak_allocated_mib",
]


def numeric(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="生成对比CSV、JSON与Markdown报告")
    p.add_argument("--config", default="comparison/config.json")
    p.add_argument("--run-root", default=None)
    args = p.parse_args()

    cfg, project_root = load_config(Path(args.config))
    output_parent = resolve_from(project_root, cfg["output_root"])
    if args.run_root:
        run_root = Path(args.run_root).resolve()
    else:
        run_root = Path(
            (output_parent / "LATEST").read_text(encoding="utf-8").strip()
        ).resolve()

    det = pd.read_csv(run_root / "detection_metrics.csv")
    bench_path = run_root / "benchmark_metrics.csv"
    bench = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()

    detection_rows: List[Dict[str, Any]] = []
    for dataset, group in det.groupby("dataset"):
        indexed = group.set_index("model")
        if not {"original", "v4"}.issubset(indexed.index):
            continue
        row: Dict[str, Any] = {"dataset": dataset}
        for metric in DETECTION_METRICS:
            if metric not in indexed.columns:
                continue
            original = numeric(indexed.loc["original", metric])
            v4 = numeric(indexed.loc["v4", metric])
            row[f"original_{metric}"] = original
            row[f"v4_{metric}"] = v4
            row[f"delta_{metric}_v4_minus_original"] = (
                None if original is None or v4 is None else v4 - original
            )
        detection_rows.append(row)

    detection_compare = pd.DataFrame(detection_rows)
    detection_compare.to_csv(
        run_root / "detection_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lightweight_rows: List[Dict[str, Any]] = []
    if not bench.empty:
        for (dataset, protocol), group in bench.groupby(["dataset", "protocol"]):
            indexed = group.set_index("model")
            if not {"original", "v4"}.issubset(indexed.index):
                continue
            row = {"dataset": dataset, "protocol": protocol}
            for metric in LIGHTWEIGHT_METRICS:
                if metric not in indexed.columns:
                    continue
                original = numeric(indexed.loc["original", metric])
                v4 = numeric(indexed.loc["v4", metric])
                row[f"original_{metric}"] = original
                row[f"v4_{metric}"] = v4
                if original is not None and v4 is not None:
                    row[f"delta_{metric}_v4_minus_original"] = v4 - original
                    if original != 0:
                        row[f"reduction_percent_{metric}"] = (
                            (original - v4) / original * 100.0
                        )
                    if v4 != 0:
                        row[f"original_div_v4_{metric}"] = original / v4
            lightweight_rows.append(row)

    lightweight_compare = pd.DataFrame(lightweight_rows)
    lightweight_compare.to_csv(
        run_root / "lightweight_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "run_root": str(run_root),
        "detection_comparison": detection_rows,
        "lightweight_comparison": lightweight_rows,
    }
    write_json(run_root / "comparison_summary.json", payload)

    lines = [
        "# Original PPLAD 与 V4 实验对比",
        "",
        f"实验目录：`{run_root}`",
        "",
        "## 检测性能",
        "",
    ]
    display_det = [
        "dataset",
        "original_pa_precision", "v4_pa_precision",
        "original_pa_recall", "v4_pa_recall",
        "original_pa_f1", "v4_pa_f1",
        "original_roc_auc", "v4_roc_auc",
        "original_project_r_auc_pr", "v4_project_r_auc_pr",
        "original_project_vus_pr", "v4_project_vus_pr",
    ]
    existing = [x for x in display_det if x in detection_compare.columns]
    if existing:
        lines.append(detection_compare[existing].to_markdown(index=False))
    else:
        lines.append("没有可展示的检测结果。")

    lines.extend(["", "## 轻量化结果", ""])
    display_light = [
        "dataset", "protocol",
        "original_trainable_params", "v4_trainable_params",
        "original_single_batch_latency_ms_mean",
        "v4_single_batch_latency_ms_mean",
        "original_full_test_seconds", "v4_full_test_seconds",
        "original_full_test_points_per_second",
        "v4_full_test_points_per_second",
        "original_full_test_gpu_peak_allocated_mib",
        "v4_full_test_gpu_peak_allocated_mib",
    ]
    existing = [x for x in display_light if x in lightweight_compare.columns]
    if existing:
        lines.append(lightweight_compare[existing].to_markdown(index=False))
    else:
        lines.append("尚未运行轻量化基准。")

    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- 检测指标的 `delta` 为 V4 − Original。",
            "- 资源指标的 `reduction_percent` 为 `(Original − V4) / Original × 100%`。",
            "- 延迟和内存越低越好；吞吐率越高越好。",
            "- `native` 使用各模型自己的正式配置；`controlled` 使用相同窗口和 batch。",
        ]
    )
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("报告：", run_root / "summary.md")


if __name__ == "__main__":
    main()
