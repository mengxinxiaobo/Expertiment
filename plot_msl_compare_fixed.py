#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("/mnt/c/Users/DING/Desktop/Experiment/CODE")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_first(data: dict[str, Any], keys: Iterable[str]) -> float:
    """按顺序读取第一个存在的数值字段，兼容旧版和新版结果键名。"""
    for key in keys:
        if key in data and data[key] is not None:
            return float(data[key])
    raise KeyError(f"以下字段均不存在：{list(keys)}\n实际字段：{sorted(data.keys())}")


def add_bar_labels(ax, bars, formatter) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            formatter(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_figure(fig, output_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def grouped_bar_chart(
    labels: list[str],
    original_values: list[float],
    v4_values: list[float],
    title: str,
    ylabel: str,
    output_base: Path,
    formatter=lambda value: f"{value:.4f}",
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_original = ax.bar(
        x - width / 2,
        original_values,
        width,
        label="Original PPLAD",
    )
    bars_v4 = ax.bar(
        x + width / 2,
        v4_values,
        width,
        label="V4 Best",
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    if ylim is not None:
        ax.set_ylim(*ylim)

    add_bar_labels(ax, bars_original, formatter)
    add_bar_labels(ax, bars_v4, formatter)
    save_figure(fig, output_base)


def single_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_base: Path,
    formatter=lambda value: f"{value:.2f}",
    reference_line: float | None = None,
) -> None:
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(x, values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    if reference_line is not None:
        ax.axhline(reference_line, linewidth=1, linestyle="--")

    add_bar_labels(ax, bars, formatter)
    save_figure(fig, output_base)


def main(root: Path = DEFAULT_ROOT) -> None:
    result_root = root / "results" / "MSL_OFFICIAL_VS_V4_BEST"
    benchmark_root = result_root / "BENCHMARK"
    output_dir = result_root / "FIGURES"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_metrics = load_json(result_root / "original_official_metrics.json")
    v4_run = load_json(result_root / "v4_best_run.json")
    original_bench = load_json(benchmark_root / "original.json")
    v4_bench = load_json(benchmark_root / "v4_best.json")

    v4_best = v4_run["best"]

    # 1. Detection performance
    detection_labels = ["PA-Precision", "PA-Recall", "PA-F1"]
    original_detection = [
        float(original_metrics["pa_precision"]),
        float(original_metrics["pa_recall"]),
        float(original_metrics["pa_f1"]),
    ]
    v4_detection = [
        float(v4_best["pa_precision"]),
        float(v4_best["pa_recall"]),
        float(v4_best["pa_f1"]),
    ]

    grouped_bar_chart(
        detection_labels,
        original_detection,
        v4_detection,
        title="MSL Detection Performance",
        ylabel="Score",
        output_base=output_dir / "01_detection_performance",
        formatter=lambda value: f"{value:.4f}",
        ylim=(0.85, 1.01),
    )

    # 2. Absolute model size
    original_params = get_first(original_bench, ["trainable_params"])
    v4_params = get_first(v4_bench, ["trainable_params"])
    original_state_kib = get_first(
        original_bench,
        ["state_dict_disk_bytes", "checkpoint_disk_bytes", "serialized_state_dict_bytes"],
    ) / 1024.0
    v4_state_kib = get_first(
        v4_bench,
        ["state_dict_disk_bytes", "checkpoint_disk_bytes", "serialized_state_dict_bytes"],
    ) / 1024.0

    grouped_bar_chart(
        ["Trainable Params", "State Dict (KiB)"],
        [original_params, original_state_kib],
        [v4_params, v4_state_kib],
        title="MSL Model Size",
        ylabel="Count / KiB",
        output_base=output_dir / "02_model_size",
        formatter=lambda value: f"{value:,.2f}",
    )

    # 3. V4 retained percentage for lower-is-better resource metrics.
    resource_keys = [
        (
            "Parameters",
            ["trainable_params"],
        ),
        (
            "State Dict",
            ["state_dict_disk_bytes", "checkpoint_disk_bytes", "serialized_state_dict_bytes"],
        ),
        (
            "Online latency",
            ["online_batch1_latency_ms_mean"],
        ),
        (
            "Batch latency",
            ["single_batch_latency_ms_mean"],
        ),
        (
            "Full-test time",
            ["full_test_seconds_mean", "full_test_seconds"],
        ),
        (
            "GPU peak",
            ["full_test_gpu_peak_allocated_mib"],
        ),
        (
            "GPU incremental",
            ["full_test_gpu_incremental_allocated_mib"],
        ),
        (
            "CPU RSS",
            ["full_test_cpu_incremental_rss_mib"],
        ),
    ]

    retained_labels: list[str] = []
    retained_values: list[float] = []
    for label, aliases in resource_keys:
        original_value = get_first(original_bench, aliases)
        v4_value = get_first(v4_bench, aliases)
        retained_labels.append(label)
        retained_values.append(v4_value / original_value * 100.0)

    single_bar_chart(
        retained_labels,
        retained_values,
        title="V4 Resource Cost Relative to Original PPLAD",
        ylabel="V4 / Original (%) — lower is better",
        output_base=output_dir / "03_resource_retained_percentage",
        formatter=lambda value: f"{value:.1f}%",
        reference_line=100.0,
    )

    # 4. Throughput comparison
    original_throughput = get_first(
        original_bench,
        ["full_test_points_per_second"],
    )
    v4_throughput = get_first(
        v4_bench,
        ["full_test_points_per_second"],
    )

    grouped_bar_chart(
        ["Full-test throughput"],
        [original_throughput],
        [v4_throughput],
        title="MSL Full-Test Throughput",
        ylabel="Points per second",
        output_base=output_dir / "04_throughput",
        formatter=lambda value: f"{value:,.0f}",
    )

    # 5. Training time: total and per epoch must be separated conceptually.
    original_training_total = float(original_metrics["training_seconds"])
    v4_training_total = float(v4_run["training_seconds"])
    original_epochs = int(original_metrics["config"]["num_epochs"])
    v4_epochs = int(v4_run["config"]["num_epochs"])

    grouped_bar_chart(
        ["Total time", "Time per epoch"],
        [
            original_training_total,
            original_training_total / original_epochs,
        ],
        [
            v4_training_total,
            v4_training_total / v4_epochs,
        ],
        title="MSL Training Time",
        ylabel="Seconds",
        output_base=output_dir / "05_training_time",
        formatter=lambda value: f"{value:.3f}",
    )

    # 6. Export clean comparison data for papers/tables.
    csv_path = output_dir / "msl_comparison_data.csv"
    rows = [
        ("PA-Precision", original_detection[0], v4_detection[0]),
        ("PA-Recall", original_detection[1], v4_detection[1]),
        ("PA-F1", original_detection[2], v4_detection[2]),
        ("Trainable parameters", original_params, v4_params),
        ("State Dict KiB", original_state_kib, v4_state_kib),
        (
            "Online batch1 latency ms",
            get_first(original_bench, ["online_batch1_latency_ms_mean"]),
            get_first(v4_bench, ["online_batch1_latency_ms_mean"]),
        ),
        (
            "Batch256 latency ms",
            get_first(original_bench, ["single_batch_latency_ms_mean"]),
            get_first(v4_bench, ["single_batch_latency_ms_mean"]),
        ),
        (
            "Full-test seconds",
            get_first(original_bench, ["full_test_seconds_mean", "full_test_seconds"]),
            get_first(v4_bench, ["full_test_seconds_mean", "full_test_seconds"]),
        ),
        ("Throughput points/s", original_throughput, v4_throughput),
        (
            "GPU peak MiB",
            get_first(original_bench, ["full_test_gpu_peak_allocated_mib"]),
            get_first(v4_bench, ["full_test_gpu_peak_allocated_mib"]),
        ),
        (
            "GPU incremental MiB",
            get_first(original_bench, ["full_test_gpu_incremental_allocated_mib"]),
            get_first(v4_bench, ["full_test_gpu_incremental_allocated_mib"]),
        ),
        (
            "CPU incremental RSS MiB",
            get_first(original_bench, ["full_test_cpu_incremental_rss_mib"]),
            get_first(v4_bench, ["full_test_cpu_incremental_rss_mib"]),
        ),
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Original PPLAD", "V4 Best", "V4 relative change (%)"])
        for metric, original_value, v4_value in rows:
            relative_change = (v4_value / original_value - 1.0) * 100.0
            writer.writerow([metric, original_value, v4_value, relative_change])

    print("图表已生成：")
    for path in sorted(output_dir.glob("*.png")):
        print(path)
    for path in sorted(output_dir.glob("*.pdf")):
        print(path)
    print(csv_path)


if __name__ == "__main__":
    main()
