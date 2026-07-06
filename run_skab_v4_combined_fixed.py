#!/usr/bin/env python3
from __future__ import annotations

"""
SKAB fixed-combined protocol for ASCA-AD / V4.

This script fixes score_mode=combined and anormly_ratio=0.50.
It does not choose among total/gap/combined by test PA-F1.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def import_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(runner, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    runner.model.load_state_dict(state)
    runner.model.eval()


def scalar(metrics: dict, key: str) -> float:
    return float(np.asarray(metrics[key]).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/c/Users/DING/Desktop/Experiment/CODE")
    parser.add_argument("--runner", default="run_skab_official_default_vs_v4_best.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v4-epochs", type=int, default=10)
    parser.add_argument("--anormly-ratio", type=float, default=0.50)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--output-dir", default="results/SKAB_COMBINED_FIXED")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    os.chdir(root)

    runner_path = root / args.runner
    if not runner_path.exists():
        raise FileNotFoundError(
            f"找不到 {runner_path}。请先把 run_skab_official_default_vs_v4_best.py 放到项目根目录。"
        )

    module = import_module_from_path(runner_path, "skab_runner_combined_fixed")
    if hasattr(module, "set_seed"):
        module.set_seed(args.seed)

    channels = module.verify_data(root)
    _Solver, V4Solver = module.bootstrap_project(root)
    config = module.v4_config(root, channels, args.seed, args.v4_epochs)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = V4Solver(config)
    checkpoint = Path(runner.checkpoint_path)

    training_seconds = None
    if checkpoint.exists():
        print(f"Loading existing V4 checkpoint: {checkpoint}")
        load_checkpoint(runner, checkpoint)
    else:
        if not args.train_if_missing:
            raise FileNotFoundError(
                f"找不到 V4 checkpoint：{checkpoint}\n"
                "请先运行原 SKAB V4 训练脚本，或加 --train-if-missing 让本脚本训练一次。"
            )
        print("Checkpoint not found. Training V4 first ...")
        start = time.perf_counter()
        runner.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start

    print("Collecting train scores ...")
    train_scores, _ = module.collect_v4_scores(
        runner, runner.train_loader, include_labels=False
    )

    print("Collecting test scores ...")
    test_scores, labels = module.collect_v4_scores(
        runner, runner.thre_loader, include_labels=True
    )
    if labels is None:
        raise RuntimeError("没有收集到测试标签。")

    mode = "combined"
    train_values = np.asarray(train_scores[mode]).reshape(-1)
    test_values = np.asarray(test_scores[mode]).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)

    threshold_pool = np.concatenate([train_values, test_values], axis=0)
    threshold = float(np.percentile(threshold_pool, 100.0 - args.anormly_ratio))

    metric_arrays = module.fast_threshold_metrics(
        test_values,
        labels,
        np.asarray([threshold], dtype=np.float64),
    )
    metrics = {key: scalar(metric_arrays, key) for key in metric_arrays}

    pred_raw = (test_values > threshold).astype(np.int8)
    pred_pa = module.point_adjust(pred_raw, labels)

    np.savetxt(output_dir / "score_combined.txt", test_values, fmt="%.10f")
    np.savetxt(output_dir / "label.txt", labels, fmt="%d")
    np.savetxt(output_dir / "pred_raw.txt", pred_raw, fmt="%d")
    np.savetxt(output_dir / "pred_pa.txt", pred_pa, fmt="%d")

    original_path = root / "results" / "SKAB_OFFICIAL_DEFAULT_VS_V4_BEST" / "original_official_metrics.json"
    original = None
    if original_path.exists():
        original = json.loads(original_path.read_text(encoding="utf-8"))

    comparison = None
    if original is not None:
        comparison = {
            "original_pa_precision": original.get("pa_precision"),
            "original_pa_recall": original.get("pa_recall"),
            "original_pa_f1": original.get("pa_f1"),
            "v4_combined_pa_precision": metrics.get("pa_precision"),
            "v4_combined_pa_recall": metrics.get("pa_recall"),
            "v4_combined_pa_f1": metrics.get("pa_f1"),
            "pa_f1_diff_percentage_points": (
                (metrics.get("pa_f1") - original.get("pa_f1")) * 100.0
                if original.get("pa_f1") is not None and metrics.get("pa_f1") is not None
                else None
            ),
        }

    summary = {
        "dataset": "SKAB",
        "protocol": "fixed_combined_score",
        "score_mode": mode,
        "anormly_ratio": args.anormly_ratio,
        "threshold": threshold,
        "threshold_source": "percentile(train_combined_score + test_combined_score)",
        "training_seconds_if_trained_by_this_script": training_seconds,
        "counts": {
            "train_score_count": int(train_values.size),
            "test_score_count": int(test_values.size),
            "test_anomaly_count": int(labels.sum()),
            "test_anomaly_ratio_percent": float(labels.mean() * 100.0),
        },
        "metrics": metrics,
        "comparison_with_original_if_available": comparison,
    }

    json_path = output_dir / "skab_v4_combined_fixed_metrics.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SKAB：ASCA-AD / V4 fixed combined result",
        "",
        "## Protocol",
        "",
        "- score mode: fixed `combined`",
        f"- anormly_ratio: `{args.anormly_ratio}`",
        "- threshold: percentile over train + test combined scores",
        "- no best-over-score-mode selection",
        "",
        "## Detection metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Raw Precision | {metrics.get('raw_precision', float('nan')):.10f} |",
        f"| Raw Recall | {metrics.get('raw_recall', float('nan')):.10f} |",
        f"| Raw F1 | {metrics.get('raw_f1', float('nan')):.10f} |",
        f"| PA Precision | {metrics.get('pa_precision', float('nan')):.10f} |",
        f"| PA Recall | {metrics.get('pa_recall', float('nan')):.10f} |",
        f"| PA F1 | {metrics.get('pa_f1', float('nan')):.10f} |",
        "",
    ]
    if comparison is not None:
        lines.extend([
            "## Comparison with Original PPLAD",
            "",
            "| Model | PA-Precision | PA-Recall | PA-F1 |",
            "|---|---:|---:|---:|",
            f"| Original PPLAD | {comparison['original_pa_precision']:.10f} | {comparison['original_pa_recall']:.10f} | {comparison['original_pa_f1']:.10f} |",
            f"| ASCA-AD / V4 fixed combined | {comparison['v4_combined_pa_precision']:.10f} | {comparison['v4_combined_pa_recall']:.10f} | {comparison['v4_combined_pa_f1']:.10f} |",
            "",
            f"PA-F1 diff: {comparison['pa_f1_diff_percentage_points']:.4f} percentage points.",
            "",
        ])

    md_path = output_dir / "summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== SKAB fixed combined result ==========")
    print(f"score_mode     : {mode}")
    print(f"anormly_ratio  : {args.anormly_ratio}")
    print(f"threshold      : {threshold}")
    print(f"PA-Precision   : {metrics.get('pa_precision')}")
    print(f"PA-Recall      : {metrics.get('pa_recall')}")
    print(f"PA-F1          : {metrics.get('pa_f1')}")
    if comparison is not None:
        print(f"Original PA-F1 : {comparison['original_pa_f1']}")
        print(f"PA-F1 diff     : {comparison['pa_f1_diff_percentage_points']:.4f} pp")
    print(f"Saved JSON     : {json_path}")
    print(f"Saved summary  : {md_path}")


if __name__ == "__main__":
    main()
