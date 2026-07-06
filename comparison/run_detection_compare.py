#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import (
    dataset_metadata,
    environment_metadata,
    evaluate_scores,
    flatten_dict,
    link_dataset,
    load_config,
    parse_threshold,
    parse_training_times,
    resolve_from,
    run_logged,
    write_csv,
    write_json,
)


V4_SHARED_ARGS = [
    "--local_candidate_lags", "1", "2", "3", "4", "5", "6", "7", "8",
    "--global_candidate_lags", "12", "16", "20", "24", "28", "32", "40", "48",
    "--local_topk", "2",
    "--global_topk", "4",
    "--selector_hidden", "8",
    "--fitter_hidden", "8",
    "--selector_temperature", "0.5",
    "--similarity_tau", "1.0",
    "--sigma_min", "0.03",
    "--sigma_max", "1.50",
    "--area_weight", "0.1",
    "--selector_balance_weight", "0.05",
    "--gap_weight", "1.0",
    "--relation_input", "instance",
    "--score_modes", "total",
    "--primary_score", "total",
    "--score_normalization", "official",
    "--threshold_source", "original",
    "--quantile_method", "exact",
]


def original_command(
    python: str,
    original_root: Path,
    dataset_key: str,
    dataset_cfg: Dict[str, Any],
    channels: int,
) -> List[str]:
    cfg = dataset_cfg["original"]
    return [
        python,
        str(original_root / "main.py"),
        "--anormly_ratio", str(cfg["anormly_ratio"]),
        "--num_epochs", str(cfg["num_epochs"]),
        "--batch_size", str(cfg["batch_size"]),
        "--d_model", str(cfg["d_model"]),
        "--dataset", dataset_cfg["loader_dataset"],
        "--data_path", dataset_cfg["original_data_link"],
        "--input_c", str(channels),
        "--output_c", str(channels),
        "--global_size", ",".join(str(x) for x in cfg["global_size"]),
        "--local_size", ",".join(str(x) for x in cfg["local_size"]),
        "--r", str(cfg["r"]),
        "--win_size", str(cfg["win_size"]),
        "--lr", str(cfg.get("lr", 1e-4)),
        "--similar", str(cfg.get("similar", "MSE")),
        "--loss_fuc", str(cfg.get("loss_fuc", "MSE")),
        "--model_save_path", "checkpoints",
    ]


def v4_command(
    python: str,
    v4_root: Path,
    dataset_key: str,
    dataset_cfg: Dict[str, Any],
    channels: int,
    model_dir: Path,
) -> List[str]:
    cfg = dataset_cfg["v4"]
    return [
        python,
        str(v4_root / "main.py"),
        "--mode", "train",
        "--dataset", dataset_cfg["loader_dataset"],
        "--data_path", dataset_cfg["folder"],
        "--input_c", str(channels),
        "--output_c", str(channels),
        "--win_size", str(cfg["win_size"]),
        "--batch_size", str(cfg["batch_size"]),
        "--num_epochs", str(cfg["num_epochs"]),
        "--lr", str(cfg["lr"]),
        "--anormly_ratio", str(cfg["anormly_ratio"]),
        "--seed", "42",
        *V4_SHARED_ARGS,
        "--model_save_path", str(model_dir / "checkpoints"),
        "--result_path", str(model_dir / "model_outputs"),
    ]


def clear_original_outputs(original_root: Path) -> None:
    for name in ("score.txt", "fact.txt", "pred.txt", "discrepancy.txt"):
        path = original_root / name
        if path.exists():
            path.unlink()


def copy_original_outputs(original_root: Path, model_dir: Path) -> Dict[str, Path]:
    mapping = {
        "score": original_root / "score.txt",
        "label": original_root / "fact.txt",
        "pred_pa_author": original_root / "pred.txt",
        "correctness": original_root / "discrepancy.txt",
    }
    output: Dict[str, Path] = {}
    for key, source in mapping.items():
        if not source.exists():
            raise FileNotFoundError(f"原版运行后缺少输出：{source}")
        destination = model_dir / f"original_{source.name}"
        shutil.copy2(source, destination)
        output[key] = destination
    return output


def locate_v4_outputs(
    model_dir: Path,
    dataset_arg: str,
) -> Dict[str, Path]:
    prefix = model_dir / "model_outputs" / f"{dataset_arg}_adaptive_anchor_v4_total"
    candidates = {
        "score": Path(str(prefix) + "_score.txt"),
        "label": Path(str(prefix) + "_label.txt"),
        "pred_raw_author": Path(str(prefix) + "_pred_raw.txt"),
        "pred_pa_author": Path(str(prefix) + "_pred_pa.txt"),
    }
    for path in candidates.values():
        if not path.exists():
            raise FileNotFoundError(f"V4运行后缺少输出：{path}")
    return candidates


def select_datasets(cfg: Dict[str, Any], requested: List[str]) -> List[str]:
    if requested:
        keys = [x.upper() for x in requested]
    else:
        keys = [
            name for name, item in cfg["datasets"].items()
            if item.get("enabled", False)
        ]
    for name in keys:
        if name not in cfg["datasets"]:
            raise KeyError(f"config.json 中没有数据集 {name}")
        original_ratio = cfg["datasets"][name].get("original", {}).get("anormly_ratio")
        if original_ratio is None:
            raise ValueError(
                f"{name} 的原版PPLAD配置尚未核验，不能进行正式对比。"
            )
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Original PPLAD 与 V4 检测性能对比")
    parser.add_argument("--config", default="comparison/config.json")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--hash-data", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg, project_root = load_config(config_path)
    v4_root = resolve_from(project_root, cfg["v4_root"])
    original_root = resolve_from(project_root, cfg["original_root"])
    dataset_root = resolve_from(project_root, cfg["dataset_root"])
    output_parent = resolve_from(project_root, cfg["output_root"])
    python = cfg.get("python", sys.executable)
    poll = float(cfg.get("resource_poll_seconds", 0.1))

    for required in (v4_root / "main.py", original_root / "main.py", dataset_root):
        if not required.exists():
            raise FileNotFoundError(required)

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    run_root = output_parent / tag
    run_root.mkdir(parents=True, exist_ok=False)
    (output_parent / "LATEST").write_text(str(run_root), encoding="utf-8")

    datasets = select_datasets(cfg, args.datasets)
    write_json(run_root / "config_snapshot.json", cfg)
    write_json(
        run_root / "environment.json",
        environment_metadata(project_root, v4_root, original_root),
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if "gpu_index" in cfg:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_index"])

    all_rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    for dataset_key in datasets:
        dc = cfg["datasets"][dataset_key]
        data_dir = dataset_root / dc["folder"]
        meta = dataset_metadata(data_dir, dc["file_prefix"], args.hash_data)
        metadata[dataset_key] = meta
        channels = int(meta["train"]["shape"][1])
        link_dataset(data_dir, original_root, dc["original_data_link"])

        # Original PPLAD
        original_dir = run_root / dataset_key / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        (original_root / "result").mkdir(parents=True, exist_ok=True)
        (original_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        clear_original_outputs(original_root)
        command = original_command(
            python, original_root, dataset_key, dc, channels
        )
        (original_dir / "command.txt").write_text(
            " ".join(command), encoding="utf-8"
        )
        write_json(original_dir / "model_config.json", dc["original"])
        resource = run_logged(
            command,
            cwd=original_root,
            log_path=original_dir / "stdout.log",
            env=env,
            poll_seconds=poll,
        )
        stdout = resource.pop("stdout")
        threshold = parse_threshold(stdout)
        timing = parse_training_times(stdout)
        resource.update(timing)
        if timing["training_epoch_time_total_s"] is not None:
            resource["evaluation_and_overhead_seconds_approx"] = max(
                0.0,
                resource["wall_seconds"] - timing["training_epoch_time_total_s"],
            )
        write_json(original_dir / "resource.json", resource)
        outputs = copy_original_outputs(original_root, original_dir)
        metrics, raw_pred, pa_pred = evaluate_scores(
            outputs["score"], outputs["label"], threshold, project_root
        )
        np.savetxt(original_dir / "pred_raw_unified.txt", raw_pred, fmt="%d")
        np.savetxt(original_dir / "pred_pa_unified.txt", pa_pred, fmt="%d")
        metrics.update(
            {
                "dataset": dataset_key,
                "model": "original",
                "threshold_protocol": "PPLAD official dataset ratio",
                "anormly_ratio": dc["original"]["anormly_ratio"],
                "config_source": dc["original"]["config_source"],
                "win_size": dc["original"]["win_size"],
                "batch_size": dc["original"]["batch_size"],
                "num_epochs": dc["original"]["num_epochs"],
                "wall_seconds": resource["wall_seconds"],
                "peak_process_tree_rss_mib": resource["peak_process_tree_rss_mib"],
                "peak_process_tree_gpu_mib": resource["peak_process_tree_gpu_mib"],
                "training_epoch_time_total_s": resource.get(
                    "training_epoch_time_total_s"
                ),
            }
        )
        write_json(original_dir / "metrics.json", metrics)
        all_rows.append(flatten_dict(metrics))

        # V4
        v4_dir = run_root / dataset_key / "v4"
        v4_dir.mkdir(parents=True, exist_ok=True)
        command = v4_command(python, v4_root, dataset_key, dc, channels, v4_dir)
        (v4_dir / "command.txt").write_text(" ".join(command), encoding="utf-8")
        write_json(v4_dir / "model_config.json", dc["v4"])
        resource = run_logged(
            command,
            cwd=v4_root,
            log_path=v4_dir / "stdout.log",
            env=env,
            poll_seconds=poll,
        )
        stdout = resource.pop("stdout")
        threshold = parse_threshold(stdout)
        timing = parse_training_times(stdout)
        resource.update(timing)
        if timing["training_epoch_time_total_s"] is not None:
            resource["evaluation_and_overhead_seconds_approx"] = max(
                0.0,
                resource["wall_seconds"] - timing["training_epoch_time_total_s"],
            )
        write_json(v4_dir / "resource.json", resource)
        outputs = locate_v4_outputs(v4_dir, dc["loader_dataset"])
        metrics, raw_pred, pa_pred = evaluate_scores(
            outputs["score"], outputs["label"], threshold, project_root
        )
        metrics.update(
            {
                "dataset": dataset_key,
                "model": "v4",
                "threshold_protocol": "V4 fixed ratio from existing experiment record",
                "anormly_ratio": dc["v4"]["anormly_ratio"],
                "config_source": dc["v4"]["config_source"],
                "win_size": dc["v4"]["win_size"],
                "batch_size": dc["v4"]["batch_size"],
                "num_epochs": dc["v4"]["num_epochs"],
                "wall_seconds": resource["wall_seconds"],
                "peak_process_tree_rss_mib": resource["peak_process_tree_rss_mib"],
                "peak_process_tree_gpu_mib": resource["peak_process_tree_gpu_mib"],
                "training_epoch_time_total_s": resource.get(
                    "training_epoch_time_total_s"
                ),
            }
        )
        write_json(v4_dir / "metrics.json", metrics)
        all_rows.append(flatten_dict(metrics))

    write_json(run_root / "dataset_metadata.json", metadata)
    write_json(run_root / "detection_metrics.json", all_rows)
    write_csv(run_root / "detection_metrics.csv", all_rows)

    print("\n检测实验完成：", run_root)
    print("检测汇总：", run_root / "detection_metrics.csv")


if __name__ == "__main__":
    main()
