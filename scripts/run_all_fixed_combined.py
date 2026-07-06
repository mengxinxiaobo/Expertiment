#!/usr/bin/env python3
from __future__ import annotations

"""
Run fixed-combined evaluation for all existing ASCA-AD / V4 checkpoints.

This script does not retrain the model by default.
It loads the trained V4 checkpoint for each dataset and evaluates a fixed
combined anomaly score:

    score_mode = combined

No best-over-score-mode selection is performed.

Compatibility note: metrics and point adjustment are computed locally, so this script also works with older dataset runners that do not expose fast_threshold_metrics.

Default anormly_ratio values follow the Original PPLAD official/default
configuration for each dataset:

    MSL  = 0.83
    SMAP = 0.80
    PSM  = 0.80
    HAI  = 0.98
    SMD  = 0.90
    WADI = 0.50
    PUMP = 0.50
    SKAB = 0.50

Threshold protocol follows the PPLAD-style percentile rule:

    threshold = percentile(train_combined_score + test_combined_score,
                           100 - anormly_ratio)

Test labels are used only for final metric calculation.
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


DATASET_META: Dict[str, Dict[str, object]] = {
    "MSL": {
        "runner_candidates": [
            "run_msl_official_vs_v4_best_v3.py",
            "run_msl_official_vs_v4_best_v2.py",
            "run_msl_official_vs_v4_best.py",
        ],
        "default_ratio": 0.83,
        "original_result_candidates": [
            "results/MSL_OFFICIAL_VS_V4_BEST/original_official_metrics.json",
            "results/MSL_OFFICIAL_DEFAULT_VS_V4_BEST/original_official_metrics.json",
        ],
    },
    "SMAP": {
        "runner_candidates": ["run_smap_official_vs_v4_best.py"],
        "default_ratio": 0.80,
        "original_result_candidates": [
            "results/SMAP_OFFICIAL_VS_V4_BEST/original_official_metrics.json",
        ],
    },
    "PSM": {
        "runner_candidates": ["run_psm_official_vs_v4_best.py"],
        "default_ratio": 0.80,
        "original_result_candidates": [
            "results/PSM_OFFICIAL_VS_V4_BEST/original_official_metrics.json",
        ],
    },
    "HAI": {
        "runner_candidates": ["run_hai_official_vs_v4_best.py"],
        "default_ratio": 0.98,
        "original_result_candidates": [
            "results/HAI_OFFICIAL_VS_V4_BEST/original_official_metrics.json",
        ],
    },
    "SMD": {
        "runner_candidates": ["run_smd_official_vs_v4_best.py"],
        "default_ratio": 0.90,
        "original_result_candidates": [
            "results/SMD_OFFICIAL_VS_V4_BEST/original_official_metrics.json",
        ],
    },
    "WADI": {
        "runner_candidates": ["run_wadi_official_default_vs_v4_best.py"],
        "default_ratio": 0.50,
        "original_result_candidates": [
            "results/WADI_OFFICIAL_DEFAULT_VS_V4_BEST/original_official_metrics.json",
        ],
    },
    "PUMP": {
        "runner_candidates": ["run_pump_official_default_vs_v4_best.py"],
        "default_ratio": 0.50,
        "original_result_candidates": [
            "results/PUMP_OFFICIAL_DEFAULT_VS_V4_BEST/original_official_metrics.json",
        ],
    },
    "SKAB": {
        "runner_candidates": ["run_skab_official_default_vs_v4_best.py"],
        "default_ratio": 0.50,
        "original_result_candidates": [
            "results/SKAB_OFFICIAL_DEFAULT_VS_V4_BEST/original_official_metrics.json",
        ],
    },
}


def parse_datasets(value: str) -> List[str]:
    value = value.strip()
    if value.lower() == "all":
        return list(DATASET_META.keys())
    datasets = [x.strip().upper() for x in value.split(",") if x.strip()]
    unknown = [x for x in datasets if x not in DATASET_META]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return datasets


def parse_ratio_overrides(value: str) -> Dict[str, float]:
    """
    Example:
        MSL=0.83,SMAP=0.8,SKAB=0.5
    """
    result: Dict[str, float] = {}
    value = value.strip()
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad ratio override: {item}")
        key, raw = item.split("=", 1)
        dataset = key.strip().upper()
        if dataset not in DATASET_META:
            raise ValueError(f"Unknown dataset in ratio override: {dataset}")
        result[dataset] = float(raw)
    return result


def locate_first_existing(root: Path, candidates: List[str]) -> Optional[Path]:
    search_roots = [
        root,
        root / "scripts" / "dataset_runners",
        root / "legacy",
    ]

    for item in candidates:
        item_path = Path(item)

        if item_path.is_absolute() and item_path.exists():
            return item_path

        for base in search_roots:
            path = base / item
            if path.exists():
                return path

    return None


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


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred).astype(int).reshape(-1)
    gt = np.asarray(gt).astype(int).reshape(-1)

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / pred.size if pred.size > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def point_adjust_np(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred).astype(int).copy().reshape(-1)
    gt = np.asarray(gt).astype(int).reshape(-1)

    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True

            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1

            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1

        elif gt[i] == 0:
            anomaly_state = False

        if anomaly_state:
            pred[i] = 1

    return pred.astype(np.int8)


def local_threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, np.ndarray]:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)
    thresholds = np.asarray(thresholds).reshape(-1)

    output = {
        "raw_precision": [],
        "raw_recall": [],
        "raw_f1": [],
        "raw_accuracy": [],
        "pa_precision": [],
        "pa_recall": [],
        "pa_f1": [],
        "pa_accuracy": [],
    }

    for threshold in thresholds:
        pred_raw = (scores > float(threshold)).astype(np.int8)
        raw = binary_metrics(pred_raw, labels)

        pred_pa = point_adjust_np(pred_raw, labels)
        pa = binary_metrics(pred_pa, labels)

        output["raw_precision"].append(raw["precision"])
        output["raw_recall"].append(raw["recall"])
        output["raw_f1"].append(raw["f1"])
        output["raw_accuracy"].append(raw["accuracy"])

        output["pa_precision"].append(pa["precision"])
        output["pa_recall"].append(pa["recall"])
        output["pa_f1"].append(pa["f1"])
        output["pa_accuracy"].append(pa["accuracy"])

    return {key: np.asarray(value, dtype=np.float64) for key, value in output.items()}


def scalar(metric_arrays: dict, key: str) -> float:
    return float(np.asarray(metric_arrays[key]).reshape(-1)[0])


def extract_original_metrics(root: Path, dataset: str) -> Optional[dict]:
    candidates = DATASET_META[dataset]["original_result_candidates"]
    path = locate_first_existing(root, candidates)  # type: ignore[arg-type]
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        return data
    except Exception as exc:
        print(f"[{dataset}] Warning: cannot read original metrics {path}: {exc}")
        return None


def evaluate_dataset(
    root: Path,
    dataset: str,
    ratio: float,
    output_root: Path,
    seed: int,
    v4_epochs: int,
    train_if_missing: bool,
) -> dict:
    meta = DATASET_META[dataset]
    runner_path = locate_first_existing(root, meta["runner_candidates"])  # type: ignore[arg-type]
    if runner_path is None:
        raise FileNotFoundError(
            f"[{dataset}] Cannot find runner. Expected one of: {meta['runner_candidates']}"
        )

    print("\n" + "=" * 80)
    print(f"[{dataset}] Fixed combined evaluation")
    print("=" * 80)
    print(f"[{dataset}] runner         : {runner_path}")
    print(f"[{dataset}] anormly_ratio  : {ratio}")

    module = import_module_from_path(runner_path, f"runner_{dataset.lower()}_combined_fixed")

    if hasattr(module, "set_seed"):
        module.set_seed(seed)

    channels = module.verify_data(root)
    _Solver, V4Solver = module.bootstrap_project(root)
    config = module.v4_config(root, channels, seed, v4_epochs)

    runner = V4Solver(config)
    checkpoint = Path(runner.checkpoint_path)

    training_seconds = None
    if checkpoint.exists():
        print(f"[{dataset}] loading checkpoint: {checkpoint}")
        load_checkpoint(runner, checkpoint)
    else:
        if not train_if_missing:
            raise FileNotFoundError(
                f"[{dataset}] Missing V4 checkpoint: {checkpoint}\n"
                "Run the dataset's V4 training script first, or add --train-if-missing."
            )
        print(f"[{dataset}] checkpoint missing; training V4 first ...")
        start = time.perf_counter()
        runner.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start

    if not hasattr(runner, "score_modes") or "combined" not in list(runner.score_modes):
        raise RuntimeError(f"[{dataset}] runner does not expose score mode 'combined'.")

    print(f"[{dataset}] collecting train scores ...")
    train_scores, _ = module.collect_v4_scores(
        runner, runner.train_loader, include_labels=False
    )

    print(f"[{dataset}] collecting test scores ...")
    test_scores, labels = module.collect_v4_scores(
        runner, runner.thre_loader, include_labels=True
    )
    if labels is None:
        raise RuntimeError(f"[{dataset}] No labels collected from test loader.")

    mode = "combined"
    train_values = np.asarray(train_scores[mode]).reshape(-1)
    test_values = np.asarray(test_scores[mode]).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)

    threshold_pool = np.concatenate([train_values, test_values], axis=0)
    threshold = float(np.percentile(threshold_pool, 100.0 - ratio))

    metric_arrays = local_threshold_metrics(
        test_values,
        labels,
        np.asarray([threshold], dtype=np.float64),
    )
    metrics = {key: scalar(metric_arrays, key) for key in metric_arrays}

    pred_raw = (test_values > threshold).astype(np.int8)
    pred_pa = point_adjust_np(pred_raw, labels)

    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_dir / "score_combined.txt", test_values, fmt="%.10f")
    np.savetxt(output_dir / "label.txt", labels, fmt="%d")
    np.savetxt(output_dir / "pred_raw.txt", pred_raw, fmt="%d")
    np.savetxt(output_dir / "pred_pa.txt", pred_pa, fmt="%d")

    original = extract_original_metrics(root, dataset)
    comparison = None
    if original is not None and original.get("pa_f1") is not None:
        comparison = {
            "original_metrics_path": original.get("_path"),
            "original_pa_precision": original.get("pa_precision"),
            "original_pa_recall": original.get("pa_recall"),
            "original_pa_f1": original.get("pa_f1"),
            "v4_combined_pa_precision": metrics.get("pa_precision"),
            "v4_combined_pa_recall": metrics.get("pa_recall"),
            "v4_combined_pa_f1": metrics.get("pa_f1"),
            "pa_f1_diff_percentage_points": (
                (metrics.get("pa_f1") - original.get("pa_f1")) * 100.0
                if metrics.get("pa_f1") is not None
                else None
            ),
        }

    result = {
        "dataset": dataset,
        "protocol": "fixed_combined_score",
        "description": (
            "ASCA-AD / V4 uses fixed combined score. No best-over-score-mode "
            "selection is performed. Threshold follows PPLAD-style percentile "
            "over train+test combined scores."
        ),
        "runner_script": str(runner_path),
        "checkpoint": str(checkpoint),
        "score_mode": mode,
        "anormly_ratio": float(ratio),
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
        "files": {
            "score_combined": str(output_dir / "score_combined.txt"),
            "label": str(output_dir / "label.txt"),
            "pred_raw": str(output_dir / "pred_raw.txt"),
            "pred_pa": str(output_dir / "pred_pa.txt"),
        },
    }

    (output_dir / "combined_fixed_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# {dataset}: ASCA-AD / V4 fixed combined result",
        "",
        "## Protocol",
        "",
        "- score mode: fixed `combined`",
        f"- anormly_ratio: `{ratio}`",
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
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[{dataset}] PA-Precision : {metrics.get('pa_precision')}")
    print(f"[{dataset}] PA-Recall    : {metrics.get('pa_recall')}")
    print(f"[{dataset}] PA-F1        : {metrics.get('pa_f1')}")
    if comparison is not None:
        print(f"[{dataset}] Original PA-F1: {comparison['original_pa_f1']}")
        print(f"[{dataset}] PA-F1 diff   : {comparison['pa_f1_diff_percentage_points']:.4f} pp")
    print(f"[{dataset}] saved         : {output_dir / 'summary.md'}")

    return result


def write_global_outputs(output_root: Path, results: List[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "all_combined_fixed_metrics.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_root / "all_combined_fixed_metrics.csv"
    fields = [
        "dataset",
        "score_mode",
        "anormly_ratio",
        "threshold",
        "test_anomaly_ratio_percent",
        "raw_precision",
        "raw_recall",
        "raw_f1",
        "pa_precision",
        "pa_recall",
        "pa_f1",
        "original_pa_precision",
        "original_pa_recall",
        "original_pa_f1",
        "pa_f1_diff_percentage_points",
        "checkpoint",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            metrics = item["metrics"]
            comp = item.get("comparison_with_original_if_available") or {}
            writer.writerow({
                "dataset": item["dataset"],
                "score_mode": item["score_mode"],
                "anormly_ratio": item["anormly_ratio"],
                "threshold": item["threshold"],
                "test_anomaly_ratio_percent": item["counts"]["test_anomaly_ratio_percent"],
                "raw_precision": metrics.get("raw_precision"),
                "raw_recall": metrics.get("raw_recall"),
                "raw_f1": metrics.get("raw_f1"),
                "pa_precision": metrics.get("pa_precision"),
                "pa_recall": metrics.get("pa_recall"),
                "pa_f1": metrics.get("pa_f1"),
                "original_pa_precision": comp.get("original_pa_precision"),
                "original_pa_recall": comp.get("original_pa_recall"),
                "original_pa_f1": comp.get("original_pa_f1"),
                "pa_f1_diff_percentage_points": comp.get("pa_f1_diff_percentage_points"),
                "checkpoint": item["checkpoint"],
            })

    md_path = output_root / "summary.md"
    lines = [
        "# All datasets: ASCA-AD / V4 fixed combined results",
        "",
        "| Dataset | Ratio | Original PA-F1 | ASCA-AD fixed combined PA-F1 | Diff (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results:
        comp = item.get("comparison_with_original_if_available") or {}
        original_f1 = comp.get("original_pa_f1")
        diff = comp.get("pa_f1_diff_percentage_points")
        lines.append(
            f"| {item['dataset']} | {item['anormly_ratio']:.2f} | "
            f"{original_f1:.10f}" if isinstance(original_f1, (float, int)) else f"| {item['dataset']} | {item['anormly_ratio']:.2f} | NA"
        )

    # Rebuild table rows explicitly to avoid formatting ambiguity.
    lines = [
        "# All datasets: ASCA-AD / V4 fixed combined results",
        "",
        "| Dataset | Ratio | Original PA-F1 | ASCA-AD fixed combined PA-F1 | Diff (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results:
        comp = item.get("comparison_with_original_if_available") or {}
        metrics = item["metrics"]
        original_f1 = comp.get("original_pa_f1")
        diff = comp.get("pa_f1_diff_percentage_points")
        original_txt = f"{original_f1:.10f}" if isinstance(original_f1, (float, int)) else "NA"
        diff_txt = f"{diff:.4f}" if isinstance(diff, (float, int)) else "NA"
        lines.append(
            f"| {item['dataset']} | {item['anormly_ratio']:.2f} | "
            f"{original_txt} | {metrics.get('pa_f1', float('nan')):.10f} | {diff_txt} |"
        )

    lines.extend([
        "",
        "Protocol: fixed `combined` score; no best-over-score-mode selection.",
        "Threshold: percentile over train + test combined scores, using the listed `anormly_ratio`.",
        "",
        f"CSV: `{csv_path}`",
        f"JSON: `{json_path}`",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nGlobal summary files:")
    print(f"  {md_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/c/Users/DING/Desktop/Experiment/CODE")
    parser.add_argument("--datasets", default="all", help="all or comma-separated list, e.g. PUMP,SKAB")
    parser.add_argument("--ratio-overrides", default="", help="Example: MSL=0.83,SKAB=0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v4-epochs", type=int, default=10)
    parser.add_argument("--output-dir", default="results/ALL_COMBINED_FIXED")
    parser.add_argument("--train-if-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    os.chdir(root)

    datasets = parse_datasets(args.datasets)
    ratio_overrides = parse_ratio_overrides(args.ratio_overrides)

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    for dataset in datasets:
        ratio = ratio_overrides.get(dataset, float(DATASET_META[dataset]["default_ratio"]))
        try:
            result = evaluate_dataset(
                root=root,
                dataset=dataset,
                ratio=ratio,
                output_root=output_root,
                seed=args.seed,
                v4_epochs=args.v4_epochs,
                train_if_missing=args.train_if_missing,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n[{dataset}] FAILED: {exc}", file=sys.stderr)
            failures.append({"dataset": dataset, "error": str(exc)})

    write_global_outputs(output_root, results)

    if failures:
        fail_path = output_root / "failures.json"
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSome datasets failed. See: {fail_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
