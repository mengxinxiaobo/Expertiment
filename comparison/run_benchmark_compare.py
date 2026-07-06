#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import flatten_dict, load_config, resolve_from, write_csv, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="批量运行 Original 与 V4 轻量化基准")
    p.add_argument("--config", default="comparison/config.json")
    p.add_argument("--run-root", default=None)
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--skip-threshold", action="store_true")
    args = p.parse_args()

    config_path = Path(args.config).resolve()
    cfg, project_root = load_config(config_path)
    v4_root = resolve_from(project_root, cfg["v4_root"])
    original_root = resolve_from(project_root, cfg["original_root"])
    dataset_root = resolve_from(project_root, cfg["dataset_root"])
    output_parent = resolve_from(project_root, cfg["output_root"])

    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
    else:
        latest = output_parent / "LATEST"
        if not latest.exists():
            raise FileNotFoundError("找不到 comparison_runs/LATEST，请先运行检测实验。")
        run_root = Path(latest.read_text(encoding="utf-8").strip()).resolve()

    requested = [x.upper() for x in args.datasets]
    datasets = requested or [
        name for name, item in cfg["datasets"].items()
        if item.get("enabled", False)
    ]
    bench_cfg = cfg["benchmark"]
    include_threshold = bool(
        bench_cfg.get("include_exact_threshold", True)
        and not args.skip_threshold
    )
    protocols = bench_cfg.get("protocols", ["native", "controlled"])
    python = cfg.get("python", sys.executable)
    script = Path(__file__).resolve().with_name("benchmark_one.py")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if "gpu_index" in cfg:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_index"])

    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        dc = cfg["datasets"][dataset]
        if dc["original"].get("anormly_ratio") is None:
            print(f"[SKIP] {dataset}: 原版配置未核验")
            continue
        data_dir = dataset_root / dc["folder"]
        for protocol in protocols:
            for model in ("original", "v4"):
                mc = dc[model]
                if protocol == "native":
                    win_size = int(mc["win_size"])
                    batch_size = int(mc["batch_size"])
                else:
                    win_size = int(bench_cfg["controlled_win_size"])
                    batch_size = int(bench_cfg["controlled_batch_size"])

                output = (
                    run_root / dataset / model / "benchmark"
                    / f"{protocol}.json"
                )
                output.parent.mkdir(parents=True, exist_ok=True)

                project = original_root if model == "original" else v4_root
                checkpoint = "none"
                if model == "v4":
                    candidates = sorted(
                        (run_root / dataset / "v4" / "checkpoints").glob("*.pt")
                    )
                    if candidates:
                        checkpoint = str(candidates[-1])

                command = [
                    python, str(script),
                    "--model", model,
                    "--project-root", str(project),
                    "--dataset-dir", str(data_dir),
                    "--file-prefix", dc["file_prefix"],
                    "--dataset", dataset,
                    "--protocol", protocol,
                    "--win-size", str(win_size),
                    "--batch-size", str(batch_size),
                    "--anormly-ratio", str(mc["anormly_ratio"]),
                    "--checkpoint", checkpoint,
                    "--warmup", str(bench_cfg.get("warmup", 10)),
                    "--repeats", str(bench_cfg.get("repeats", 100)),
                    "--device", cfg.get("device", "auto"),
                    "--output", str(output),
                ]
                if model == "original":
                    command.extend(
                        [
                            "--local-size", str(mc["local_size"][0]),
                            "--global-size", str(mc["global_size"][0]),
                            "--d-model", str(mc["d_model"]),
                        ]
                    )
                if include_threshold:
                    command.append("--include-threshold")

                command_path = output.with_suffix(".command.txt")
                command_path.write_text(" ".join(command), encoding="utf-8")
                log_path = output.with_suffix(".log")
                print("\n[RUN]", " ".join(command))
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=str(project_root),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, command)
                result = json.loads(output.read_text(encoding="utf-8"))
                rows.append(flatten_dict(result))

    write_json(run_root / "benchmark_metrics.json", rows)
    write_csv(run_root / "benchmark_metrics.csv", rows)
    print("\n轻量化基准完成：", run_root / "benchmark_metrics.csv")


if __name__ == "__main__":
    main()
