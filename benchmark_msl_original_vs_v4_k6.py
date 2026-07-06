#!/usr/bin/env python3
from __future__ import annotations

"""
MSL 上 Original PPLAD 与 V4-official_k6 的统一轻量化基准。

测量项目
--------
1. 可训练参数量、参数字节数、序列化 state_dict 大小；
2. batch=1 在线前向时延；
3. batch=256 批处理前向时延；
4. 完整测试集端到端时间与吞吐率；
5. GPU 峰值显存与增量显存；
6. CPU RSS 峰值与增量；
7. checkpoint/state_dict 磁盘大小；
8. 从已有日志提取 PA-Precision / PA-Recall / PA-F1。

公平性
------
- 数据集：MSL，55 通道；
- win_size=90，batch_size=256；
- Original：local_size=7，global_size=30，d_model=128；
- V4：local lags=[1,2,3], Top-k=2；
      global lags=[4..18], Top-k=6；
- batch latency 只计模型前向，不计 DataLoader、CPU→GPU 和归一化；
- full-test 计 DataLoader 迭代、CPU→GPU、归一化和模型前向；
- Original 官方代码不保存训练 checkpoint，因此脚本保存其 state_dict
  作为可比磁盘文件；权重数值不影响参数量、时延和显存。
"""

import argparse
import csv
import gc
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn

MIB = 1024 ** 2


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class RSSMonitor:
    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "RSSMonitor":
        current = self.process.memory_info().rss
        self.baseline_bytes = current
        self.peak_bytes = current

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    rss = self.process.memory_info().rss
                    if rss > self.peak_bytes:
                        self.peak_bytes = rss
                except psutil.Error:
                    pass
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            rss = self.process.memory_info().rss
            self.peak_bytes = max(self.peak_bytes, rss)
        except psutil.Error:
            pass


def tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def serialized_state_dict_bytes(model: nn.Module) -> int:
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return int(stream.tell())


def percentile_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_latency_ms_mean": float(arr.mean()),
        f"{prefix}_latency_ms_std": float(arr.std()),
        f"{prefix}_latency_ms_p50": float(np.percentile(arr, 50)),
        f"{prefix}_latency_ms_p95": float(np.percentile(arr, 95)),
        f"{prefix}_latency_ms_p99": float(np.percentile(arr, 99)),
    }


def reset_gpu_peak(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    gc.collect()
    torch.cuda.empty_cache()
    sync(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    return baseline_allocated, baseline_reserved


def gpu_peak_result(
    device: torch.device,
    baseline_allocated: int,
    baseline_reserved: int,
) -> Dict[str, float]:
    if device.type != "cuda":
        return {
            "gpu_baseline_allocated_mib": 0.0,
            "gpu_peak_allocated_mib": 0.0,
            "gpu_incremental_allocated_mib": 0.0,
            "gpu_baseline_reserved_mib": 0.0,
            "gpu_peak_reserved_mib": 0.0,
            "gpu_incremental_reserved_mib": 0.0,
        }
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "gpu_baseline_allocated_mib": baseline_allocated / MIB,
        "gpu_peak_allocated_mib": peak_allocated / MIB,
        "gpu_incremental_allocated_mib": max(0, peak_allocated - baseline_allocated) / MIB,
        "gpu_baseline_reserved_mib": baseline_reserved / MIB,
        "gpu_peak_reserved_mib": peak_reserved / MIB,
        "gpu_incremental_reserved_mib": max(0, peak_reserved - baseline_reserved) / MIB,
    }


def instance_normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True).detach()
    var = x.var(dim=1, keepdim=True, unbiased=False).detach()
    return (x - mean) / torch.sqrt(var + eps)


class OriginalPPLADCore(nn.Module):
    """复现官方 PPLAD 的关系构造和模型前向，不包含 RevIN/数据搬运。"""

    def __init__(
        self,
        batch_size: int,
        win_size: int,
        channels: int,
        local_size: int = 7,
        global_size: int = 30,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        try:
            from model.PPLAD import PPLAD
        except ImportError as exc:
            raise RuntimeError(
                "无法导入 model.PPLAD。请把脚本放在包含 model/、main.py 的项目根目录。"
            ) from exc

        self.win_size = int(win_size)
        self.channels = int(channels)
        self.local_size = int(local_size)
        self.global_size = int(global_size)
        self.model = PPLAD(
            batch_size=batch_size,
            win_size=win_size,
            enc_in=channels,
            c_out=channels,
            d_model=d_model,
            local_size=[local_size],
            global_size=[global_size],
            channel=channels,
        )

    def _build_relations(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        batch, length, channels = x.shape
        total = self.local_size + self.global_size
        front = total // 2
        back = total - front
        boundary = length - back
        windows: List[torch.Tensor] = []

        for index in range(length):
            if index < front:
                prefix = x[:, 0, :].unsqueeze(1).repeat(1, front - index, 1)
                current = torch.cat((prefix, x[:, :index, :]), dim=1)
                current = torch.cat((current, x[:, index:index + back, :]), dim=1)
            elif index > boundary:
                suffix = x[:, length - 1, :].unsqueeze(1).repeat(
                    1, back + index - length, 1
                )
                current = torch.cat((x[:, index - front:length, :], suffix), dim=1)
            else:
                current = x[:, index - front:index + back, :]
            windows.append(current)

        relation = (
            torch.cat(windows, dim=0)
            .reshape(length, batch, total, channels)
            .permute(1, 0, 3, 2)
        )
        relation = torch.sum(
            (relation - x.unsqueeze(-1).expand(-1, -1, -1, total)).square(),
            dim=2,
        )
        relation = torch.softmax(relation, dim=-1)

        site = total // 2
        local_front = self.local_size // 2
        local_back = self.local_size - local_front
        local = relation[:, :, site - local_front:site + local_back]
        global_ = torch.cat(
            (
                relation[:, :, :site - local_front],
                relation[:, :, site + local_back:],
            ),
            dim=-1,
        )
        return [local], [global_], relation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local, global_, full = self._build_relations(x)
        series, prior, _, _, _, _, _, _ = self.model(
            x, local, global_, "test", 0, full
        )
        local_error = torch.sum((series[0] - local[0]).square(), dim=-1)
        global_error = torch.sum((prior[0] - global_[0]).square(), dim=-1)
        return local_error + global_error


class V4K6Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from main import AdaptiveSparseAnchorCompetitiveModelV4
        except ImportError as exc:
            raise RuntimeError(
                "无法从 main.py 导入 AdaptiveSparseAnchorCompetitiveModelV4。"
            ) from exc

        self.model = AdaptiveSparseAnchorCompetitiveModelV4(
            local_candidate_lags=[1, 2, 3],
            global_candidate_lags=list(range(4, 19)),
            local_topk=2,
            global_topk=6,
            selector_hidden=8,
            fitter_hidden=8,
            selector_temperature=0.5,
            similarity_tau=1.0,
            sigma_min=0.03,
            sigma_max=1.50,
            gap_weight=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, details = self.model(x)
        return details["score_total"]


def prepare_input(
    model_name: str,
    batch_cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch = batch_cpu.float().to(device, non_blocking=True)
    if model_name == "original":
        # 与官方 RevIN 的 norm 等价；affine 初始值为 1/0。
        return instance_normalize(batch)
    if model_name == "v4_k6":
        return instance_normalize(batch)
    raise ValueError(model_name)


def load_v4_checkpoint(model: V4K6Core, checkpoint: Path, device: torch.device) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"找不到 V4 official_k6 checkpoint：{checkpoint}\n"
            "请先运行锚点实验中的 official_k6 配置。"
        )
    payload = torch.load(checkpoint, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.model.load_state_dict(state)


def build_model(
    model_name: str,
    batch_size: int,
    win_size: int,
    channels: int,
    device: torch.device,
    v4_checkpoint: Path,
) -> nn.Module:
    if model_name == "original":
        model: nn.Module = OriginalPPLADCore(
            batch_size=batch_size,
            win_size=win_size,
            channels=channels,
            local_size=7,
            global_size=30,
            d_model=128,
        )
    elif model_name == "v4_k6":
        v4 = V4K6Core()
        load_v4_checkpoint(v4, v4_checkpoint, device)
        model = v4
    else:
        raise ValueError(model_name)
    return model.to(device).eval()


def get_test_loader(
    dataset_root: Path,
    batch_size: int,
    win_size: int,
):
    try:
        from data_factory.data_loader import get_loader_segment
    except ImportError as exc:
        raise RuntimeError("无法导入 data_factory.data_loader.get_loader_segment。") from exc

    return get_loader_segment(
        137,
        str(dataset_root / "MSL"),
        batch_size=batch_size,
        win_size=win_size,
        mode="thre",
        dataset="MSL",
    )


@torch.inference_mode()
def benchmark_batch(
    model_name: str,
    model: nn.Module,
    batch_cpu: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    prefix: str,
) -> Dict[str, Any]:
    prepared = prepare_input(model_name, batch_cpu, device)

    for _ in range(max(1, warmup)):
        _ = model(prepared)
    sync(device)

    baseline_allocated, baseline_reserved = reset_gpu_peak(device)
    timings: List[float] = []

    with RSSMonitor() as rss:
        for _ in range(repeats):
            sync(device)
            start = time.perf_counter()
            output = model(prepared)
            # 不调用 .item()，避免额外同步；显式 synchronize 已保证准确计时。
            _ = output
            sync(device)
            timings.append((time.perf_counter() - start) * 1000.0)

    result: Dict[str, Any] = {
        f"{prefix}_batch_size_actual": int(batch_cpu.shape[0]),
        f"{prefix}_cpu_baseline_rss_mib": rss.baseline_bytes / MIB,
        f"{prefix}_cpu_peak_rss_mib": rss.peak_bytes / MIB,
        f"{prefix}_cpu_incremental_rss_mib": max(
            0, rss.peak_bytes - rss.baseline_bytes
        ) / MIB,
    }
    result.update(percentile_stats(timings, prefix))
    result.update(
        {
            f"{prefix}_{key}": value
            for key, value in gpu_peak_result(
                device, baseline_allocated, baseline_reserved
            ).items()
        }
    )
    return result


@torch.inference_mode()
def benchmark_full_test(
    model_name: str,
    model: nn.Module,
    dataset_root: Path,
    device: torch.device,
    batch_size: int,
    win_size: int,
    repeats: int,
) -> Dict[str, Any]:
    elapsed_values: List[float] = []
    windows_values: List[int] = []
    batches_values: List[int] = []
    cpu_baselines: List[int] = []
    cpu_peaks: List[int] = []
    gpu_results: List[Dict[str, float]] = []

    for repeat_index in range(repeats):
        loader = get_test_loader(dataset_root, batch_size, win_size)
        baseline_allocated, baseline_reserved = reset_gpu_peak(device)
        batches = 0
        windows = 0

        with RSSMonitor() as rss:
            sync(device)
            start = time.perf_counter()
            for input_data, _ in loader:
                prepared = prepare_input(model_name, input_data, device)
                _ = model(prepared)
                batches += 1
                windows += int(input_data.shape[0])
            sync(device)
            elapsed = time.perf_counter() - start

        elapsed_values.append(float(elapsed))
        windows_values.append(windows)
        batches_values.append(batches)
        cpu_baselines.append(rss.baseline_bytes)
        cpu_peaks.append(rss.peak_bytes)
        gpu_results.append(
            gpu_peak_result(device, baseline_allocated, baseline_reserved)
        )
        print(
            f"  full-test repeat {repeat_index + 1}/{repeats}: "
            f"{elapsed:.6f}s, windows={windows}"
        )

    times = np.asarray(elapsed_values, dtype=np.float64)
    windows = int(statistics.median(windows_values))
    batches = int(statistics.median(batches_values))
    mean_seconds = float(times.mean())

    result: Dict[str, Any] = {
        "full_test_repeats": repeats,
        "full_test_batches": batches,
        "full_test_windows": windows,
        "full_test_points": windows * win_size,
        "full_test_seconds_mean": mean_seconds,
        "full_test_seconds_std": float(times.std()),
        "full_test_seconds_p50": float(np.percentile(times, 50)),
        "full_test_seconds_p95": float(np.percentile(times, 95)),
        "full_test_windows_per_second": windows / max(mean_seconds, 1e-12),
        "full_test_points_per_second": (windows * win_size) / max(mean_seconds, 1e-12),
        "full_test_cpu_baseline_rss_mib": min(cpu_baselines) / MIB,
        "full_test_cpu_peak_rss_mib": max(cpu_peaks) / MIB,
        "full_test_cpu_incremental_rss_mib": max(
            0, max(cpu_peaks) - min(cpu_baselines)
        ) / MIB,
    }

    for key in gpu_results[0]:
        if "baseline" in key:
            result[f"full_test_{key}"] = min(item[key] for item in gpu_results)
        else:
            result[f"full_test_{key}"] = max(item[key] for item in gpu_results)
    return result


def parse_detection_log(model_name: str, root: Path) -> Dict[str, float | None]:
    if model_name == "original":
        path = root / "logs/MSL_COMPARE/pplad_msl_official.log"
    else:
        path = root / "logs/MSL_V4_ANCHORS/official_k6.log"

    result: Dict[str, float | None] = {
        "detection_log": str(path),
        "pa_accuracy": None,
        "pa_precision": None,
        "pa_recall": None,
        "pa_f1": None,
    }
    if not path.exists():
        return result

    text = path.read_text(encoding="utf-8", errors="ignore")
    pa_lines = re.findall(
        r"pa_accuracy\s*:\s*([0-9.]+).*?"
        r"pa_precision\s*:\s*([0-9.]+).*?"
        r"pa_recall\s*:\s*([0-9.]+).*?"
        r"pa_f_score\s*:\s*([0-9.]+)",
        text,
        flags=re.S,
    )
    if pa_lines:
        acc, precision, recall, f1 = map(float, pa_lines[-1])
    else:
        classic = re.findall(
            r"Accuracy\s*:\s*([0-9.]+),\s*Precision\s*:\s*([0-9.]+),\s*"
            r"Recall\s*:\s*([0-9.]+),\s*F-score\s*:\s*([0-9.]+)",
            text,
        )
        if not classic:
            return result
        acc, precision, recall, f1 = map(float, classic[-1])

    result.update(
        {
            "pa_accuracy": acc,
            "pa_precision": precision,
            "pa_recall": recall,
            "pa_f1": f1,
        }
    )
    return result


def save_json_csv(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    flat = {
        key: value
        for key, value in data.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def benchmark_one(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    dataset_root = root / "dataset"
    train_file = dataset_root / "MSL/MSL_train.npy"
    test_file = dataset_root / "MSL/MSL_test.npy"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("找不到 dataset/MSL/MSL_train.npy 或 MSL_test.npy。")

    channels = int(np.load(train_file, mmap_mode="r").shape[1])
    if channels != 55:
        raise ValueError(f"MSL 应为 55 通道，实际为 {channels}。")

    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境没有可用 GPU。")

    set_seed(args.seed)
    v4_checkpoint = root / args.v4_checkpoint
    model = build_model(
        model_name=args.model,
        batch_size=args.batch_size,
        win_size=args.win_size,
        channels=channels,
        device=device,
        v4_checkpoint=v4_checkpoint,
    )

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 为两种模型都保存同一格式的 state_dict 文件，作为公平的磁盘大小对比。
    state_path = output_dir / f"{args.model}_state_dict.pt"
    torch.save(model.state_dict(), state_path)

    loader = get_test_loader(dataset_root, args.batch_size, args.win_size)
    try:
        first_batch, _ = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("MSL 测试集没有可用 batch。") from exc

    results: Dict[str, Any] = {
        "dataset": "MSL",
        "model": args.model,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "channels": channels,
        "win_size": args.win_size,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "latency_repeats": args.repeats,
        "full_test_repeats": args.full_test_repeats,
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "parameter_tensor_bytes": tensor_bytes(model.parameters()),
        "buffer_tensor_bytes": tensor_bytes(model.buffers()),
        "serialized_state_dict_bytes": serialized_state_dict_bytes(model),
        "state_dict_disk_path": str(state_path),
        "state_dict_disk_bytes": int(state_path.stat().st_size),
    }
    results.update(parse_detection_log(args.model, root))

    print(f"========== {args.model} / MSL ==========")
    print(f"params={results['trainable_params']:,}")
    print("Benchmark: batch=1 在线时延")
    results.update(
        benchmark_batch(
            args.model,
            model,
            first_batch[:1],
            device,
            args.warmup,
            args.repeats,
            prefix="online_batch1",
        )
    )

    print("Benchmark: batch=256 批处理时延")
    results.update(
        benchmark_batch(
            args.model,
            model,
            first_batch,
            device,
            args.warmup,
            args.repeats,
            prefix="single_batch",
        )
    )

    print("Benchmark: 完整测试集")
    results.update(
        benchmark_full_test(
            args.model,
            model,
            dataset_root,
            device,
            args.batch_size,
            args.win_size,
            args.full_test_repeats,
        )
    )

    output_path = output_dir / f"{args.model}.json"
    save_json_csv(output_path, results)
    print("结果：", output_path)


def reduction_pct(new: float, old: float) -> float:
    return 100.0 * (1.0 - new / old) if old else float("nan")


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    out = root / args.output_dir
    original_path = out / "original.json"
    v4_path = out / "v4_k6.json"
    if not original_path.exists() or not v4_path.exists():
        raise FileNotFoundError("缺少 original.json 或 v4_k6.json。")

    original = json.loads(original_path.read_text(encoding="utf-8"))
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))

    keys = [
        ("parameter_reduction_pct", "trainable_params"),
        ("state_dict_disk_reduction_pct", "state_dict_disk_bytes"),
        ("online_batch1_latency_reduction_pct", "online_batch1_latency_ms_mean"),
        ("single_batch_latency_reduction_pct", "single_batch_latency_ms_mean"),
        ("full_test_time_reduction_pct", "full_test_seconds_mean"),
        ("full_test_gpu_peak_reduction_pct", "full_test_gpu_peak_allocated_mib"),
        ("full_test_gpu_incremental_reduction_pct", "full_test_gpu_incremental_allocated_mib"),
        ("full_test_cpu_incremental_reduction_pct", "full_test_cpu_incremental_rss_mib"),
    ]

    comparison: Dict[str, Any] = {
        "dataset": "MSL",
        "original": original,
        "v4_k6": v4,
    }
    for output_key, metric_key in keys:
        comparison[output_key] = reduction_pct(
            float(v4[metric_key]), float(original[metric_key])
        )
    comparison["throughput_increase_pct"] = 100.0 * (
        float(v4["full_test_points_per_second"])
        / float(original["full_test_points_per_second"])
        - 1.0
    )

    if original.get("pa_precision") is not None and v4.get("pa_precision") is not None:
        comparison["precision_delta_pp"] = 100.0 * (
            float(v4["pa_precision"]) - float(original["pa_precision"])
        )
        comparison["recall_delta_pp"] = 100.0 * (
            float(v4["pa_recall"]) - float(original["pa_recall"])
        )
        comparison["f1_delta_pp"] = 100.0 * (
            float(v4["pa_f1"]) - float(original["pa_f1"])
        )

    (out / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "# MSL：Original PPLAD vs V4 official_k6",
        "",
        "## 检测精度",
        "",
        "| 模型 | PA-Precision | PA-Recall | PA-F1 |",
        "|---|---:|---:|---:|",
        f"| Original PPLAD | {original.get('pa_precision')} | {original.get('pa_recall')} | {original.get('pa_f1')} |",
        f"| V4 official_k6 | {v4.get('pa_precision')} | {v4.get('pa_recall')} | {v4.get('pa_f1')} |",
        "",
        "## 轻量化指标",
        "",
        "| 指标 | Original PPLAD | V4 official_k6 | V4 相对变化 |",
        "|---|---:|---:|---:|",
        f"| 参数量 | {original['trainable_params']:,} | {v4['trainable_params']:,} | -{comparison['parameter_reduction_pct']:.2f}% |",
        f"| State Dict 磁盘 | {original['state_dict_disk_bytes']/1024:.2f} KiB | {v4['state_dict_disk_bytes']/1024:.2f} KiB | -{comparison['state_dict_disk_reduction_pct']:.2f}% |",
        f"| 在线 batch=1 平均时延 | {original['online_batch1_latency_ms_mean']:.4f} ms | {v4['online_batch1_latency_ms_mean']:.4f} ms | -{comparison['online_batch1_latency_reduction_pct']:.2f}% |",
        f"| batch=256 平均时延 | {original['single_batch_latency_ms_mean']:.4f} ms | {v4['single_batch_latency_ms_mean']:.4f} ms | -{comparison['single_batch_latency_reduction_pct']:.2f}% |",
        f"| 完整测试集时间 | {original['full_test_seconds_mean']:.6f} s | {v4['full_test_seconds_mean']:.6f} s | -{comparison['full_test_time_reduction_pct']:.2f}% |",
        f"| 吞吐率 | {original['full_test_points_per_second']:.2f} points/s | {v4['full_test_points_per_second']:.2f} points/s | +{comparison['throughput_increase_pct']:.2f}% |",
        f"| GPU 峰值显存 | {original['full_test_gpu_peak_allocated_mib']:.2f} MiB | {v4['full_test_gpu_peak_allocated_mib']:.2f} MiB | -{comparison['full_test_gpu_peak_reduction_pct']:.2f}% |",
        f"| GPU 增量显存 | {original['full_test_gpu_incremental_allocated_mib']:.2f} MiB | {v4['full_test_gpu_incremental_allocated_mib']:.2f} MiB | -{comparison['full_test_gpu_incremental_reduction_pct']:.2f}% |",
        f"| CPU 增量 RSS | {original['full_test_cpu_incremental_rss_mib']:.2f} MiB | {v4['full_test_cpu_incremental_rss_mib']:.2f} MiB | -{comparison['full_test_cpu_incremental_reduction_pct']:.2f}% |",
        "",
        "说明：负的“减少率”意味着 V4 实际更大/更慢；CPU RSS 容易受 Python 分配器影响，应以多次独立进程均值为正式结果。",
    ]
    summary = "\n".join(summary_lines)
    (out / "summary.md").write_text(summary, encoding="utf-8")

    flat_rows = []
    for model_name, row in (("original", original), ("v4_k6", v4)):
        flat_rows.append(
            {
                "model": model_name,
                "pa_precision": row.get("pa_precision"),
                "pa_recall": row.get("pa_recall"),
                "pa_f1": row.get("pa_f1"),
                "trainable_params": row["trainable_params"],
                "state_dict_disk_bytes": row["state_dict_disk_bytes"],
                "online_batch1_latency_ms_mean": row["online_batch1_latency_ms_mean"],
                "online_batch1_latency_ms_p95": row["online_batch1_latency_ms_p95"],
                "single_batch_latency_ms_mean": row["single_batch_latency_ms_mean"],
                "single_batch_latency_ms_p95": row["single_batch_latency_ms_p95"],
                "full_test_seconds_mean": row["full_test_seconds_mean"],
                "full_test_points_per_second": row["full_test_points_per_second"],
                "full_test_gpu_peak_allocated_mib": row["full_test_gpu_peak_allocated_mib"],
                "full_test_gpu_incremental_allocated_mib": row["full_test_gpu_incremental_allocated_mib"],
                "full_test_cpu_incremental_rss_mib": row["full_test_cpu_incremental_rss_mib"],
            }
        )

    with (out / "comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    print("\n" + summary)
    print("\n结果目录：", out)


def run_all(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    common = [
        sys.executable,
        "-u",
        str(script),
        "--root",
        args.root,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--win-size",
        str(args.win_size),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--full-test-repeats",
        str(args.full_test_repeats),
        "--seed",
        str(args.seed),
        "--v4-checkpoint",
        args.v4_checkpoint,
    ]

    for model_name in ("original", "v4_k6"):
        command = common + ["--model", model_name]
        print("\n运行：", " ".join(command))
        subprocess.run(command, check=True)

    aggregate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", "original", "v4_k6", "aggregate"],
        default="all",
    )
    parser.add_argument(
        "--root",
        default="/mnt/c/Users/DING/Desktop/Experiment/CODE",
    )
    parser.add_argument(
        "--output-dir",
        default="results/MSL_LIGHTWEIGHT_COMPARE",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--win-size", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--full-test-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--v4-checkpoint",
        default=(
            "checkpoints/MSL_V4_ANCHORS/official_k6/"
            "MSL_adaptive_anchor_v4_l1-2-3_"
            "g4-5-6-7-8-9-10-11-12-13-14-15-16-17-18_kl2_kg6.pt"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.model == "all":
        run_all(args)
    elif args.model == "aggregate":
        aggregate(args)
    else:
        benchmark_one(args)


if __name__ == "__main__":
    main()
