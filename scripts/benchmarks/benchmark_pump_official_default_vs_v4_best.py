#!/usr/bin/env python3
from __future__ import annotations

"""
PUMP 上 Original PPLAD 与 ASCA-AD / V4 的统一轻量化基准。

公平性
------
- 轻量化测试统一使用 win_size=60、batch_size=128；
- Original 的 local/global/d_model 优先从 original_official_metrics.json
  中读取，确保与训练配置一致；
- V4 使用 ASCA-AD 原始 V4 结构；
- 两个模型使用相同测试数据、设备、预热次数和重复次数。
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

def load_original_model_hparams(root: Path) -> Dict[str, object]:
    """
    优先读取训练阶段保存的 Original 配置；如果没有，则回退到
    官方 main.py 默认参数。
    """
    path = (
        root
        / "results"
        / "PUMP_OFFICIAL_DEFAULT_VS_V4_BEST"
        / "original_official_metrics.json"
    )
    if not path.exists():
        return {"local_size": [3], "global_size": [20], "d_model": 128}

    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    return {
        "local_size": config.get("local_size", [3]),
        "global_size": config.get("global_size", [20]),
        "d_model": int(config.get("d_model", 128)),
    }



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
        baseline_root: Path,
        local_size: int = 7,
        global_size: int = 11,
        d_model: int = 128,
    ) -> None:
        super().__init__()

        baseline_root = baseline_root.resolve()
        pplad_file = baseline_root / "model" / "PPLAD.py"
        if not pplad_file.exists():
            raise FileNotFoundError(
                f"找不到官方 PPLAD 模型文件：{pplad_file}"
            )

        # Original PPLAD 位于 BaselineModels/PPLAD-main，而 V4 位于项目根目录。
        # 每个模型在独立子进程中运行，因此把官方仓库放到 sys.path 首位不会污染 V4。
        baseline_str = str(baseline_root)
        if baseline_str in sys.path:
            sys.path.remove(baseline_str)
        sys.path.insert(0, baseline_str)

        # 避免此前从项目根目录导入了同名 model 包。
        for name in list(sys.modules):
            if name == "model" or name.startswith("model."):
                del sys.modules[name]

        try:
            from model.PPLAD import PPLAD
        except ImportError as exc:
            raise RuntimeError(
                f"无法从官方仓库导入 model.PPLAD：{pplad_file}"
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


class V4BestCore(nn.Module):
    def __init__(self, score_mode: str = "total") -> None:
        super().__init__()

        baseline_root = Path.cwd() / "BaselineModels" / "PPLAD-main"
        pplad_file = baseline_root / "model" / "PPLAD.py"
        if not pplad_file.exists():
            raise FileNotFoundError(f"找不到官方 PPLAD：{pplad_file}")

        baseline_str = str(baseline_root)
        if baseline_str in sys.path:
            sys.path.remove(baseline_str)
        sys.path.insert(0, baseline_str)
        for name in list(sys.modules):
            if name == "model" or name.startswith("model."):
                del sys.modules[name]
        __import__("model.RevIN")
        __import__("model.PPLAD")

        if baseline_str in sys.path:
            sys.path.remove(baseline_str)

        project_root = Path.cwd().resolve()
        project_str = str(project_root)
        if project_str in sys.path:
            sys.path.remove(project_str)
        sys.path.insert(0, project_str)

        cached_main = sys.modules.get("main")
        if cached_main is not None:
            cached_file = Path(getattr(cached_main, "__file__", "")).resolve()
            expected_file = (project_root / "main.py").resolve()
            if cached_file != expected_file:
                del sys.modules["main"]

        try:
            from main import AdaptiveSparseAnchorCompetitiveModelV4
        except ImportError as exc:
            raise RuntimeError(
                "无法从项目根目录 main.py 导入 "
                "AdaptiveSparseAnchorCompetitiveModelV4。"
            ) from exc

        if score_mode not in {"gap", "total", "combined"}:
            raise ValueError(f"未知 V4 score mode：{score_mode}")
        self.score_mode = score_mode

        self.model = AdaptiveSparseAnchorCompetitiveModelV4(
            local_candidate_lags=[1, 2, 3, 4, 5, 6, 7, 8],
            global_candidate_lags=[12, 16, 20, 24, 28, 32, 40, 48],
            local_topk=2,
            global_topk=4,
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
        mapping = {
            "gap": details["score_gap"],
            "total": details["score_total"],
            "combined": details["score_combined"],
        }
        return mapping[self.score_mode]


def prepare_input(
    model_name: str,
    batch_cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch = batch_cpu.float().to(device, non_blocking=True)
    if model_name == "original":
        # 与官方 RevIN 的 norm 等价；affine 初始值为 1/0。
        return instance_normalize(batch)
    if model_name == "v4_best":
        return instance_normalize(batch)
    raise ValueError(model_name)


def load_v4_checkpoint(model: V4BestCore, checkpoint: Path, device: torch.device) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"找不到 V4 best checkpoint：{checkpoint}\n"
            "请先运行 run_pump_official_default_vs_v4_best.py --model v4。"
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
    root: Path | None = None,
) -> nn.Module:
    if model_name == "original":
        hparams = load_original_model_hparams(root or Path.cwd())
        local_size = hparams.get("local_size", [3])
        global_size = hparams.get("global_size", [20])
        if isinstance(local_size, list):
            local_size_value = int(local_size[0])
        else:
            local_size_value = int(local_size)
        if isinstance(global_size, list):
            global_size_value = int(global_size[0])
        else:
            global_size_value = int(global_size)

        model: nn.Module = OriginalPPLADCore(
            batch_size=batch_size,
            win_size=win_size,
            channels=channels,
            baseline_root=Path.cwd() / "BaselineModels" / "PPLAD-main",
            local_size=local_size_value,
            global_size=global_size_value,
            d_model=int(hparams.get("d_model", 128)),
        )
    elif model_name == "v4_best":
        result_path = (
            Path.cwd()
            / "results"
            / "PUMP_OFFICIAL_DEFAULT_VS_V4_BEST"
            / "v4_best_run.json"
        )
        score_mode = "total"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            score_mode = str(payload.get("best", {}).get("score_mode", "total"))

        v4 = V4BestCore(score_mode=score_mode)
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
        str(dataset_root / "PUMP"),
        batch_size=batch_size,
        win_size=win_size,
        mode="thre",
        dataset="PUMP",
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


def parse_detection_log(model_name: str, root: Path) -> Dict[str, Any]:
    """读取训练程序保存的正式检测结果，而不是依赖终端日志正则匹配。"""
    result: Dict[str, Any] = {
        "detection_result_path": None,
        "pa_accuracy": None,
        "pa_precision": None,
        "pa_recall": None,
        "pa_f1": None,
        "training_seconds": None,
        "best_score_mode": None,
        "best_anormly_ratio": None,
        "best_threshold": None,
        "selection_protocol": None,
    }

    if model_name == "original":
        path = root / "results/PUMP_OFFICIAL_DEFAULT_VS_V4_BEST/original_official_metrics.json"
        if not path.exists():
            return result
        data = json.loads(path.read_text(encoding="utf-8"))
        result.update(
            {
                "detection_result_path": str(path),
                "pa_accuracy": data.get("pa_accuracy"),
                "pa_precision": data.get("pa_precision"),
                "pa_recall": data.get("pa_recall"),
                "pa_f1": data.get("pa_f1"),
                "training_seconds": data.get("training_seconds"),
                "selection_protocol": data.get("protocol"),
            }
        )
        return result

    path = root / "results/PUMP_OFFICIAL_DEFAULT_VS_V4_BEST/v4_best_run.json"
    if not path.exists():
        return result
    data = json.loads(path.read_text(encoding="utf-8"))
    best = data.get("best", {})
    result.update(
        {
            "detection_result_path": str(path),
            "pa_accuracy": best.get("pa_accuracy"),
            "pa_precision": best.get("pa_precision"),
            "pa_recall": best.get("pa_recall"),
            "pa_f1": best.get("pa_f1"),
            "training_seconds": data.get("training_seconds"),
            "best_score_mode": best.get("score_mode"),
            "best_anormly_ratio": best.get("anormly_ratio"),
            "best_threshold": best.get("threshold"),
            "selection_protocol": data.get("selection_protocol"),
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
    train_file = dataset_root / "PUMP/PUMP_train.npy"
    test_file = dataset_root / "PUMP/PUMP_test.npy"
    label_file = dataset_root / "PUMP/PUMP_test_label.npy"
    if not train_file.exists() or not test_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            "找不到 dataset/PUMP/PUMP_train.npy、PUMP_test.npy 或 PUMP_test_label.npy。"
        )

    channels = int(np.load(train_file, mmap_mode="r").shape[1])

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
        root=root,
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
        raise RuntimeError("PUMP 测试集没有可用 batch。") from exc

    results: Dict[str, Any] = {
        "dataset": "PUMP",
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

    print(f"========== {args.model} / PUMP ==========")
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

    print("Benchmark: batch=128 批处理时延")
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



def format_reduction(value: float) -> str:
    """减少率为正时显示 -x%；为负时表示实际增加，显示 +x%。"""
    return f"-{value:.2f}%" if value >= 0 else f"+{-value:.2f}%"


def format_signed_pct(value: float) -> str:
    return f"{value:+.2f}%"


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    out = root / args.output_dir
    original_path = out / "original.json"
    v4_path = out / "v4_best.json"
    if not original_path.exists() or not v4_path.exists():
        raise FileNotFoundError("缺少 original.json 或 v4_best.json。")

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
        "dataset": "PUMP",
        "original": original,
        "v4_best": v4,
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
        "# PUMP：Original PPLAD（官方配置）vs V4（各自最优结果）",
        "",
        "## 检测精度",
        "",
        "| 模型 | PA-Precision | PA-Recall | PA-F1 |",
        "|---|---:|---:|---:|",
        f"| Original PPLAD | {original.get('pa_precision')} | {original.get('pa_recall')} | {original.get('pa_f1')} |",
        f"| V4 best | {v4.get('pa_precision')} | {v4.get('pa_recall')} | {v4.get('pa_f1')} |",
        "",
        f"V4 最优设置：score mode={v4.get('best_score_mode')}，"
        f"anormly_ratio={v4.get('best_anormly_ratio')}，"
        f"threshold={v4.get('best_threshold')}。",
        "",
        "## 轻量化指标",
        "",
        "| 指标 | Original PPLAD | V4 best | V4 相对变化 |",
        "|---|---:|---:|---:|",
        f"| 训练总时间 | {original.get('training_seconds')} s | {v4.get('training_seconds')} s | 仅记录，各模型训练轮数不同 |",
        f"| 参数量 | {original['trainable_params']:,} | {v4['trainable_params']:,} | {format_reduction(comparison['parameter_reduction_pct'])} |",
        f"| State Dict 磁盘 | {original['state_dict_disk_bytes']/1024:.2f} KiB | {v4['state_dict_disk_bytes']/1024:.2f} KiB | {format_reduction(comparison['state_dict_disk_reduction_pct'])} |",
        f"| 在线 batch=1 平均时延 | {original['online_batch1_latency_ms_mean']:.4f} ms | {v4['online_batch1_latency_ms_mean']:.4f} ms | {format_reduction(comparison['online_batch1_latency_reduction_pct'])} |",
        f"| batch=128 平均时延 | {original['single_batch_latency_ms_mean']:.4f} ms | {v4['single_batch_latency_ms_mean']:.4f} ms | {format_reduction(comparison['single_batch_latency_reduction_pct'])} |",
        f"| 完整测试集时间 | {original['full_test_seconds_mean']:.6f} s | {v4['full_test_seconds_mean']:.6f} s | {format_reduction(comparison['full_test_time_reduction_pct'])} |",
        f"| 吞吐率 | {original['full_test_points_per_second']:.2f} points/s | {v4['full_test_points_per_second']:.2f} points/s | {format_signed_pct(comparison['throughput_increase_pct'])} |",
        f"| GPU 峰值显存 | {original['full_test_gpu_peak_allocated_mib']:.2f} MiB | {v4['full_test_gpu_peak_allocated_mib']:.2f} MiB | {format_reduction(comparison['full_test_gpu_peak_reduction_pct'])} |",
        f"| GPU 增量显存 | {original['full_test_gpu_incremental_allocated_mib']:.2f} MiB | {v4['full_test_gpu_incremental_allocated_mib']:.2f} MiB | {format_reduction(comparison['full_test_gpu_incremental_reduction_pct'])} |",
        f"| CPU 增量 RSS | {original['full_test_cpu_incremental_rss_mib']:.2f} MiB | {v4['full_test_cpu_incremental_rss_mib']:.2f} MiB | {format_reduction(comparison['full_test_cpu_incremental_reduction_pct'])} |",
        "",
        "说明：检测精度部分，Original 使用PUMP 配置，V4 使用自身最佳配置及预定义阈值网格中的最高 PA-F1。轻量化部分为受控协议：两种模型统一使用 win_size=60、batch_size=128、相同硬件和相同测试流程。CPU RSS 容易受 Python 分配器影响，应以多次独立进程均值为正式结果。",
    ]
    summary = "\n".join(summary_lines)
    (out / "summary.md").write_text(summary, encoding="utf-8")

    flat_rows = []
    for model_name, row in (("original", original), ("v4_best", v4)):
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

    for model_name in ("original", "v4_best"):
        command = common + ["--model", model_name]
        print("\n运行：", " ".join(command))
        subprocess.run(command, check=True)

    aggregate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", "original", "v4_best", "aggregate"],
        default="all",
    )
    parser.add_argument(
        "--root",
        default="/mnt/c/Users/DING/Desktop/Experiment/CODE",
    )
    parser.add_argument(
        "--output-dir",
        default="results/PUMP_OFFICIAL_DEFAULT_VS_V4_BEST/BENCHMARK",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--win-size", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--full-test-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--v4-checkpoint",
        default=(
            "checkpoints/PUMP_OFFICIAL_DEFAULT_VS_V4_BEST/V4/"
            "PUMP_adaptive_anchor_v4_l1-2-3-4-5-6-7-8_"
            "g12-16-20-24-28-32-40-48_kl2_kg4.pt"
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
