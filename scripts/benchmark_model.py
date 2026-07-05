#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import platform
import resource
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# 允许把本文件放在 <项目根目录>/scripts/ 下，并从项目根目录执行。
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from data_factory.data_loader import get_loader_segment  # noqa: E402


MIB = 1024 ** 2

DEFAULT_RATIOS = {
    "SMAP": 2.0,
    "SKAB": 2.0,
    "MSL": 0.75,
    "HAI": 0.98,
    "PSM": 0.5,
    "SMD": 0.5,
    "PUMP": 0.5,
    "WADI": 0.5,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def instance_normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True).detach()
    var = x.var(dim=1, keepdim=True, unbiased=False).detach()
    return (x - mean) / torch.sqrt(var + eps)


def current_rss_bytes() -> int:
    """读取当前进程常驻内存；优先使用 Linux/WSL 的 /proc。"""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value * 1024)


class PeakRSSMonitor:
    """后台采样当前进程 RSS（Resident Set Size，常驻内存）。"""

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.baseline_bytes = current_rss_bytes()
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, current_rss_bytes())
            self._stop.wait(self.interval)
        self.peak_bytes = max(self.peak_bytes, current_rss_bytes())

    def __enter__(self) -> "PeakRSSMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def baseline_mib(self) -> float:
        return self.baseline_bytes / MIB

    @property
    def peak_mib(self) -> float:
        return self.peak_bytes / MIB

    @property
    def incremental_mib(self) -> float:
        return max(0.0, (self.peak_bytes - self.baseline_bytes) / MIB)


def serialized_state_dict_bytes(module: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return int(buffer.tell())


def parameter_stats(module: nn.Module) -> Dict[str, int]:
    parameters = list(module.parameters())
    buffers = list(module.buffers())
    return {
        "trainable_params": int(sum(p.numel() for p in parameters if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in parameters)),
        "parameter_tensor_bytes": int(sum(p.numel() * p.element_size() for p in parameters)),
        "buffer_elements": int(sum(b.numel() for b in buffers)),
        "buffer_tensor_bytes": int(sum(b.numel() * b.element_size() for b in buffers)),
        "serialized_state_dict_bytes": serialized_state_dict_bytes(module),
    }


def infer_channels(dataset_dir: Path) -> Tuple[int, Path]:
    files = sorted(dataset_dir.glob("*_train.npy"))
    if not files:
        raise FileNotFoundError(f"未找到训练数据：{dataset_dir}/*_train.npy")
    train_file = files[0]
    data = np.load(train_file, mmap_mode="r", allow_pickle=False)
    if data.ndim != 2:
        raise ValueError(f"{train_file} 应为二维数组，实际 shape={data.shape}")
    return int(data.shape[1]), train_file


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU。")
        return torch.device("cuda:0")
    return torch.device("cpu")


def find_v4_checkpoint(dataset: str) -> Optional[Path]:
    folder = PROJECT_ROOT / "checkpoints" / dataset
    candidates = sorted(
        folder.glob("*adaptive_anchor_v4*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_checkpoint_payload(path: Optional[Path]) -> Optional[Any]:
    if path is None:
        return None
    return torch.load(path, map_location="cpu")


class V4BenchmarkModel(nn.Module):
    """V4完整单窗口前向：实例归一化 + 稀疏锚点模型 + total score。"""

    def __init__(self, checkpoint_payload: Optional[Any] = None) -> None:
        super().__init__()
        from main import AdaptiveSparseAnchorCompetitiveModelV4

        config: Dict[str, Any] = {}
        if isinstance(checkpoint_payload, dict):
            config = checkpoint_payload.get("config", {}) or {}

        self.core = AdaptiveSparseAnchorCompetitiveModelV4(
            local_candidate_lags=config.get(
                "local_candidate_lags", [1, 2, 3, 4, 5, 6, 7, 8]
            ),
            global_candidate_lags=config.get(
                "global_candidate_lags", [12, 16, 20, 24, 28, 32, 40, 48]
            ),
            local_topk=int(config.get("local_topk", 2)),
            global_topk=int(config.get("global_topk", 4)),
            selector_hidden=int(config.get("selector_hidden", 8)),
            fitter_hidden=int(config.get("fitter_hidden", 8)),
            selector_temperature=0.5,
            similarity_tau=1.0,
            sigma_min=0.03,
            sigma_max=1.50,
            gap_weight=1.0,
        )

        if checkpoint_payload is not None:
            if isinstance(checkpoint_payload, dict) and "model" in checkpoint_payload:
                state_dict = checkpoint_payload["model"]
            else:
                state_dict = checkpoint_payload
            self.core.load_state_dict(state_dict, strict=True)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = instance_normalize(input_data)
        _, details = self.core(x)
        return details["score_total"]


class OriginalPPLADBenchmarkModel(nn.Module):
    """
    原版 PPLAD 完整单窗口前向：
    实例归一化 + 原版时间邻域构造 + PPLAD前向 + 异常分数。
    """

    def __init__(
        self,
        batch_size: int,
        win_size: int,
        channels: int,
        checkpoint_payload: Optional[Any] = None,
        local_size: int = 3,
        global_size: int = 20,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        try:
            from model.PPLAD import PPLAD
        except ImportError as exc:
            raise RuntimeError(
                "原版模式需要 model/PPLAD.py。请把本脚本复制到原版 PPLAD 项目中运行。"
            ) from exc

        self.win_size = int(win_size)
        self.local_size = int(local_size)
        self.global_size = int(global_size)
        self.core = PPLAD(
            batch_size=batch_size,
            win_size=win_size,
            enc_in=channels,
            c_out=channels,
            d_model=d_model,
            local_size=[local_size],
            global_size=[global_size],
            channel=channels,
        )

        if checkpoint_payload is not None:
            if isinstance(checkpoint_payload, dict) and "model" in checkpoint_payload:
                state_dict = checkpoint_payload["model"]
            else:
                state_dict = checkpoint_payload
            self.core.load_state_dict(state_dict, strict=False)

    def _relations(
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
                current = torch.cat((prefix, x[:, 0:index, :]), dim=1)
                current = torch.cat((current, x[:, index:index + back, :]), dim=1)
            elif index > boundary:
                suffix = x[:, length - 1, :].unsqueeze(1).repeat(
                    1, back + index - length, 1
                )
                current = torch.cat((x[:, index - front:length, :], suffix), dim=1)
            else:
                current = x[:, index - front:index + back, :]
            windows.append(current)

        relation = torch.cat(windows, dim=0).reshape(
            length, batch, total, channels
        ).permute(1, 0, 3, 2)

        relation = torch.sum(
            (
                relation
                - x.unsqueeze(-1).expand(-1, -1, -1, total)
            ).square(),
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

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = instance_normalize(input_data)
        local, global_, relation = self._relations(x)
        series, prior, _, _, _, _, _, _ = self.core(
            x, local, global_, "test", 0, relation
        )
        local_error = torch.sum((series[0] - local[0]).square(), dim=-1)
        global_error = torch.sum((prior[0] - global_[0]).square(), dim=-1)
        return local_error + global_error


def build_model(
    model_name: str,
    batch_size: int,
    win_size: int,
    channels: int,
    checkpoint_payload: Optional[Any],
) -> nn.Module:
    if model_name == "v4":
        return V4BenchmarkModel(checkpoint_payload)
    return OriginalPPLADBenchmarkModel(
        batch_size=batch_size,
        win_size=win_size,
        channels=channels,
        checkpoint_payload=checkpoint_payload,
    )


def official_score_normalization(score: torch.Tensor) -> torch.Tensor:
    minimum = score.min(dim=-1, keepdim=True).values
    maximum = score.max(dim=-1, keepdim=True).values
    scaled = (score - minimum) / (maximum - minimum + 1e-5)
    return torch.softmax(scaled, dim=-1)


def make_loader(
    dataset_dir: Path,
    loader_dataset: str,
    batch_size: int,
    win_size: int,
    mode: str,
):
    return get_loader_segment(
        137,
        str(dataset_dir),
        batch_size=batch_size,
        win_size=win_size,
        mode=mode,
        dataset=loader_dataset,
    )


def reset_gpu_peak(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device) / MIB
    baseline_reserved = torch.cuda.memory_reserved(device) / MIB
    torch.cuda.reset_peak_memory_stats(device)
    return baseline_allocated, baseline_reserved


def read_gpu_peak(
    device: torch.device,
    baseline_allocated: float,
    baseline_reserved: float,
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

    synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device) / MIB
    peak_reserved = torch.cuda.max_memory_reserved(device) / MIB
    return {
        "gpu_baseline_allocated_mib": baseline_allocated,
        "gpu_peak_allocated_mib": peak_allocated,
        "gpu_incremental_allocated_mib": max(
            0.0, peak_allocated - baseline_allocated
        ),
        "gpu_baseline_reserved_mib": baseline_reserved,
        "gpu_peak_reserved_mib": peak_reserved,
        "gpu_incremental_reserved_mib": max(
            0.0, peak_reserved - baseline_reserved
        ),
    }


@torch.no_grad()
def benchmark_single_batch(
    model: nn.Module,
    batch_cpu: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    model.eval()
    batch_device = batch_cpu.to(device, non_blocking=True)

    for _ in range(max(1, warmup)):
        _ = model(batch_device)
    synchronize(device)

    baseline_allocated, baseline_reserved = reset_gpu_peak(device)
    times_ms: List[float] = []

    with PeakRSSMonitor() as rss:
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            _ = model(batch_device)
            synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "single_batch_latency_ms_mean": float(statistics.mean(times_ms)),
        "single_batch_latency_ms_std": (
            float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0
        ),
        "single_window_latency_ms": float(
            statistics.mean(times_ms) / max(int(batch_cpu.shape[0]), 1)
        ),
        "single_batch_cpu_baseline_rss_mib": rss.baseline_mib,
        "single_batch_cpu_peak_rss_mib": rss.peak_mib,
        "single_batch_cpu_incremental_rss_mib": rss.incremental_mib,
    }
    result.update(
        read_gpu_peak(
            device,
            baseline_allocated=baseline_allocated,
            baseline_reserved=baseline_reserved,
        )
    )
    return result


@torch.no_grad()
def benchmark_full_test(
    model: nn.Module,
    loader,
    device: torch.device,
    win_size: int,
) -> Dict[str, float]:
    model.eval()
    baseline_allocated, baseline_reserved = reset_gpu_peak(device)

    batches = 0
    windows = 0
    with PeakRSSMonitor() as rss:
        synchronize(device)
        start = time.perf_counter()
        for input_data, _ in loader:
            batch = input_data.float().to(device, non_blocking=True)
            _ = model(batch)
            batches += 1
            windows += int(batch.shape[0])
        synchronize(device)
        elapsed = time.perf_counter() - start

    result = {
        "full_test_seconds": float(elapsed),
        "full_test_batches": int(batches),
        "full_test_windows": int(windows),
        "full_test_points": int(windows * win_size),
        "full_test_windows_per_second": float(windows / max(elapsed, 1e-12)),
        "full_test_points_per_second": float(
            windows * win_size / max(elapsed, 1e-12)
        ),
        "full_test_cpu_baseline_rss_mib": rss.baseline_mib,
        "full_test_cpu_peak_rss_mib": rss.peak_mib,
        "full_test_cpu_incremental_rss_mib": rss.incremental_mib,
    }
    gpu = read_gpu_peak(
        device,
        baseline_allocated=baseline_allocated,
        baseline_reserved=baseline_reserved,
    )
    result.update({f"full_test_{key}": value for key, value in gpu.items()})
    return result


@torch.no_grad()
def collect_normalized_scores(
    model: nn.Module,
    loader,
    device: torch.device,
) -> List[np.ndarray]:
    output: List[np.ndarray] = []
    model.eval()
    for input_data, _ in loader:
        batch = input_data.float().to(device, non_blocking=True)
        score = official_score_normalization(model(batch))
        output.append(score.detach().cpu().numpy().reshape(-1))
    return output


@torch.no_grad()
def benchmark_exact_threshold(
    model: nn.Module,
    train_loader,
    threshold_loader,
    device: torch.device,
    anormly_ratio: float,
    full_test_seconds: float,
) -> Dict[str, float]:
    """
    复刻 PPLAD 兼容阈值阶段：
    train scores + test scores -> exact percentile。
    不计算 PA、R-AUC、VUS；这些属于评估指标开销，不是模型推理开销。
    """
    model.eval()
    baseline_allocated, baseline_reserved = reset_gpu_peak(device)

    with PeakRSSMonitor() as rss:
        synchronize(device)
        start = time.perf_counter()
        collected = collect_normalized_scores(model, train_loader, device)
        collected.extend(collect_normalized_scores(model, threshold_loader, device))
        combined = np.concatenate(collected, axis=0)
        threshold = float(
            np.percentile(combined, 100.0 - float(anormly_ratio))
        )
        synchronize(device)
        elapsed = time.perf_counter() - start
        samples = int(combined.size)

    result = {
        "exact_threshold_seconds": float(elapsed),
        "exact_threshold_samples": samples,
        "exact_threshold_value": threshold,
        "threshold_plus_second_test_scoring_seconds": float(
            elapsed + full_test_seconds
        ),
        "exact_threshold_cpu_baseline_rss_mib": rss.baseline_mib,
        "exact_threshold_cpu_peak_rss_mib": rss.peak_mib,
        "exact_threshold_cpu_incremental_rss_mib": rss.incremental_mib,
    }
    gpu = read_gpu_peak(
        device,
        baseline_allocated=baseline_allocated,
        baseline_reserved=baseline_reserved,
    )
    result.update({f"exact_threshold_{key}": value for key, value in gpu.items()})
    return result


def save_results(output_dir: Path, stem: str, results: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / f"{stem}.csv"
    flat = {
        key: value
        for key, value in results.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    print(f"JSON结果：{json_path}")
    print(f"CSV结果 ：{csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V4与原版PPLAD统一轻量化基准测试"
    )
    parser.add_argument("--model", choices=["v4", "original"], default="v4")
    parser.add_argument("--dataset", type=str, default="SMAP")
    parser.add_argument(
        "--loader-dataset",
        type=str,
        default=None,
        help="传给data_loader的名称；WADI通常使用WaDi。",
    )
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--win-size", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="auto",
        help="auto、none或具体检查点路径。",
    )
    parser.add_argument(
        "--include-threshold",
        action="store_true",
        help="额外测试精确阈值阶段；大数据集会明显耗时并占用内存。",
    )
    parser.add_argument("--anormly-ratio", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = args.dataset.upper()
    dataset_dir = PROJECT_ROOT / args.dataset_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_dir}")

    loader_dataset = args.loader_dataset
    if loader_dataset is None:
        loader_dataset = "WaDi" if dataset == "WADI" else dataset

    channels, train_file = infer_channels(dataset_dir)
    device = resolve_device(args.device)

    checkpoint_path: Optional[Path]
    if args.checkpoint.lower() == "none":
        checkpoint_path = None
    elif args.checkpoint.lower() == "auto":
        checkpoint_path = find_v4_checkpoint(dataset) if args.model == "v4" else None
    else:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在：{checkpoint_path}")

    payload = load_checkpoint_payload(checkpoint_path)

    # 先获取一个真实batch，再按实际batch大小创建原版模型。
    probe_loader = make_loader(
        dataset_dir=dataset_dir,
        loader_dataset=loader_dataset,
        batch_size=args.batch_size,
        win_size=args.win_size,
        mode="thre",
    )
    try:
        first_batch, _ = next(iter(probe_loader))
    except StopIteration as exc:
        raise RuntimeError(f"{dataset} 测试集为空。") from exc

    model = build_model(
        model_name=args.model,
        batch_size=int(first_batch.shape[0]),
        win_size=args.win_size,
        channels=channels,
        checkpoint_payload=payload,
    ).to(device)

    stats = parameter_stats(model)
    checkpoint_disk_bytes = (
        int(checkpoint_path.stat().st_size) if checkpoint_path is not None else 0
    )

    single = benchmark_single_batch(
        model=model,
        batch_cpu=first_batch.float(),
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # 重新创建loader，避免probe迭代器状态影响统计。
    full_test_loader = make_loader(
        dataset_dir=dataset_dir,
        loader_dataset=loader_dataset,
        batch_size=args.batch_size,
        win_size=args.win_size,
        mode="thre",
    )
    full_test = benchmark_full_test(
        model=model,
        loader=full_test_loader,
        device=device,
        win_size=args.win_size,
    )

    ratio = (
        float(args.anormly_ratio)
        if args.anormly_ratio is not None
        else float(DEFAULT_RATIOS.get(dataset, 0.5))
    )

    threshold_results: Dict[str, Any] = {}
    if args.include_threshold:
        train_loader = make_loader(
            dataset_dir=dataset_dir,
            loader_dataset=loader_dataset,
            batch_size=args.batch_size,
            win_size=args.win_size,
            mode="train",
        )
        threshold_loader = make_loader(
            dataset_dir=dataset_dir,
            loader_dataset=loader_dataset,
            batch_size=args.batch_size,
            win_size=args.win_size,
            mode="thre",
        )
        threshold_results = benchmark_exact_threshold(
            model=model,
            train_loader=train_loader,
            threshold_loader=threshold_loader,
            device=device,
            anormly_ratio=ratio,
            full_test_seconds=full_test["full_test_seconds"],
        )

    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )

    results: Dict[str, Any] = {
        "model": args.model,
        "dataset": dataset,
        "loader_dataset": loader_dataset,
        "device": str(device),
        "device_name": gpu_name,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "win_size": args.win_size,
        "channels": channels,
        "train_file": str(train_file),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_disk_bytes": checkpoint_disk_bytes,
        "checkpoint_disk_kib": checkpoint_disk_bytes / 1024.0,
        "anormly_ratio": ratio,
        **stats,
        **single,
        **full_test,
        **threshold_results,
    }

    print("\n========== Lightweight Benchmark ==========")
    print(f"Model                         : {args.model}")
    print(f"Dataset                       : {dataset}")
    print(f"Device                        : {device} ({gpu_name})")
    print(f"Trainable parameters          : {stats['trainable_params']:,}")
    print(f"Total parameters              : {stats['total_params']:,}")
    print(f"Buffer elements               : {stats['buffer_elements']:,}")
    print(
        "Parameter tensor size         : "
        f"{stats['parameter_tensor_bytes'] / 1024.0:.3f} KiB"
    )
    print(
        "Serialized state_dict size    : "
        f"{stats['serialized_state_dict_bytes'] / 1024.0:.3f} KiB"
    )
    if checkpoint_path:
        print(
            "Checkpoint file size          : "
            f"{checkpoint_disk_bytes / 1024.0:.3f} KiB"
        )
    else:
        print("Checkpoint file size          : not loaded")

    print(
        "Single-batch inference         : "
        f"{single['single_batch_latency_ms_mean']:.3f} "
        f"± {single['single_batch_latency_ms_std']:.3f} ms"
    )
    print(
        "Single-window inference        : "
        f"{single['single_window_latency_ms']:.6f} ms"
    )
    print(
        "Full-test inference            : "
        f"{full_test['full_test_seconds']:.3f} s"
    )
    print(
        "Full-test throughput           : "
        f"{full_test['full_test_points_per_second']:.1f} points/s"
    )
    print(
        "Full-test GPU peak allocated   : "
        f"{full_test['full_test_gpu_peak_allocated_mib']:.3f} MiB"
    )
    print(
        "Full-test GPU incremental      : "
        f"{full_test['full_test_gpu_incremental_allocated_mib']:.3f} MiB"
    )
    print(
        "Full-test CPU peak RSS         : "
        f"{full_test['full_test_cpu_peak_rss_mib']:.3f} MiB"
    )
    print(
        "Full-test CPU incremental RSS  : "
        f"{full_test['full_test_cpu_incremental_rss_mib']:.3f} MiB"
    )

    if threshold_results:
        print(
            "Exact-threshold stage          : "
            f"{threshold_results['exact_threshold_seconds']:.3f} s"
        )
        print(
            "Exact-threshold samples        : "
            f"{threshold_results['exact_threshold_samples']:,}"
        )
        print(
            "Exact-threshold CPU peak RSS   : "
            f"{threshold_results['exact_threshold_cpu_peak_rss_mib']:.3f} MiB"
        )
        print(
            "Threshold + second test scoring: "
            f"{threshold_results['threshold_plus_second_test_scoring_seconds']:.3f} s"
        )
        print(
            "Note                          : "
            "上述总时间不含PA、R-AUC、VUS等高级评估指标。"
        )

    stem = f"{args.model}_{dataset.lower()}_{device.type}"
    save_results(Path(args.output_dir), stem, results)


if __name__ == "__main__":
    main()
