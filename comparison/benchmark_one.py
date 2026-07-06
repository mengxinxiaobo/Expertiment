#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import io
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import write_json


MIB = 1024 ** 2


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, win_size: int, mode: str) -> None:
        self.data = data
        self.win_size = int(win_size)
        self.mode = mode

    def __len__(self) -> int:
        n = int(self.data.shape[0])
        if self.mode == "train":
            return max(0, n - self.win_size + 1)
        return max(0, (n - self.win_size) // self.win_size + 1)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index if self.mode == "train" else index * self.win_size
        window = np.asarray(
            self.data[start : start + self.win_size], dtype=np.float32
        )
        return torch.from_numpy(window)


class RSSMonitor:
    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.baseline = self.current()
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    @staticmethod
    def current() -> int:
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text(errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        return 0

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self.current())
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.current())


def instance_normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True).detach()
    var = x.var(dim=1, keepdim=True, unbiased=False).detach()
    return (x - mean) / torch.sqrt(var + eps)


class OriginalWrapper(nn.Module):
    def __init__(
        self,
        project_root: Path,
        batch_size: int,
        win_size: int,
        channels: int,
        local_size: int,
        global_size: int,
        d_model: int,
    ) -> None:
        super().__init__()
        sys.path.insert(0, str(project_root))
        try:
            from model.PPLAD import PPLAD
        except ImportError as exc:
            raise RuntimeError(
                f"无法从 {project_root} 导入 model/PPLAD.py"
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

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = instance_normalize(input_data)
        local, global_, relation = self._relations(x)
        series, prior, _, _, _, _, _, _ = self.core(
            x, local, global_, "test", 0, relation
        )
        local_error = torch.sum((series[0] - local[0]).square(), dim=-1)
        global_error = torch.sum((prior[0] - global_[0]).square(), dim=-1)
        return local_error + global_error


class V4Wrapper(nn.Module):
    def __init__(self, project_root: Path, checkpoint: Optional[Path]) -> None:
        super().__init__()
        sys.path.insert(0, str(project_root))
        from main import AdaptiveSparseAnchorCompetitiveModelV4
        payload: Dict[str, Any] = {}
        if checkpoint is not None:
            loaded = torch.load(checkpoint, map_location="cpu")
            if isinstance(loaded, dict):
                payload = loaded
        cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        self.core = AdaptiveSparseAnchorCompetitiveModelV4(
            local_candidate_lags=cfg.get(
                "local_candidate_lags", [1, 2, 3, 4, 5, 6, 7, 8]
            ),
            global_candidate_lags=cfg.get(
                "global_candidate_lags", [12, 16, 20, 24, 28, 32, 40, 48]
            ),
            local_topk=int(cfg.get("local_topk", 2)),
            global_topk=int(cfg.get("global_topk", 4)),
            selector_hidden=int(cfg.get("selector_hidden", 8)),
            fitter_hidden=int(cfg.get("fitter_hidden", 8)),
            selector_temperature=0.5,
            similarity_tau=1.0,
            sigma_min=0.03,
            sigma_max=1.50,
            gap_weight=1.0,
        )
        if payload:
            state = payload.get("model", payload)
            self.core.load_state_dict(state, strict=True)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        _, details = self.core(instance_normalize(input_data))
        return details["score_total"]


def official_normalize(score: torch.Tensor) -> torch.Tensor:
    minimum = score.min(dim=-1, keepdim=True).values
    maximum = score.max(dim=-1, keepdim=True).values
    return torch.softmax((score - minimum) / (maximum - minimum + 1e-5), dim=-1)


def parameter_stats(model: nn.Module) -> Dict[str, int]:
    params = list(model.parameters())
    buffers = list(model.buffers())
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return {
        "trainable_params": int(sum(p.numel() for p in params if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in params)),
        "parameter_tensor_bytes": int(
            sum(p.numel() * p.element_size() for p in params)
        ),
        "buffer_elements": int(sum(b.numel() for b in buffers)),
        "buffer_tensor_bytes": int(
            sum(b.numel() * b.element_size() for b in buffers)
        ),
        "serialized_state_dict_bytes": int(stream.tell()),
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_gpu(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    gc.collect()
    torch.cuda.empty_cache()
    sync(device)
    base_alloc = torch.cuda.memory_allocated(device) / MIB
    base_reserved = torch.cuda.memory_reserved(device) / MIB
    torch.cuda.reset_peak_memory_stats(device)
    return base_alloc, base_reserved


def gpu_result(device: torch.device, base_alloc: float, base_reserved: float) -> Dict[str, float]:
    if device.type != "cuda":
        return {
            "gpu_peak_allocated_mib": 0.0,
            "gpu_incremental_allocated_mib": 0.0,
            "gpu_peak_reserved_mib": 0.0,
            "gpu_incremental_reserved_mib": 0.0,
        }
    sync(device)
    peak_alloc = torch.cuda.max_memory_allocated(device) / MIB
    peak_reserved = torch.cuda.max_memory_reserved(device) / MIB
    return {
        "gpu_peak_allocated_mib": float(peak_alloc),
        "gpu_incremental_allocated_mib": float(max(0.0, peak_alloc - base_alloc)),
        "gpu_peak_reserved_mib": float(peak_reserved),
        "gpu_incremental_reserved_mib": float(
            max(0.0, peak_reserved - base_reserved)
        ),
    }


@torch.no_grad()
def benchmark_batch(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    model.eval()
    batch = batch.to(device)
    for _ in range(max(1, warmup)):
        model(batch)
    sync(device)
    base_alloc, base_reserved = reset_gpu(device)
    values: List[float] = []
    with RSSMonitor() as rss:
        for _ in range(repeats):
            sync(device)
            start = time.perf_counter()
            model(batch)
            sync(device)
            values.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(values)
    result = {
        "single_batch_size_actual": int(batch.shape[0]),
        "single_batch_latency_ms_mean": float(arr.mean()),
        "single_batch_latency_ms_std": float(arr.std()),
        "single_batch_latency_ms_p50": float(np.percentile(arr, 50)),
        "single_batch_latency_ms_p95": float(np.percentile(arr, 95)),
        "single_batch_latency_ms_p99": float(np.percentile(arr, 99)),
        "single_window_latency_ms_mean": float(arr.mean() / batch.shape[0]),
        "single_batch_cpu_baseline_rss_mib": float(rss.baseline / MIB),
        "single_batch_cpu_peak_rss_mib": float(rss.peak / MIB),
        "single_batch_cpu_incremental_rss_mib": float(
            max(0, rss.peak - rss.baseline) / MIB
        ),
    }
    result.update(gpu_result(device, base_alloc, base_reserved))
    return result


@torch.no_grad()
def benchmark_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    win_size: int,
) -> Dict[str, Any]:
    model.eval()
    base_alloc, base_reserved = reset_gpu(device)
    windows = 0
    batches = 0
    with RSSMonitor() as rss:
        sync(device)
        start = time.perf_counter()
        for batch in loader:
            model(batch.float().to(device))
            windows += int(batch.shape[0])
            batches += 1
        sync(device)
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
        "full_test_cpu_baseline_rss_mib": float(rss.baseline / MIB),
        "full_test_cpu_peak_rss_mib": float(rss.peak / MIB),
        "full_test_cpu_incremental_rss_mib": float(
            max(0, rss.peak - rss.baseline) / MIB
        ),
    }
    result.update(
        {
            f"full_test_{key}": value
            for key, value in gpu_result(device, base_alloc, base_reserved).items()
        }
    )
    return result


@torch.no_grad()
def exact_threshold_benchmark(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    ratio: float,
) -> Dict[str, Any]:
    model.eval()
    base_alloc, base_reserved = reset_gpu(device)
    arrays: List[np.ndarray] = []
    with RSSMonitor() as rss:
        sync(device)
        start = time.perf_counter()
        for loader in (train_loader, test_loader):
            for batch in loader:
                score = official_normalize(model(batch.float().to(device)))
                arrays.append(score.detach().cpu().numpy().reshape(-1))
        combined = np.concatenate(arrays)
        threshold = float(np.percentile(combined, 100.0 - ratio))
        sync(device)
        elapsed = time.perf_counter() - start
    result = {
        "exact_threshold_seconds": float(elapsed),
        "exact_threshold_samples": int(combined.size),
        "exact_threshold_value": threshold,
        "exact_threshold_cpu_baseline_rss_mib": float(rss.baseline / MIB),
        "exact_threshold_cpu_peak_rss_mib": float(rss.peak / MIB),
        "exact_threshold_cpu_incremental_rss_mib": float(
            max(0, rss.peak - rss.baseline) / MIB
        ),
    }
    result.update(
        {
            f"exact_threshold_{key}": value
            for key, value in gpu_result(device, base_alloc, base_reserved).items()
        }
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["original", "v4"], required=True)
    p.add_argument("--project-root", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--file-prefix", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", choices=["native", "controlled"], required=True)
    p.add_argument("--win-size", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--anormly-ratio", type=float, required=True)
    p.add_argument("--local-size", type=int, default=3)
    p.add_argument("--global-size", type=int, default=20)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--checkpoint", default="none")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--include-threshold", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    set_seed(42)
    project_root = Path(args.project_root).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    train_path = dataset_dir / f"{args.file_prefix}_train.npy"
    test_path = dataset_dir / f"{args.file_prefix}_test.npy"
    train = np.load(train_path, mmap_mode="r", allow_pickle=False)
    test = np.load(test_path, mmap_mode="r", allow_pickle=False)
    channels = int(train.shape[1])

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用CUDA。")

    test_dataset = WindowDataset(test, args.win_size, "thre")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    first = next(iter(test_loader)).float()

    checkpoint = None
    if args.checkpoint.lower() != "none":
        checkpoint = Path(args.checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    if args.model == "original":
        model = OriginalWrapper(
            project_root,
            batch_size=int(first.shape[0]),
            win_size=args.win_size,
            channels=channels,
            local_size=args.local_size,
            global_size=args.global_size,
            d_model=args.d_model,
        )
        weights_source = "fresh instance; original code does not save a checkpoint"
    else:
        model = V4Wrapper(project_root, checkpoint)
        weights_source = str(checkpoint) if checkpoint else "fresh instance"

    model = model.to(device)
    result: Dict[str, Any] = {
        "dataset": args.dataset,
        "model": args.model,
        "protocol": args.protocol,
        "project_root": str(project_root),
        "dataset_dir": str(dataset_dir),
        "file_prefix": args.file_prefix,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "win_size": args.win_size,
        "batch_size": args.batch_size,
        "channels": channels,
        "anormly_ratio": args.anormly_ratio,
        "weights_source": weights_source,
        "checkpoint_disk_bytes": (
            int(checkpoint.stat().st_size) if checkpoint is not None else 0
        ),
    }
    result.update(parameter_stats(model))
    result.update(
        benchmark_batch(model, first, device, args.warmup, args.repeats)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    result.update(benchmark_loader(model, test_loader, device, args.win_size))

    if args.include_threshold:
        train_loader = DataLoader(
            WindowDataset(train, args.win_size, "train"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        result.update(
            exact_threshold_benchmark(
                model,
                train_loader,
                test_loader,
                device,
                args.anormly_ratio,
            )
        )

    write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
