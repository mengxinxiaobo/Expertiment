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
from typing import Dict, List, Optional

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
PATCHAD_ROOT = SCRIPT.parent
sys.path.insert(0, str(PATCHAD_ROOT))
os.chdir(PATCHAD_ROOT)

from data.data_loader2 import data_name2nc, get_loader_segment
from patchad_model.models import PatchMLPAD

MIB = 1024 ** 2


def current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class PeakRSS:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.base = current_rss_bytes()
        self.peak = self.base
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, current_rss_bytes())
            self.stop.wait(self.interval)
        self.peak = max(self.peak, current_rss_bytes())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def state_dict_bytes(model: torch.nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def make_loader(data_root: Path, dataset: str, batch_size: int, win_size: int, mode: str):
    return get_loader_segment(
        137,
        str(data_root / dataset),
        batch_size=batch_size,
        win_size=win_size,
        mode=mode,
        dataset=dataset,
        step=1,
    )


@torch.no_grad()
def bench_batch(model, batch, device, warmup, repeats):
    model.eval()
    batch = batch.float().to(device)
    for _ in range(warmup):
        model(batch)
    sync(device)

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        sync(device)
        base_alloc = torch.cuda.memory_allocated(device) / MIB
        torch.cuda.reset_peak_memory_stats(device)
    else:
        base_alloc = 0.0

    times = []
    with PeakRSS() as rss:
        for _ in range(repeats):
            sync(device)
            t0 = time.perf_counter()
            model(batch)
            sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)

    peak_alloc = (
        torch.cuda.max_memory_allocated(device) / MIB
        if device.type == "cuda" else 0.0
    )
    return {
        "single_batch_latency_ms_mean": statistics.mean(times),
        "single_batch_latency_ms_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "single_window_latency_ms": statistics.mean(times) / max(batch.shape[0], 1),
        "single_batch_gpu_peak_allocated_mib": peak_alloc,
        "single_batch_gpu_incremental_allocated_mib": max(0.0, peak_alloc - base_alloc),
        "single_batch_cpu_peak_rss_mib": rss.peak / MIB,
        "single_batch_cpu_incremental_rss_mib": max(0.0, (rss.peak - rss.base) / MIB),
    }


@torch.no_grad()
def bench_full(model, loader, device, win_size):
    model.eval()
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        sync(device)
        base_alloc = torch.cuda.memory_allocated(device) / MIB
        torch.cuda.reset_peak_memory_stats(device)
    else:
        base_alloc = 0.0

    windows = 0
    batches = 0
    with PeakRSS() as rss:
        sync(device)
        t0 = time.perf_counter()
        for x, _ in loader:
            x = x.float().to(device)
            model(x)
            windows += x.shape[0]
            batches += 1
        sync(device)
        elapsed = time.perf_counter() - t0

    peak_alloc = (
        torch.cuda.max_memory_allocated(device) / MIB
        if device.type == "cuda" else 0.0
    )
    return {
        "full_test_seconds": elapsed,
        "full_test_batches": batches,
        "full_test_windows": windows,
        "full_test_points": windows * win_size,
        "full_test_windows_per_second": windows / max(elapsed, 1e-12),
        "full_test_points_per_second": windows * win_size / max(elapsed, 1e-12),
        "full_test_gpu_peak_allocated_mib": peak_alloc,
        "full_test_gpu_incremental_allocated_mib": max(0.0, peak_alloc - base_alloc),
        "full_test_cpu_peak_rss_mib": rss.peak / MIB,
        "full_test_cpu_incremental_rss_mib": max(0.0, (rss.peak - rss.base) / MIB),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["MSL", "SMAP"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", default="none")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--win-size", type=int, default=105)
    p.add_argument("--patch-sizes", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--d-model", type=int, default=60)
    p.add_argument("--e-layer", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results_compare/benchmark")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = PatchMLPAD(
        win_size=args.win_size,
        e_layer=args.e_layer,
        patch_sizes=args.patch_sizes,
        dropout=0.1,
        activation="relu",
        output_attention=True,
        channel=data_name2nc(args.dataset),
        d_model=args.d_model,
        cont_model=args.win_size,
        norm="n",
    )

    ckpt_path: Optional[Path] = None
    if args.checkpoint.lower() != "none":
        ckpt_path = Path(args.checkpoint).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"检查点不存在：{ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    tensor_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    serialized = state_dict_bytes(model)

    loader_probe = make_loader(
        Path(args.data_root), args.dataset, args.batch_size, args.win_size, "thre"
    )
    first_batch, _ = next(iter(loader_probe))

    single = bench_batch(model, first_batch, device, args.warmup, args.repeats)
    full_loader = make_loader(
        Path(args.data_root), args.dataset, args.batch_size, args.win_size, "thre"
    )
    full = bench_full(model, full_loader, device, args.win_size)

    ckpt_bytes = ckpt_path.stat().st_size if ckpt_path else 0
    result = {
        "model": "PatchAD",
        "dataset": args.dataset,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "win_size": args.win_size,
        "patch_sizes": str(args.patch_sizes),
        "d_model": args.d_model,
        "e_layer": args.e_layer,
        "trainable_params": trainable,
        "total_params": total,
        "parameter_tensor_bytes": tensor_bytes,
        "serialized_state_dict_bytes": serialized,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "checkpoint_disk_bytes": ckpt_bytes,
        **single,
        **full,
    }

    print("\n========== PatchAD Lightweight Benchmark ==========")
    print(f"Dataset                       : {args.dataset}")
    print(f"Device                        : {device} ({result['device_name']})")
    print(f"Trainable parameters          : {trainable:,}")
    print(f"Parameter tensor size         : {tensor_bytes / 1024:.3f} KiB")
    print(f"Serialized state_dict size    : {serialized / 1024:.3f} KiB")
    print(f"Checkpoint file size          : {ckpt_bytes / 1024:.3f} KiB")
    print(
        f"Single-batch inference        : "
        f"{single['single_batch_latency_ms_mean']:.3f} ± "
        f"{single['single_batch_latency_ms_std']:.3f} ms"
    )
    print(f"Single-window inference       : {single['single_window_latency_ms']:.6f} ms")
    print(f"Full-test inference           : {full['full_test_seconds']:.3f} s")
    print(f"Full-test throughput          : {full['full_test_points_per_second']:.1f} points/s")
    print(f"Full-test GPU peak allocated  : {full['full_test_gpu_peak_allocated_mib']:.3f} MiB")
    print(f"Full-test GPU incremental     : {full['full_test_gpu_incremental_allocated_mib']:.3f} MiB")
    print(f"Full-test CPU peak RSS        : {full['full_test_cpu_peak_rss_mib']:.3f} MiB")
    print(f"Full-test CPU incremental RSS : {full['full_test_cpu_incremental_rss_mib']:.3f} MiB")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"patchad_{args.dataset.lower()}_{device.type}"
    (out / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / f"{stem}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(f"JSON结果：{out / f'{stem}.json'}")
    print(f"CSV结果 ：{out / f'{stem}.csv'}")


if __name__ == "__main__":
    main()
