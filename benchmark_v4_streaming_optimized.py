#!/usr/bin/env python3
from __future__ import annotations

"""
V4 流式候选锚点优化版基准脚本

用途：
1. 不修改原来的 main.py。
2. 加载已有 V4 checkpoint（检查点）。
3. 使用流式 lag 计算，避免一次性生成 [B, L, K, M] 大张量。
4. 验证优化前后异常分数是否近似一致。
5. 只测试优化后的 V4，保存时延、吞吐率、CPU/GPU 峰值内存和精确阈值开销。

建议从项目根目录执行：
    python benchmark_v4_streaming_optimized.py --dataset SMAP --include-threshold
"""

import argparse
import csv
import gc
import io
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU。")
        return torch.device("cuda:0")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def instance_normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True).detach()
    var = x.var(dim=1, keepdim=True, unbiased=False).detach()
    return (x - mean) / torch.sqrt(var + eps)


class WindowDataset(Dataset):
    """
    与此前 comparison/benchmark_one.py 相同的窗口规则：
    - train：步长 1；
    - test/thre：步长 win_size，非重叠窗口。
    """

    def __init__(self, data: np.ndarray, win_size: int, mode: str) -> None:
        if data.ndim != 2:
            raise ValueError(f"数据必须是二维数组，实际 shape={data.shape}")
        if win_size <= 0:
            raise ValueError("win_size 必须大于 0。")
        if mode not in {"train", "test"}:
            raise ValueError(f"未知 mode：{mode}")
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
            self.data[start : start + self.win_size],
            dtype=np.float32,
        )
        return torch.from_numpy(window)


class RSSMonitor:
    """采样当前 Python 进程的 RSS（Resident Set Size，常驻内存）。"""

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = float(interval)
        self.baseline_bytes = self.current_rss_bytes()
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def current_rss_bytes() -> int:
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        try:
            import psutil

            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(
                self.peak_bytes,
                self.current_rss_bytes(),
            )
            self._stop.wait(self.interval)
        self.peak_bytes = max(
            self.peak_bytes,
            self.current_rss_bytes(),
        )

    def __enter__(self) -> "RSSMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()


# 从当前项目 main.py 继承原 V4，确保参数名称和 checkpoint 完全兼容。
try:
    from main import AdaptiveSparseAnchorCompetitiveModelV4
except ImportError as exc:
    raise RuntimeError(
        "请把本文件放在包含 main.py 的 V4 项目根目录中执行。"
    ) from exc


class StreamingAdaptiveSparseAnchorCompetitiveModelV4(
    AdaptiveSparseAnchorCompetitiveModelV4
):
    """
    流式 lag 优化版本。

    原实现会一次性构造：
        left_x/right_x: [B, L, K, M]

    本实现逐个 lag 计算并立即沿通道维压缩，只保留：
        单个 lag 的 shifted x: [B, L, M]
        最终 affinity:          [B, L, K]
        最终 delta:             [B, L, K, 4]

    模型参数、候选 lag、Top-k、Gaussian fitting（高斯拟合）和异常分数定义不变。
    """

    @staticmethod
    def _shift_replicate(
        x: torch.Tensor,
        lag: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim < 2:
            raise ValueError("输入至少需要 [B, L, ...] 两个维度。")
        length = int(x.shape[1])
        if not 0 < lag < length:
            raise ValueError(
                f"必须满足 0 < lag < sequence length，"
                f"实际 lag={lag}, length={length}"
            )

        left_padding = x[:, :1, ...].expand(
            -1,
            lag,
            *([-1] * (x.ndim - 2)),
        )
        right_padding = x[:, -1:, ...].expand(
            -1,
            lag,
            *([-1] * (x.ndim - 2)),
        )

        left = torch.cat(
            [left_padding, x[:, : length - lag, ...]],
            dim=1,
        )
        right = torch.cat(
            [x[:, lag:, ...], right_padding],
            dim=1,
        )
        return left, right

    def _candidate_group(
        self,
        x: torch.Tensor,
        sketch: torch.Tensor,
        lags: torch.Tensor,
        group_value: float,
    ) -> Dict[str, torch.Tensor]:
        affinity_parts: List[torch.Tensor] = []
        delta_parts: List[torch.Tensor] = []

        # 逐个 lag 处理，避免 [B,L,K,M] 级别的完整候选张量。
        for lag_tensor in lags:
            lag = int(lag_tensor.item())

            left_x, right_x = self._shift_replicate(x, lag)
            left_distance = (x - left_x).square().mean(dim=-1)
            right_distance = (x - right_x).square().mean(dim=-1)
            affinity = (
                torch.exp(-left_distance / self.similarity_tau)
                + torch.exp(-right_distance / self.similarity_tau)
            )
            affinity_parts.append(affinity)

            left_sketch, right_sketch = self._shift_replicate(sketch, lag)
            delta = 0.5 * (
                torch.abs(sketch - left_sketch)
                + torch.abs(sketch - right_sketch)
            )
            delta_parts.append(delta)

            # 显式去掉当前 lag 的大张量引用，缩短生命周期。
            del left_x, right_x
            del left_distance, right_distance
            del left_sketch, right_sketch

        affinity = torch.stack(affinity_parts, dim=-1)  # [B,L,K]
        delta = torch.stack(delta_parts, dim=2)         # [B,L,K,4]

        batch, length, count, _ = delta.shape
        lag_feature = (
            lags.to(dtype=x.dtype, device=x.device)
            .view(1, 1, count, 1)
            / self.max_lag
        ).expand(batch, length, -1, -1)

        group_feature = x.new_full(
            (batch, length, count, 1),
            float(group_value),
        )
        selector_features = torch.cat(
            [delta, lag_feature, group_feature],
            dim=-1,
        )
        logits = self.selector(selector_features)

        return {
            "affinity": affinity,
            "logits": logits,
            "lag_norm": lag_feature.squeeze(-1),
        }

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = True,
    ):
        if x.ndim != 3:
            raise ValueError("输入必须为 [B,L,M]。")

        sketch = self._sequence_sketch(x)
        local = self._candidate_group(
            x,
            sketch,
            self.local_lags,
            group_value=0.0,
        )
        global_ = self._candidate_group(
            x,
            sketch,
            self.global_lags,
            group_value=1.0,
        )

        local_gate, local_weights, local_index = self._straight_through_topk(
            local["logits"],
            self.local_topk,
        )
        global_gate, global_weights, global_index = self._straight_through_topk(
            global_["logits"],
            self.global_topk,
        )

        center_affinity = torch.ones_like(
            local["affinity"][..., :1]
        )
        joint_affinity = torch.cat(
            [
                center_affinity,
                local["affinity"],
                global_["affinity"],
            ],
            dim=-1,
        )
        joint_target = joint_affinity / joint_affinity.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        local_count = self.local_lags.numel()
        target_center = joint_target[..., :1]
        target_local = joint_target[..., 1 : 1 + local_count]
        target_global = joint_target[..., 1 + local_count :]

        local_summary = self._weighted_summary(
            target_local,
            local["lag_norm"],
            local_gate,
            local_weights,
        )
        global_summary = self._weighted_summary(
            target_global,
            global_["lag_norm"],
            global_gate,
            global_weights,
        )

        sigma = self.fitter(
            torch.cat(
                [local_summary, global_summary],
                dim=-1,
            )
        )

        sqrt_two_pi = math.sqrt(2.0 * math.pi)
        center_pdf = 1.0 / (sqrt_two_pi * sigma).unsqueeze(-1)
        local_pdf = (
            2.0
            / (sqrt_two_pi * sigma.unsqueeze(-1))
            * torch.exp(
                -0.5
                * (
                    local["lag_norm"] / sigma.unsqueeze(-1)
                ).square()
            )
        )
        global_pdf = (
            2.0
            / (sqrt_two_pi * sigma.unsqueeze(-1))
            * torch.exp(
                -0.5
                * (
                    global_["lag_norm"] / sigma.unsqueeze(-1)
                ).square()
            )
        )

        joint_pdf = torch.cat(
            [center_pdf, local_pdf, global_pdf],
            dim=-1,
        )
        joint_gaussian = joint_pdf / joint_pdf.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        pred_center = joint_gaussian[..., :1]
        pred_local = joint_gaussian[..., 1 : 1 + local_count]
        pred_global = joint_gaussian[..., 1 + local_count :]

        center_weight = local_weights.new_full(
            (*local_weights.shape[:-1], 1),
            1.0 / float(self.local_topk + 1),
        )
        local_pair_weights = (
            local_weights
            * (
                float(self.local_topk)
                / float(self.local_topk + 1)
            )
        )

        local_fit = (
            center_weight.squeeze(-1)
            * (
                target_center.squeeze(-1)
                - pred_center.squeeze(-1)
            ).square()
            + self._weighted_fit_error(
                target_local,
                pred_local,
                local_pair_weights,
            )
        )
        global_fit = self._weighted_fit_error(
            target_global,
            pred_global,
            global_weights,
        )

        standard_normal = torch.distributions.Normal(
            torch.zeros_like(sigma),
            torch.ones_like(sigma),
        )
        edge_z = sigma.new_tensor(self.local_edge) / sigma
        pred_local_area = (
            standard_normal.cdf(edge_z)
            - standard_normal.cdf(-edge_z)
        )
        pred_global_area = 1.0 - pred_local_area

        target_local_area = (
            target_center.squeeze(-1)
            + target_local.sum(dim=-1)
        )
        target_global_area = target_global.sum(dim=-1)

        local_area_error = (
            pred_local_area - target_local_area
        ).square()
        global_area_error = (
            pred_global_area - target_global_area
        ).square()
        area_error = local_area_error + global_area_error

        score_gap = torch.abs(local_fit - global_fit)
        score_total = local_fit + global_fit
        score_combined = (
            score_total
            + self.gap_weight * score_gap
        )

        # 部署/基准阶段只返回 score_total，避免 details 长时间持有中间张量。
        if not return_details:
            return score_total

        details = {
            "local_fit": local_fit,
            "global_fit": global_fit,
            "area_error": area_error,
            "local_area_error": local_area_error,
            "global_area_error": global_area_error,
            "score_gap": score_gap,
            "score_total": score_total,
            "score_combined": score_combined,
            "sigma": sigma,
            "local_probabilities": torch.softmax(
                local["logits"] / self.selector_temperature,
                dim=-1,
            ),
            "global_probabilities": torch.softmax(
                global_["logits"] / self.selector_temperature,
                dim=-1,
            ),
            "local_gate": local_gate,
            "global_gate": global_gate,
            "local_selected_index": local_index,
            "global_selected_index": global_index,
        }
        return score_combined, details


class OptimizedV4Wrapper(nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()

        payload = torch.load(checkpoint, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                "V4 checkpoint 应为包含 model/config 的字典。"
            )

        config = payload.get("config", {}) or {}
        self.core = StreamingAdaptiveSparseAnchorCompetitiveModelV4(
            local_candidate_lags=config.get(
                "local_candidate_lags",
                [1, 2, 3, 4, 5, 6, 7, 8],
            ),
            global_candidate_lags=config.get(
                "global_candidate_lags",
                [12, 16, 20, 24, 28, 32, 40, 48],
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

        state_dict = payload.get("model", payload)
        self.core.load_state_dict(
            state_dict,
            strict=True,
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        normalized = instance_normalize(input_data)
        return self.core(
            normalized,
            return_details=False,
        )


class OriginalV4Wrapper(nn.Module):
    """仅用于一次数值等价性检查，不参与优化版性能计时。"""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()

        payload = torch.load(checkpoint, map_location="cpu")
        config = payload.get("config", {}) or {}
        self.core = AdaptiveSparseAnchorCompetitiveModelV4(
            local_candidate_lags=config.get(
                "local_candidate_lags",
                [1, 2, 3, 4, 5, 6, 7, 8],
            ),
            global_candidate_lags=config.get(
                "global_candidate_lags",
                [12, 16, 20, 24, 28, 32, 40, 48],
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
        self.core.load_state_dict(
            payload.get("model", payload),
            strict=True,
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        normalized = instance_normalize(input_data)
        _, details = self.core(normalized)
        return details["score_total"]


def find_checkpoint(
    project_root: Path,
    dataset: str,
) -> Path:
    candidates: List[Path] = []

    candidates.extend(
        (
            project_root
            / "checkpoints"
            / dataset
        ).glob("*adaptive_anchor_v4*.pt")
    )
    candidates.extend(
        project_root.glob(
            f"comparison_runs/*/{dataset}/v4/checkpoints/"
            "*adaptive_anchor_v4*.pt"
        )
    )

    candidates = [
        path for path in candidates
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"未找到 {dataset} 的 V4 checkpoint。"
            "可通过 --checkpoint 指定具体路径。"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    ).resolve()


def parameter_stats(module: nn.Module) -> Dict[str, int]:
    parameters = list(module.parameters())
    buffers = list(module.buffers())

    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)

    return {
        "trainable_params": int(
            sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad
            )
        ),
        "total_params": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "parameter_tensor_bytes": int(
            sum(
                parameter.numel() * parameter.element_size()
                for parameter in parameters
            )
        ),
        "buffer_elements": int(
            sum(buffer.numel() for buffer in buffers)
        ),
        "buffer_tensor_bytes": int(
            sum(
                buffer.numel() * buffer.element_size()
                for buffer in buffers
            )
        ),
        "serialized_state_dict_bytes": int(stream.tell()),
    }


def reset_gpu_peak(
    device: torch.device,
) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0

    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)

    baseline_allocated = (
        torch.cuda.memory_allocated(device) / MIB
    )
    baseline_reserved = (
        torch.cuda.memory_reserved(device) / MIB
    )
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
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / MIB
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / MIB
    )

    return {
        "gpu_baseline_allocated_mib": float(
            baseline_allocated
        ),
        "gpu_peak_allocated_mib": float(
            peak_allocated
        ),
        "gpu_incremental_allocated_mib": float(
            max(
                0.0,
                peak_allocated - baseline_allocated,
            )
        ),
        "gpu_baseline_reserved_mib": float(
            baseline_reserved
        ),
        "gpu_peak_reserved_mib": float(
            peak_reserved
        ),
        "gpu_incremental_reserved_mib": float(
            max(
                0.0,
                peak_reserved - baseline_reserved,
            )
        ),
    }


@torch.inference_mode()
def equivalence_check(
    checkpoint: Path,
    batch_cpu: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    original = OriginalV4Wrapper(checkpoint).to(device).eval()
    optimized = OptimizedV4Wrapper(checkpoint).to(device).eval()
    batch = batch_cpu.to(device)

    original_score = original(batch)
    optimized_score = optimized(batch)
    synchronize(device)

    absolute = torch.abs(
        original_score - optimized_score
    )

    result = {
        "equivalence_shape_equal": (
            list(original_score.shape)
            == list(optimized_score.shape)
        ),
        "equivalence_max_abs_diff": float(
            absolute.max().item()
        ),
        "equivalence_mean_abs_diff": float(
            absolute.mean().item()
        ),
        "equivalence_allclose_rtol_1e_5_atol_1e_6": bool(
            torch.allclose(
                original_score,
                optimized_score,
                rtol=1e-5,
                atol=1e-6,
            )
        ),
    }

    del original, optimized
    del original_score, optimized_score, absolute
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)

    return result


@torch.inference_mode()
def benchmark_batch(
    model: nn.Module,
    batch_cpu: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    prefix: str,
) -> Dict[str, Any]:
    model.eval()
    batch = batch_cpu.to(
        device,
        non_blocking=True,
    )

    for _ in range(max(1, warmup)):
        _ = model(batch)
    synchronize(device)

    baseline_allocated, baseline_reserved = (
        reset_gpu_peak(device)
    )
    timings_ms: List[float] = []

    with RSSMonitor() as rss:
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            _ = model(batch)
            synchronize(device)
            timings_ms.append(
                (time.perf_counter() - start) * 1000.0
            )

    values = np.asarray(
        timings_ms,
        dtype=np.float64,
    )

    result = {
        f"{prefix}_batch_size_actual": int(
            batch_cpu.shape[0]
        ),
        f"{prefix}_latency_ms_mean": float(
            values.mean()
        ),
        f"{prefix}_latency_ms_std": float(
            values.std()
        ),
        f"{prefix}_latency_ms_p50": float(
            np.percentile(values, 50)
        ),
        f"{prefix}_latency_ms_p95": float(
            np.percentile(values, 95)
        ),
        f"{prefix}_latency_ms_p99": float(
            np.percentile(values, 99)
        ),
        f"{prefix}_cpu_baseline_rss_mib": float(
            rss.baseline_bytes / MIB
        ),
        f"{prefix}_cpu_peak_rss_mib": float(
            rss.peak_bytes / MIB
        ),
        f"{prefix}_cpu_incremental_rss_mib": float(
            max(
                0,
                rss.peak_bytes - rss.baseline_bytes,
            )
            / MIB
        ),
    }

    gpu = read_gpu_peak(
        device,
        baseline_allocated,
        baseline_reserved,
    )
    result.update(
        {
            f"{prefix}_{key}": value
            for key, value in gpu.items()
        }
    )
    return result


@torch.inference_mode()
def benchmark_full_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    win_size: int,
) -> Dict[str, Any]:
    model.eval()

    baseline_allocated, baseline_reserved = (
        reset_gpu_peak(device)
    )
    batches = 0
    windows = 0

    with RSSMonitor() as rss:
        synchronize(device)
        start = time.perf_counter()

        for input_data in loader:
            batch = input_data.float().to(
                device,
                non_blocking=True,
            )
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
        "full_test_windows_per_second": float(
            windows / max(elapsed, 1e-12)
        ),
        "full_test_points_per_second": float(
            windows * win_size
            / max(elapsed, 1e-12)
        ),
        "full_test_cpu_baseline_rss_mib": float(
            rss.baseline_bytes / MIB
        ),
        "full_test_cpu_peak_rss_mib": float(
            rss.peak_bytes / MIB
        ),
        "full_test_cpu_incremental_rss_mib": float(
            max(
                0,
                rss.peak_bytes - rss.baseline_bytes,
            )
            / MIB
        ),
    }

    gpu = read_gpu_peak(
        device,
        baseline_allocated,
        baseline_reserved,
    )
    result.update(
        {
            f"full_test_{key}": value
            for key, value in gpu.items()
        }
    )
    return result


def official_score_normalization(
    score: torch.Tensor,
) -> torch.Tensor:
    minimum = score.min(
        dim=-1,
        keepdim=True,
    ).values
    maximum = score.max(
        dim=-1,
        keepdim=True,
    ).values
    scaled = (
        score - minimum
    ) / (
        maximum - minimum + 1e-5
    )
    return torch.softmax(
        scaled,
        dim=-1,
    )


@torch.inference_mode()
def benchmark_exact_threshold(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    anormly_ratio: float,
) -> Dict[str, Any]:
    model.eval()

    baseline_allocated, baseline_reserved = (
        reset_gpu_peak(device)
    )
    collected: List[np.ndarray] = []

    with RSSMonitor() as rss:
        synchronize(device)
        start = time.perf_counter()

        for loader in (train_loader, test_loader):
            for input_data in loader:
                batch = input_data.float().to(
                    device,
                    non_blocking=True,
                )
                score = official_score_normalization(
                    model(batch)
                )
                values = (
                    score.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                collected.append(values)

        combined = np.concatenate(
            collected,
            axis=0,
        )
        threshold = float(
            np.percentile(
                combined,
                100.0 - float(anormly_ratio),
            )
        )

        synchronize(device)
        elapsed = time.perf_counter() - start

    result = {
        "exact_threshold_seconds": float(elapsed),
        "exact_threshold_samples": int(
            combined.size
        ),
        "exact_threshold_value": threshold,
        "exact_threshold_cpu_baseline_rss_mib": float(
            rss.baseline_bytes / MIB
        ),
        "exact_threshold_cpu_peak_rss_mib": float(
            rss.peak_bytes / MIB
        ),
        "exact_threshold_cpu_incremental_rss_mib": float(
            max(
                0,
                rss.peak_bytes - rss.baseline_bytes,
            )
            / MIB
        ),
    }

    gpu = read_gpu_peak(
        device,
        baseline_allocated,
        baseline_reserved,
    )
    result.update(
        {
            f"exact_threshold_{key}": value
            for key, value in gpu.items()
        }
    )
    return result


def save_results(
    json_path: Path,
    results: Dict[str, Any],
) -> None:
    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    json_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = json_path.with_suffix(".csv")
    flat = {
        key: value
        for key, value in results.items()
        if isinstance(
            value,
            (str, int, float, bool),
        )
        or value is None
    }

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(flat.keys()),
        )
        writer.writeheader()
        writer.writerow(flat)

    print()
    print("JSON结果：", json_path)
    print("CSV结果 ：", csv_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V4 streaming lag（流式滞后）优化版本基准测试"
        )
    )
    parser.add_argument(
        "--dataset",
        default="SMAP",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
    )
    parser.add_argument(
        "--file-prefix",
        default=None,
        help="默认与 dataset 相同；WADI 一般使用 WaDi。",
    )
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="auto 或具体 checkpoint 路径。",
    )
    parser.add_argument(
        "--win-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--anormly-ratio",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--include-threshold",
        action="store_true",
    )
    parser.add_argument(
        "--skip-equivalence",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    dataset = args.dataset.upper()
    file_prefix = args.file_prefix
    if file_prefix is None:
        file_prefix = "WaDi" if dataset == "WADI" else dataset

    dataset_dir = (
        project_root
        / args.dataset_root
        / dataset
    ).resolve()
    train_path = (
        dataset_dir
        / f"{file_prefix}_train.npy"
    )
    test_path = (
        dataset_dir
        / f"{file_prefix}_test.npy"
    )

    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    train_data = np.load(
        train_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    test_data = np.load(
        test_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if train_data.ndim != 2 or test_data.ndim != 2:
        raise ValueError(
            "train/test 数据都必须为二维数组。"
        )

    checkpoint = (
        find_checkpoint(project_root, dataset)
        if args.checkpoint.lower() == "auto"
        else Path(args.checkpoint).expanduser().resolve()
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    device = resolve_device(args.device)

    test_dataset = WindowDataset(
        test_data,
        args.win_size,
        mode="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    try:
        first_batch = next(
            iter(test_loader)
        ).float()
    except StopIteration as exc:
        raise RuntimeError(
            f"{dataset} 测试集没有完整窗口。"
        ) from exc

    results: Dict[str, Any] = {
        "dataset": dataset,
        "model": "v4_streaming_optimized",
        "optimization": (
            "streaming lag + score-only inference "
            "+ torch.inference_mode"
        ),
        "project_root": str(project_root),
        "dataset_dir": str(dataset_dir),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "checkpoint": str(checkpoint),
        "checkpoint_disk_bytes": int(
            checkpoint.stat().st_size
        ),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "win_size": int(args.win_size),
        "batch_size": int(args.batch_size),
        "channels": int(train_data.shape[1]),
        "anormly_ratio": float(
            args.anormly_ratio
        ),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
    }

    if not args.skip_equivalence:
        print("正在进行优化前后数值等价性检查……")
        results.update(
            equivalence_check(
                checkpoint,
                first_batch,
                device,
            )
        )
        print(
            "equivalence max abs diff:",
            results["equivalence_max_abs_diff"],
        )
        print(
            "equivalence allclose:",
            results[
                "equivalence_allclose_rtol_1e_5_atol_1e_6"
            ],
        )

    model = OptimizedV4Wrapper(
        checkpoint
    ).to(device).eval()

    results.update(
        parameter_stats(model)
    )

    print("正在测试 batch=1 在线响应时延……")
    online_batch = first_batch[:1].contiguous()
    results.update(
        benchmark_batch(
            model,
            online_batch,
            device,
            args.warmup,
            args.repeats,
            prefix="online_batch1",
        )
    )

    print(
        f"正在测试 batch={first_batch.shape[0]} "
        "批处理时延……"
    )
    results.update(
        benchmark_batch(
            model,
            first_batch,
            device,
            args.warmup,
            args.repeats,
            prefix="single_batch",
        )
    )

    print("正在测试完整测试集时间和吞吐率……")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    results.update(
        benchmark_full_test(
            model,
            test_loader,
            device,
            args.win_size,
        )
    )

    if args.include_threshold:
        print("正在测试 exact threshold（精确阈值）阶段……")
        train_loader = DataLoader(
            WindowDataset(
                train_data,
                args.win_size,
                mode="train",
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        results.update(
            benchmark_exact_threshold(
                model,
                train_loader,
                test_loader,
                device,
                args.anormly_ratio,
            )
        )

    if args.output:
        output_path = Path(
            args.output
        ).expanduser().resolve()
    else:
        output_path = (
            project_root
            / "results"
            / "optimized_benchmark"
            / (
                f"{dataset}_streaming_v4_"
                f"w{args.win_size}_b{args.batch_size}.json"
            )
        )

    save_results(
        output_path,
        results,
    )

    print()
    print("========== 关键结果 ==========")
    keys = [
        "equivalence_max_abs_diff",
        "equivalence_allclose_rtol_1e_5_atol_1e_6",
        "trainable_params",
        "serialized_state_dict_bytes",
        "online_batch1_latency_ms_mean",
        "online_batch1_latency_ms_p95",
        "single_batch_latency_ms_mean",
        "single_batch_latency_ms_p95",
        "full_test_seconds",
        "full_test_points_per_second",
        "full_test_cpu_incremental_rss_mib",
        "full_test_gpu_incremental_allocated_mib",
        "full_test_gpu_peak_allocated_mib",
        "exact_threshold_seconds",
        "exact_threshold_cpu_incremental_rss_mib",
        "exact_threshold_gpu_incremental_allocated_mib",
    ]
    for key in keys:
        if key in results:
            print(f"{key:48} : {results[key]}")


if __name__ == "__main__":
    main()
