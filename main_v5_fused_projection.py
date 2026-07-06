#!/usr/bin/env python3
from __future__ import annotations

"""
V5：低维关系投影 + 融合候选 gather + 低内存前向

设计目标
--------
1. 同时降低 GPU 激活显存和推理时延；
2. 保持 V4 的候选 lag、Straight-through Top-k、联合归一化和高斯拟合逻辑；
3. 仅增加一个很小的可训练关系投影矩阵；
4. 不覆盖原 main.py，可独立训练、测试和 benchmark（基准测试）。

核心变化
--------
- 先将输入通道 M 投影到 relation_dim=D（默认 8）：
      [B,L,M] -> [B,L,D]
- 在低维关系空间中一次性计算全部局部/全局候选；
- 使用缓存的 [L,K] 索引和 index_select，避免原实现的 [B,L,K] batch_index；
- 左右锚点顺序计算，缩短大张量生命周期；
- 不再构造 joint_affinity / joint_pdf 拼接张量；
- 用 erf 直接计算高斯局部面积；
- benchmark 时只返回 score_total，并使用 torch.inference_mode()。

注意
----
关系投影改变了模型定义，因此必须重新训练，不能加载旧 V4 checkpoint。
"""

import argparse
import csv
import gc
import io
import json
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from main import (
        AdaptiveSparseAnchorCompetitiveModelV4,
        AdaptiveSparseAnchorSolverV4,
        build_parser as build_v4_parser,
        set_seed,
    )
except ImportError as exc:
    raise RuntimeError(
        "请把本文件放到包含 main.py、solver.py、dataset/ 的项目根目录后执行。"
    ) from exc


MIB = 1024 ** 2


class AdaptiveSparseAnchorCompetitiveModelV5(
    AdaptiveSparseAnchorCompetitiveModelV4
):
    """
    V5：在低维可训练关系空间中并行计算全部候选锚点。

    参数增加量（bias=False）：
        input_c * relation_dim

    SMAP 默认 input_c=25、relation_dim=8：
        新增 200 个参数；
        总参数约 146 + 200 = 346。
    """

    def __init__(
        self,
        input_c: int,
        relation_dim: int,
        local_candidate_lags: Sequence[int],
        global_candidate_lags: Sequence[int],
        local_topk: int,
        global_topk: int,
        selector_hidden: int = 8,
        fitter_hidden: int = 8,
        selector_temperature: float = 0.5,
        similarity_tau: float = 1.0,
        sigma_min: float = 0.03,
        sigma_max: float = 1.50,
        gap_weight: float = 1.0,
    ) -> None:
        super().__init__(
            local_candidate_lags=local_candidate_lags,
            global_candidate_lags=global_candidate_lags,
            local_topk=local_topk,
            global_topk=global_topk,
            selector_hidden=selector_hidden,
            fitter_hidden=fitter_hidden,
            selector_temperature=selector_temperature,
            similarity_tau=similarity_tau,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            gap_weight=gap_weight,
        )

        if input_c <= 0:
            raise ValueError("input_c 必须大于 0。")
        if relation_dim <= 0:
            raise ValueError("relation_dim 必须大于 0。")
        if relation_dim > input_c:
            raise ValueError(
                f"relation_dim 不应大于 input_c；"
                f"实际 relation_dim={relation_dim}, input_c={input_c}"
            )

        self.input_c = int(input_c)
        self.relation_dim = int(relation_dim)

        # 无 bias 的低维关系投影。正交初始化使初始距离尺度较稳定。
        self.relation_projection = nn.Linear(
            self.input_c,
            self.relation_dim,
            bias=False,
        )
        nn.init.orthogonal_(self.relation_projection.weight)

        all_lags = torch.cat(
            [self.local_lags, self.global_lags],
            dim=0,
        )
        local_count = int(self.local_lags.numel())
        global_count = int(self.global_lags.numel())

        group_values = torch.cat(
            [
                torch.zeros(local_count, dtype=torch.float32),
                torch.ones(global_count, dtype=torch.float32),
            ],
            dim=0,
        )

        self.register_buffer(
            "all_lags",
            all_lags.clone(),
            persistent=False,
        )
        self.register_buffer(
            "all_lag_norm",
            all_lags.to(torch.float32) / self.max_lag,
            persistent=False,
        )
        self.register_buffer(
            "all_group_values",
            group_values,
            persistent=False,
        )

        # 固定窗口长度时只计算一次索引；不写入 checkpoint。
        self.register_buffer(
            "_left_index_cache",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_right_index_cache",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self._cached_length = -1

    def _time_indices(
        self,
        length: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_valid = (
            self._cached_length == int(length)
            and self._left_index_cache.numel()
            == int(length) * int(self.all_lags.numel())
            and self._left_index_cache.device == device
        )
        if not cache_valid:
            time_index = torch.arange(
                length,
                device=device,
                dtype=torch.long,
            ).view(length, 1)
            lag_index = self.all_lags.to(
                device=device,
                dtype=torch.long,
            ).view(1, -1)

            self._left_index_cache = (
                time_index - lag_index
            ).clamp_(0, length - 1)
            self._right_index_cache = (
                time_index + lag_index
            ).clamp_(0, length - 1)
            self._cached_length = int(length)

        return (
            self._left_index_cache,
            self._right_index_cache,
        )

    @staticmethod
    def _indexed_anchors(
        values: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """
        values: [B,L,D]
        index : [L,K]
        output: [B,L,K,D]

        index_select 不需要构造 [B,L,K] 的 batch_index。
        """
        batch, length = values.shape[:2]
        count = int(index.shape[1])
        tail_shape = values.shape[2:]

        gathered = values.index_select(
            dim=1,
            index=index.reshape(-1),
        )
        return gathered.view(
            batch,
            length,
            count,
            *tail_shape,
        )

    def _all_candidates(
        self,
        x: torch.Tensor,
        sketch: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch, length, channels = x.shape
        if channels != self.input_c:
            raise ValueError(
                f"输入通道数不匹配：模型 input_c={self.input_c}，"
                f"实际 channels={channels}"
            )

        left_index, right_index = self._time_indices(
            length,
            x.device,
        )

        # 低维关系空间：显著减少候选锚点的最后一维。
        relation = self.relation_projection(x)
        current_relation = relation.unsqueeze(2)

        # 左右锚点顺序计算，避免二者同时长期驻留显存。
        left_relation = self._indexed_anchors(
            relation,
            left_index,
        )
        left_distance = (
            current_relation - left_relation
        ).square().mean(dim=-1)
        del left_relation

        right_relation = self._indexed_anchors(
            relation,
            right_index,
        )
        right_distance = (
            current_relation - right_relation
        ).square().mean(dim=-1)
        del right_relation
        del current_relation
        del relation

        affinity = torch.exp(
            -left_distance / self.similarity_tau
        )
        affinity.add_(
            torch.exp(
                -right_distance / self.similarity_tau
            )
        )
        del left_distance, right_distance

        current_sketch = sketch.unsqueeze(2)
        left_sketch = self._indexed_anchors(
            sketch,
            left_index,
        )
        delta = torch.abs(
            current_sketch - left_sketch
        )
        del left_sketch

        right_sketch = self._indexed_anchors(
            sketch,
            right_index,
        )
        delta.add_(
            torch.abs(
                current_sketch - right_sketch
            )
        )
        delta.mul_(0.5)
        del right_sketch, current_sketch

        count = int(self.all_lags.numel())
        lag_feature = self.all_lag_norm.to(
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, count, 1).expand(
            batch,
            length,
            -1,
            -1,
        )
        group_feature = self.all_group_values.to(
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, count, 1).expand(
            batch,
            length,
            -1,
            -1,
        )

        selector_features = torch.cat(
            [delta, lag_feature, group_feature],
            dim=-1,
        )
        logits = self.selector(selector_features)
        del selector_features, delta

        local_count = int(self.local_lags.numel())
        local = {
            "affinity": affinity[..., :local_count],
            "logits": logits[..., :local_count],
            "lag_norm": lag_feature[
                ..., :local_count, 0
            ],
        }
        global_ = {
            "affinity": affinity[..., local_count:],
            "logits": logits[..., local_count:],
            "lag_norm": lag_feature[
                ..., local_count:, 0
            ],
        }
        return local, global_

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = True,
    ):
        if x.ndim != 3:
            raise ValueError("输入必须为 [B,L,M]。")

        sketch = self._sequence_sketch(x)
        local, global_ = self._all_candidates(
            x,
            sketch,
        )
        del sketch

        (
            local_gate,
            local_weights,
            local_index,
        ) = self._straight_through_topk(
            local["logits"],
            self.local_topk,
        )
        (
            global_gate,
            global_weights,
            global_index,
        ) = self._straight_through_topk(
            global_["logits"],
            self.global_topk,
        )

        # 等价于 [center, local, global] 联合归一化，但不做 cat。
        affinity_sum = (
            1.0
            + local["affinity"].sum(
                dim=-1,
                keepdim=True,
            )
            + global_["affinity"].sum(
                dim=-1,
                keepdim=True,
            )
        ).clamp_min_(1e-8)

        target_center = affinity_sum.reciprocal()
        target_local = (
            local["affinity"] / affinity_sum
        )
        target_global = (
            global_["affinity"] / affinity_sum
        )

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
        center_pdf = (
            1.0 / (sqrt_two_pi * sigma)
        ).unsqueeze(-1)
        sigma_expanded = sigma.unsqueeze(-1)

        local_pdf = (
            2.0
            / (
                sqrt_two_pi
                * sigma_expanded
            )
            * torch.exp(
                -0.5
                * (
                    local["lag_norm"]
                    / sigma_expanded
                ).square()
            )
        )
        global_pdf = (
            2.0
            / (
                sqrt_two_pi
                * sigma_expanded
            )
            * torch.exp(
                -0.5
                * (
                    global_["lag_norm"]
                    / sigma_expanded
                ).square()
            )
        )

        # 等价于 joint_pdf / joint_pdf.sum(...)，但不做 cat。
        pdf_sum = (
            center_pdf
            + local_pdf.sum(
                dim=-1,
                keepdim=True,
            )
            + global_pdf.sum(
                dim=-1,
                keepdim=True,
            )
        ).clamp_min_(1e-8)

        pred_center = center_pdf / pdf_sum
        pred_local = local_pdf / pdf_sum
        pred_global = global_pdf / pdf_sum

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

        # Phi(z)-Phi(-z) = erf(z/sqrt(2))，避免 distributions.Normal 开销。
        edge_z = (
            sigma.new_tensor(self.local_edge)
            / sigma
        )
        pred_local_area = torch.erf(
            edge_z / math.sqrt(2.0)
        )
        pred_global_area = 1.0 - pred_local_area

        target_local_area = (
            target_center.squeeze(-1)
            + target_local.sum(dim=-1)
        )
        target_global_area = target_global.sum(
            dim=-1
        )
        local_area_error = (
            pred_local_area - target_local_area
        ).square()
        global_area_error = (
            pred_global_area - target_global_area
        ).square()
        area_error = (
            local_area_error + global_area_error
        )

        score_gap = torch.abs(
            local_fit - global_fit
        )
        score_total = local_fit + global_fit
        score_combined = (
            score_total
            + self.gap_weight * score_gap
        )

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
                local["logits"]
                / self.selector_temperature,
                dim=-1,
            ),
            "global_probabilities": torch.softmax(
                global_["logits"]
                / self.selector_temperature,
                dim=-1,
            ),
            "local_gate": local_gate,
            "global_gate": global_gate,
            "local_selected_index": local_index,
            "global_selected_index": global_index,
        }
        return score_combined, details


class AdaptiveSparseAnchorSolverV5(
    AdaptiveSparseAnchorSolverV4
):
    def build_model(self) -> None:
        self.model = AdaptiveSparseAnchorCompetitiveModelV5(
            input_c=self.input_c,
            relation_dim=self.relation_dim,
            local_candidate_lags=self.local_candidate_lags,
            global_candidate_lags=self.global_candidate_lags,
            local_topk=self.local_topk,
            global_topk=self.global_topk,
            selector_hidden=self.selector_hidden,
            fitter_hidden=self.fitter_hidden,
            selector_temperature=self.selector_temperature,
            similarity_tau=self.similarity_tau,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            gap_weight=self.gap_weight,
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        local_tag = "-".join(
            map(str, self.local_candidate_lags)
        )
        global_tag = "-".join(
            map(str, self.global_candidate_lags)
        )
        self.checkpoint_path = os.path.join(
            self.model_save_path,
            (
                f"{self.dataset}_adaptive_anchor_v5"
                f"_rd{self.relation_dim}"
                f"_l{local_tag}_g{global_tag}"
                f"_kl{self.local_topk}"
                f"_kg{self.global_topk}.pt"
            ),
        )

    def _save_checkpoint(self) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": {
                    "model_version": "v5_fused_projection",
                    "dataset": self.dataset,
                    "input_c": self.input_c,
                    "relation_dim": self.relation_dim,
                    "local_candidate_lags": self.local_candidate_lags,
                    "global_candidate_lags": self.global_candidate_lags,
                    "local_topk": self.local_topk,
                    "global_topk": self.global_topk,
                    "selector_hidden": self.selector_hidden,
                    "fitter_hidden": self.fitter_hidden,
                    "selector_temperature": self.selector_temperature,
                    "similarity_tau": self.similarity_tau,
                    "sigma_min": self.sigma_min,
                    "sigma_max": self.sigma_max,
                    "gap_weight": self.gap_weight,
                },
            },
            self.checkpoint_path,
        )


class RSSMonitor:
    """当前 Python 进程的 RSS（常驻内存）采样器。"""

    def __init__(
        self,
        interval: float = 0.01,
    ) -> None:
        self.interval = float(interval)
        self.baseline_bytes = self.current_rss_bytes()
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

    @staticmethod
    def current_rss_bytes() -> int:
        status_path = Path("/proc/self/status")
        if status_path.exists():
            for line in status_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines():
                if line.startswith("VmRSS:"):
                    return int(
                        line.split()[1]
                    ) * 1024
        try:
            import psutil

            return int(
                psutil.Process(
                    os.getpid()
                ).memory_info().rss
            )
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

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self._stop.set()
        self._thread.join()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_gpu_peak(
    device: torch.device,
) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0

    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)

    allocated = (
        torch.cuda.memory_allocated(device)
        / MIB
    )
    reserved = (
        torch.cuda.memory_reserved(device)
        / MIB
    )
    torch.cuda.reset_peak_memory_stats(device)
    return float(allocated), float(reserved)


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
        torch.cuda.max_memory_allocated(device)
        / MIB
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device)
        / MIB
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


def parameter_stats(
    model: nn.Module,
) -> Dict[str, int]:
    stream = io.BytesIO()
    torch.save(
        model.state_dict(),
        stream,
    )
    parameters = list(
        model.parameters()
    )
    buffers = list(
        model.buffers()
    )
    return {
        "trainable_params": int(
            sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad
            )
        ),
        "total_params": int(
            sum(
                parameter.numel()
                for parameter in parameters
            )
        ),
        "parameter_tensor_bytes": int(
            sum(
                parameter.numel()
                * parameter.element_size()
                for parameter in parameters
            )
        ),
        "buffer_elements": int(
            sum(
                buffer.numel()
                for buffer in buffers
            )
        ),
        "buffer_tensor_bytes": int(
            sum(
                buffer.numel()
                * buffer.element_size()
                for buffer in buffers
            )
        ),
        "serialized_state_dict_bytes": int(
            stream.tell()
        ),
    }


@torch.inference_mode()
def benchmark_batch(
    runner: AdaptiveSparseAnchorSolverV5,
    batch_cpu: torch.Tensor,
    warmup: int,
    repeats: int,
    prefix: str,
) -> Dict[str, Any]:
    device = runner.device
    model = runner.model
    model.eval()

    batch = runner._prepare_input(
        batch_cpu
    )

    for _ in range(
        max(1, warmup)
    ):
        _ = model(
            batch,
            return_details=False,
        )
    synchronize(device)

    (
        baseline_allocated,
        baseline_reserved,
    ) = reset_gpu_peak(device)

    timings: List[float] = []
    with RSSMonitor() as rss:
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            _ = model(
                batch,
                return_details=False,
            )
            synchronize(device)
            timings.append(
                (
                    time.perf_counter()
                    - start
                )
                * 1000.0
            )

    values = np.asarray(
        timings,
        dtype=np.float64,
    )
    result: Dict[str, Any] = {
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
                rss.peak_bytes
                - rss.baseline_bytes,
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
    runner: AdaptiveSparseAnchorSolverV5,
) -> Dict[str, Any]:
    device = runner.device
    model = runner.model
    model.eval()

    (
        baseline_allocated,
        baseline_reserved,
    ) = reset_gpu_peak(device)

    batches = 0
    windows = 0
    with RSSMonitor() as rss:
        synchronize(device)
        start = time.perf_counter()

        for input_data, _ in runner.thre_loader:
            batch = runner._prepare_input(
                input_data
            )
            _ = model(
                batch,
                return_details=False,
            )
            batches += 1
            windows += int(
                batch.shape[0]
            )

        synchronize(device)
        elapsed = (
            time.perf_counter()
            - start
        )

    result: Dict[str, Any] = {
        "full_test_seconds": float(
            elapsed
        ),
        "full_test_batches": int(
            batches
        ),
        "full_test_windows": int(
            windows
        ),
        "full_test_points": int(
            windows * runner.win_size
        ),
        "full_test_windows_per_second": float(
            windows
            / max(elapsed, 1e-12)
        ),
        "full_test_points_per_second": float(
            windows
            * runner.win_size
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
                rss.peak_bytes
                - rss.baseline_bytes,
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


def normalize_official(
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
    runner: AdaptiveSparseAnchorSolverV5,
) -> Dict[str, Any]:
    device = runner.device
    model = runner.model
    model.eval()

    (
        baseline_allocated,
        baseline_reserved,
    ) = reset_gpu_peak(device)

    collected: List[np.ndarray] = []
    with RSSMonitor() as rss:
        synchronize(device)
        start = time.perf_counter()

        for loader in (
            runner.train_loader,
            runner.thre_loader,
        ):
            for input_data, _ in loader:
                batch = runner._prepare_input(
                    input_data
                )
                score = model(
                    batch,
                    return_details=False,
                )
                score = normalize_official(
                    score
                )
                collected.append(
                    score.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

        combined = np.concatenate(
            collected,
            axis=0,
        )
        threshold = float(
            np.percentile(
                combined,
                100.0
                - float(
                    runner.anormly_ratio
                ),
            )
        )

        synchronize(device)
        elapsed = (
            time.perf_counter()
            - start
        )

    result: Dict[str, Any] = {
        "exact_threshold_seconds": float(
            elapsed
        ),
        "exact_threshold_samples": int(
            combined.size
        ),
        "exact_threshold_value": float(
            threshold
        ),
        "exact_threshold_cpu_baseline_rss_mib": float(
            rss.baseline_bytes / MIB
        ),
        "exact_threshold_cpu_peak_rss_mib": float(
            rss.peak_bytes / MIB
        ),
        "exact_threshold_cpu_incremental_rss_mib": float(
            max(
                0,
                rss.peak_bytes
                - rss.baseline_bytes,
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


def save_benchmark(
    output_path: Path,
    results: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = output_path.with_suffix(
        ".csv"
    )
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
            fieldnames=list(
                flat.keys()
            ),
        )
        writer.writeheader()
        writer.writerow(flat)

    print("Benchmark JSON:", output_path)
    print("Benchmark CSV :", csv_path)


def run_benchmark(
    runner: AdaptiveSparseAnchorSolverV5,
    args: argparse.Namespace,
) -> None:
    runner.load_checkpoint()
    runner.model.eval()

    try:
        first_batch, _ = next(
            iter(
                runner.thre_loader
            )
        )
    except StopIteration as exc:
        raise RuntimeError(
            "测试数据没有可用 batch。"
        ) from exc

    results: Dict[str, Any] = {
        "dataset": args.dataset,
        "model": "adaptive_anchor_v5_fused_projection",
        "optimization": (
            "learned low-dimensional relation projection"
            " + fused local/global candidates"
            " + cached index_select"
            " + score-only inference"
            " + erf area"
        ),
        "checkpoint": str(
            Path(
                runner.checkpoint_path
            ).resolve()
        ),
        "checkpoint_disk_bytes": int(
            Path(
                runner.checkpoint_path
            ).stat().st_size
        ),
        "device": str(
            runner.device
        ),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if runner.device.type
            == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "win_size": int(
            runner.win_size
        ),
        "batch_size": int(
            runner.batch_size
        ),
        "channels": int(
            runner.input_c
        ),
        "relation_dim": int(
            runner.relation_dim
        ),
        "anormly_ratio": float(
            runner.anormly_ratio
        ),
        "benchmark_warmup": int(
            args.benchmark_warmup
        ),
        "benchmark_repeats": int(
            args.benchmark_repeats
        ),
    }
    results.update(
        parameter_stats(
            runner.model
        )
    )

    print("Benchmark: batch=1 在线时延")
    results.update(
        benchmark_batch(
            runner,
            first_batch[:1],
            args.benchmark_warmup,
            args.benchmark_repeats,
            prefix="online_batch1",
        )
    )

    print(
        "Benchmark: 当前 batch 批处理时延"
    )
    results.update(
        benchmark_batch(
            runner,
            first_batch,
            args.benchmark_warmup,
            args.benchmark_repeats,
            prefix="single_batch",
        )
    )

    print("Benchmark: 完整测试集")
    results.update(
        benchmark_full_test(
            runner
        )
    )

    if args.benchmark_threshold:
        print(
            "Benchmark: exact threshold"
        )
        results.update(
            benchmark_exact_threshold(
                runner
            )
        )

    if args.benchmark_output:
        output_path = Path(
            args.benchmark_output
        ).expanduser().resolve()
    else:
        output_path = (
            Path(__file__).resolve().parent
            / "results"
            / "v5_benchmark"
            / (
                f"{args.dataset}_v5"
                f"_rd{args.relation_dim}"
                f"_w{args.win_size}"
                f"_b{args.batch_size}.json"
            )
        )

    save_benchmark(
        output_path,
        results,
    )

    print()
    print("========== V5 关键结果 ==========")
    keys = [
        "trainable_params",
        "parameter_tensor_bytes",
        "serialized_state_dict_bytes",
        "online_batch1_latency_ms_mean",
        "online_batch1_latency_ms_p95",
        "single_batch_latency_ms_mean",
        "single_batch_latency_ms_p95",
        "full_test_seconds",
        "full_test_points_per_second",
        "full_test_cpu_incremental_rss_mib",
        "full_test_gpu_peak_allocated_mib",
        "full_test_gpu_incremental_allocated_mib",
        "exact_threshold_seconds",
        "exact_threshold_cpu_incremental_rss_mib",
        "exact_threshold_gpu_incremental_allocated_mib",
    ]
    for key in keys:
        if key in results:
            print(
                f"{key:52} "
                f"{results[key]}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = build_v4_parser()

    # 扩展原 mode，不重复添加参数。
    for action in parser._actions:
        if action.dest == "mode":
            action.choices = [
                "train",
                "test",
                "benchmark",
            ]
            break

    parser.description = (
        "Adaptive Sparse Temporal Anchor "
        "Competitive Fitting V5 "
        "(Fused Projection)"
    )
    parser.add_argument(
        "--relation_dim",
        type=int,
        default=8,
        help=(
            "候选关系计算的低维通道数；"
            "SMAP 建议先使用 8。"
        ),
    )
    parser.add_argument(
        "--benchmark_warmup",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--benchmark_repeats",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--benchmark_threshold",
        action="store_true",
        help=(
            "benchmark 时同时测试 exact threshold 阶段。"
        ),
    )
    parser.add_argument(
        "--benchmark_output",
        type=str,
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    print(
        "========== Adaptive Sparse Anchor V5 "
        "(Fused Projection) =========="
    )
    runner = AdaptiveSparseAnchorSolverV5(
        vars(args)
    )
    trainable = sum(
        parameter.numel()
        for parameter
        in runner.model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters : {trainable:,}"
    )
    print(
        f"Relation projection  : "
        f"{args.input_c} -> "
        f"{args.relation_dim}"
    )
    print(
        "Candidate anchors    : "
        f"local "
        f"{len(args.local_candidate_lags)}"
        f"/{args.local_topk}, "
        f"global "
        f"{len(args.global_candidate_lags)}"
        f"/{args.global_topk}"
    )
    print(
        "Candidate execution  : fused + cached index_select"
    )
    print(
        "Area computation     : erf"
    )
    print(
        "Checkpoint           : "
        f"{runner.checkpoint_path}"
    )

    if args.mode == "train":
        runner.train()
        runner.test()
    elif args.mode == "test":
        runner.load_checkpoint()
        runner.test()
    elif args.mode == "benchmark":
        run_benchmark(
            runner,
            args,
        )
    else:
        raise ValueError(
            f"未知 mode：{args.mode}"
        )


if __name__ == "__main__":
    main()
