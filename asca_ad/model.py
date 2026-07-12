from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

try:
    from metrics.metrics import combine_all_evaluation_scores
    from solver import Solver, adjust_learning_rate
except ImportError:  # 允许脱离完整仓库执行语法和前向测试
    combine_all_evaluation_scores = None
    Solver = object  # type: ignore[assignment,misc]
    adjust_learning_rate = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReservoirQuantile:
    def __init__(self, capacity: int = 50000, seed: int = 42) -> None:
        if capacity <= 0:
            raise ValueError("quantile_buffer 必须大于 0。")
        self.capacity = int(capacity)
        self.buffer = np.empty(self.capacity, dtype=np.float64)
        self.size = 0
        self.count = 0
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        for value in values:
            if not np.isfinite(value):
                continue
            self.count += 1
            if self.size < self.capacity:
                self.buffer[self.size] = value
                self.size += 1
            else:
                index = int(self.rng.integers(0, self.count))
                if index < self.capacity:
                    self.buffer[index] = value

    def percentile(self, percentile: float) -> float:
        if self.size == 0:
            raise RuntimeError("分位数估计器为空。")
        return float(np.percentile(self.buffer[: self.size], percentile))


class SharedAnchorSelector(nn.Module):
    """局部和全局候选 lag 共用的轻量选择器。"""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class SharedGaussianFitter(nn.Module):
    """局部和全局关系共用的单 sigma 高斯拟合器。"""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 8,
        sigma_min: float = 0.03,
        sigma_max: float = 1.50,
    ) -> None:
        super().__init__()
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("必须满足 0 < sigma_min < sigma_max。")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features).squeeze(-1)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)


class AdaptiveSparseAnchorCompetitiveModelV4(nn.Module):
    """可微稀疏时间锚点 + 联合关系归一化 + 共享单高斯竞争拟合。

    每个正 lag 表示一对对称时间锚点 (t-lag, t+lag)，边界采用复制填充。
    训练时使用 straight-through Top-k，使前向保持稀疏、反向可更新所有候选。
    """

    def __init__(
        self,
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
        gather_backend: str = "shifted_stack",
    ) -> None:
        super().__init__()
        local_lags = sorted(set(int(v) for v in local_candidate_lags))
        global_lags = sorted(set(int(v) for v in global_candidate_lags))
        if not local_lags or not global_lags:
            raise ValueError("局部和全局候选 lag 不能为空。")
        if min(local_lags + global_lags) <= 0:
            raise ValueError("所有 lag 必须为正整数。")
        if set(local_lags).intersection(global_lags):
            raise ValueError("局部和全局候选 lag 不应重叠。")
        if not 1 <= local_topk <= len(local_lags):
            raise ValueError("local_topk 超出候选数量。")
        if not 1 <= global_topk <= len(global_lags):
            raise ValueError("global_topk 超出候选数量。")
        if selector_temperature <= 0 or similarity_tau <= 0:
            raise ValueError("selector_temperature 和 similarity_tau 必须大于 0。")
        if gap_weight < 0:
            raise ValueError("gap_weight 不能为负数。")

        self.local_topk = int(local_topk)
        self.global_topk = int(global_topk)
        self.selector_temperature = float(selector_temperature)
        self.similarity_tau = float(similarity_tau)
        self.gap_weight = float(gap_weight)
        self.max_lag = float(max(local_lags + global_lags))
        self.local_edge = float(max(local_lags)) / self.max_lag

        self.register_buffer("local_lags", torch.tensor(local_lags, dtype=torch.long))
        self.register_buffer("global_lags", torch.tensor(global_lags, dtype=torch.long))

        # 4个通道摘要差异 + 标准化lag + 局部/全局标记。
        self.selector = SharedAnchorSelector(input_dim=6, hidden_dim=selector_hidden)
        # 局部和全局各4个可微摘要，共8维。
        self.fitter = SharedGaussianFitter(
            input_dim=8,
            hidden_dim=fitter_hidden,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        # 推理 gather 后端。
        # advanced_index: 原始高级索引实现，完全保留旧路径。
        # cached_index  : 缓存 index tensor，减少重复索引构造开销。
        # shifted_stack : 用规则 shift/stack 替代高级索引，通常更硬件友好。
        self.gather_backend = "advanced_index"
        self._gather_index_cache: Dict[Tuple[str, int, int, Tuple[int, ...]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.set_gather_backend(gather_backend)

    @staticmethod
    def _sequence_sketch(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        rms = torch.sqrt(x.square().mean(dim=-1) + 1e-8)
        mean_abs = x.abs().mean(dim=-1)
        return torch.stack([mean, std, rms, mean_abs], dim=-1)

    def set_gather_backend(self, backend: str) -> None:
        """切换 V4 的对称锚点 gather 实现，不改变参数和数学定义。"""
        valid = {"advanced_index", "cached_index", "shifted_stack"}
        if backend not in valid:
            raise ValueError(f"未知 gather_backend={backend!r}，可选：{sorted(valid)}")
        self.gather_backend = backend

    def clear_gather_cache(self) -> None:
        """清空 cached_index 后端使用的索引缓存。"""
        self._gather_index_cache.clear()

    @staticmethod
    def _lags_to_tuple(lags: torch.Tensor) -> Tuple[int, ...]:
        # lags 很短；转为 tuple 便于缓存和 shifted_stack 循环。
        return tuple(int(v) for v in lags.detach().cpu().tolist())

    @staticmethod
    def _shift_left_replicate(x: torch.Tensor, lag: int) -> torch.Tensor:
        """left[t] = x[max(t-lag, 0)]，边界复制。"""
        if lag <= 0:
            return x
        batch = x.shape[0]
        tail_shape = tuple(x.shape[2:])
        pad = x[:, :1, ...].expand(batch, lag, *tail_shape)
        return torch.cat([pad, x[:, :-lag, ...]], dim=1)

    @staticmethod
    def _shift_right_replicate(x: torch.Tensor, lag: int) -> torch.Tensor:
        """right[t] = x[min(t+lag, L-1)]，边界复制。"""
        if lag <= 0:
            return x
        batch = x.shape[0]
        tail_shape = tuple(x.shape[2:])
        pad = x[:, -1:, ...].expand(batch, lag, *tail_shape)
        return torch.cat([x[:, lag:, ...], pad], dim=1)

    def _symmetric_gather_shifted_stack(
        self,
        x: torch.Tensor,
        lags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """用规则 shift + stack 构造对称锚点，避免高级索引。"""
        lag_values = self._lags_to_tuple(lags)
        left_parts = [self._shift_left_replicate(x, lag) for lag in lag_values]
        right_parts = [self._shift_right_replicate(x, lag) for lag in lag_values]
        return torch.stack(left_parts, dim=2), torch.stack(right_parts, dim=2)

    def _get_cached_gather_indices(
        self,
        batch: int,
        length: int,
        lags: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lag_values = self._lags_to_tuple(lags)
        key = (str(device), int(batch), int(length), lag_values)
        cached = self._gather_index_cache.get(key)
        if cached is not None:
            return cached

        lag_tensor = torch.tensor(lag_values, dtype=torch.long, device=device)
        time = torch.arange(length, device=device).view(1, length, 1)
        lag_view = lag_tensor.view(1, 1, -1)
        left_index = (time - lag_view).clamp(0, length - 1).expand(batch, -1, -1)
        right_index = (time + lag_view).clamp(0, length - 1).expand(batch, -1, -1)
        batch_index = torch.arange(batch, device=device).view(batch, 1, 1).expand_as(left_index)
        cached = (batch_index, left_index, right_index)
        self._gather_index_cache[key] = cached
        return cached

    def _symmetric_gather_advanced_index(
        self,
        x: torch.Tensor,
        lags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """原始高级索引 gather 实现。"""
        batch, length = x.shape[:2]
        time = torch.arange(length, device=x.device).view(1, length, 1)
        lag_view = lags.view(1, 1, -1)
        left_index = (time - lag_view).clamp(0, length - 1).expand(batch, -1, -1)
        right_index = (time + lag_view).clamp(0, length - 1).expand(batch, -1, -1)
        batch_index = torch.arange(batch, device=x.device).view(batch, 1, 1).expand_as(left_index)
        return x[batch_index, left_index], x[batch_index, right_index]

    def _symmetric_gather_cached_index(
        self,
        x: torch.Tensor,
        lags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """缓存高级索引 tensor，减少每次 forward 重复构造 index 的开销。"""
        batch, length = x.shape[:2]
        batch_index, left_index, right_index = self._get_cached_gather_indices(
            batch=batch,
            length=length,
            lags=lags,
            device=x.device,
        )
        return x[batch_index, left_index], x[batch_index, right_index]

    def _symmetric_gather(self, x: torch.Tensor, lags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回复制边界后的前向/后向锚点，形状均为 [B,L,K,...]。"""
        if self.gather_backend == "advanced_index":
            return self._symmetric_gather_advanced_index(x, lags)
        if self.gather_backend == "cached_index":
            return self._symmetric_gather_cached_index(x, lags)
        if self.gather_backend == "shifted_stack":
            return self._symmetric_gather_shifted_stack(x, lags)
        raise RuntimeError(f"非法 gather_backend: {self.gather_backend!r}")

    def _candidate_group(
        self,
        x: torch.Tensor,
        sketch: torch.Tensor,
        lags: torch.Tensor,
        group_value: float,
    ) -> Dict[str, torch.Tensor]:
        left_x, right_x = self._symmetric_gather(x, lags)
        left_sketch, right_sketch = self._symmetric_gather(sketch, lags)

        current_x = x.unsqueeze(2)
        left_distance = (current_x - left_x).square().mean(dim=-1)
        right_distance = (current_x - right_x).square().mean(dim=-1)

        # 每个 lag 表示一对对称锚点，其亲和度为两侧亲和度之和。
        affinity = torch.exp(-left_distance / self.similarity_tau) + torch.exp(
            -right_distance / self.similarity_tau
        )

        current_sketch = sketch.unsqueeze(2)
        delta = 0.5 * (
            torch.abs(current_sketch - left_sketch)
            + torch.abs(current_sketch - right_sketch)
        )
        batch, length, count, _ = delta.shape
        lag_feature = (
            lags.to(x.dtype).view(1, 1, count, 1) / self.max_lag
        ).expand(batch, length, -1, -1)
        group_feature = x.new_full((batch, length, count, 1), float(group_value))
        selector_features = torch.cat([delta, lag_feature, group_feature], dim=-1)
        logits = self.selector(selector_features)
        return {
            "affinity": affinity,
            "logits": logits,
            "lag_norm": lag_feature.squeeze(-1),
        }

    def _straight_through_topk(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(logits / self.selector_temperature, dim=-1)
        selected_index = torch.topk(logits, k=topk, dim=-1).indices
        hard_gate = torch.zeros_like(logits).scatter(-1, selected_index, 1.0)
        if self.training:
            soft_gate = probabilities * float(topk)
            gate = hard_gate + soft_gate - soft_gate.detach()
        else:
            gate = hard_gate
        weights = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return gate, weights, selected_index

    @staticmethod
    def _weighted_summary(
        target: torch.Tensor,
        lag_norm: torch.Tensor,
        gate: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        mean = torch.sum(weights * target, dim=-1)
        variance = torch.sum(weights * (target - mean.unsqueeze(-1)).square(), dim=-1)
        std = torch.sqrt(variance + 1e-8)
        mean_lag = torch.sum(weights * lag_norm, dim=-1)
        selected_mass = torch.sum(gate * target, dim=-1)
        return torch.stack([mean, std, mean_lag, selected_mass], dim=-1)

    @staticmethod
    def _weighted_fit_error(
        target: torch.Tensor,
        prediction: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(weights * (target - prediction).square(), dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError("输入必须为 [B,L,M]。")
        sketch = self._sequence_sketch(x)
        local = self._candidate_group(x, sketch, self.local_lags, group_value=0.0)
        global_ = self._candidate_group(x, sketch, self.global_lags, group_value=1.0)

        local_gate, local_weights, local_index = self._straight_through_topk(
            local["logits"], self.local_topk
        )
        global_gate, global_weights, global_index = self._straight_through_topk(
            global_["logits"], self.global_topk
        )

        # 当前点固定为高斯中心；局部、全局所有候选先联合归一化。
        center_affinity = torch.ones_like(local["affinity"][..., :1])
        joint_affinity = torch.cat(
            [center_affinity, local["affinity"], global_["affinity"]], dim=-1
        )
        joint_target = joint_affinity / joint_affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        local_count = self.local_lags.numel()
        target_center = joint_target[..., :1]
        target_local = joint_target[..., 1 : 1 + local_count]
        target_global = joint_target[..., 1 + local_count :]

        local_summary = self._weighted_summary(
            target_local, local["lag_norm"], local_gate, local_weights
        )
        global_summary = self._weighted_summary(
            target_global, global_["lag_norm"], global_gate, global_weights
        )
        sigma = self.fitter(torch.cat([local_summary, global_summary], dim=-1))

        sqrt_two_pi = math.sqrt(2.0 * math.pi)
        center_pdf = 1.0 / (sqrt_two_pi * sigma).unsqueeze(-1)
        local_pdf = 2.0 / (sqrt_two_pi * sigma.unsqueeze(-1)) * torch.exp(
            -0.5 * (local["lag_norm"] / sigma.unsqueeze(-1)).square()
        )
        global_pdf = 2.0 / (sqrt_two_pi * sigma.unsqueeze(-1)) * torch.exp(
            -0.5 * (global_["lag_norm"] / sigma.unsqueeze(-1)).square()
        )
        joint_pdf = torch.cat([center_pdf, local_pdf, global_pdf], dim=-1)
        joint_gaussian = joint_pdf / joint_pdf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pred_center = joint_gaussian[..., :1]
        pred_local = joint_gaussian[..., 1 : 1 + local_count]
        pred_global = joint_gaussian[..., 1 + local_count :]

        # 中心点固定加入局部拟合；其权重与每个被选局部 lag 相同。
        center_weight = local_weights.new_full(
            (*local_weights.shape[:-1], 1), 1.0 / float(self.local_topk + 1)
        )
        local_pair_weights = local_weights * (float(self.local_topk) / float(self.local_topk + 1))
        local_fit = (
            center_weight.squeeze(-1) * (target_center.squeeze(-1) - pred_center.squeeze(-1)).square()
            + self._weighted_fit_error(target_local, pred_local, local_pair_weights)
        )
        global_fit = self._weighted_fit_error(target_global, pred_global, global_weights)

        # CDF面积约束：高斯在局部时间区域的质量匹配真实关系的局部质量。
        standard_normal = torch.distributions.Normal(
            torch.zeros_like(sigma), torch.ones_like(sigma)
        )
        edge_z = sigma.new_tensor(self.local_edge) / sigma
        pred_local_area = standard_normal.cdf(edge_z) - standard_normal.cdf(-edge_z)
        pred_global_area = 1.0 - pred_local_area
        target_local_area = target_center.squeeze(-1) + target_local.sum(dim=-1)
        target_global_area = target_global.sum(dim=-1)
        local_area_error = (pred_local_area - target_local_area).square()
        global_area_error = (pred_global_area - target_global_area).square()
        area_error = local_area_error + global_area_error

        score_gap = torch.abs(local_fit - global_fit)
        score_total = local_fit + global_fit
        score_combined = score_total + self.gap_weight * score_gap

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
                local["logits"] / self.selector_temperature, dim=-1
            ),
            "global_probabilities": torch.softmax(
                global_["logits"] / self.selector_temperature, dim=-1
            ),
            "local_gate": local_gate,
            "global_gate": global_gate,
            "local_selected_index": local_index,
            "global_selected_index": global_index,
        }
        return score_combined, details


class AdaptiveSparseAnchorSolverV4(Solver):  # type: ignore[misc]
    def build_model(self) -> None:
        self.model = AdaptiveSparseAnchorCompetitiveModelV4(
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
            gather_backend=getattr(self, "gather_backend", "shifted_stack"),
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def __init__(self, config: dict) -> None:
        if Solver is object:
            raise RuntimeError("请将脚本放到完整 PPLAD 项目根目录后运行。")
        super().__init__(config)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if max(self.local_candidate_lags + self.global_candidate_lags) >= self.win_size:
            raise ValueError("最大候选 lag 必须小于 win_size。")
        if self.primary_score not in self.score_modes:
            raise ValueError("primary_score 必须包含在 score_modes 中。")

        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        local_tag = "-".join(map(str, self.local_candidate_lags))
        global_tag = "-".join(map(str, self.global_candidate_lags))
        self.checkpoint_path = os.path.join(
            self.model_save_path,
            f"{self.dataset}_adaptive_anchor_v4_l{local_tag}_g{global_tag}"
            f"_kl{self.local_topk}_kg{self.global_topk}.pt",
        )

    @staticmethod
    def _instance_normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True).detach()
        var = x.var(dim=1, keepdim=True, unbiased=False).detach()
        return (x - mean) / torch.sqrt(var + eps)

    def _prepare_input(self, input_data: torch.Tensor) -> torch.Tensor:
        x = input_data.float().to(self.device, non_blocking=True)
        if self.relation_input == "instance":
            return self._instance_normalize(x)
        if self.relation_input == "standardized":
            return x
        raise ValueError(f"未知 relation_input：{self.relation_input}")

    def _forward_batch(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.model(self._prepare_input(input_data))

    @staticmethod
    def _coverage_loss(gate: torch.Tensor, topk: int) -> torch.Tensor:
        expected = float(topk) / float(gate.shape[-1])
        usage = gate.mean(dim=(0, 1))
        return (usage - expected).square().mean()

    def _training_loss(self, details: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        local_fit = details["local_fit"].mean()
        global_fit = details["global_fit"].mean()
        fit_loss = local_fit + global_fit
        area_loss = details["area_error"].mean()
        coverage_loss = self._coverage_loss(details["local_gate"], self.local_topk)
        coverage_loss = coverage_loss + self._coverage_loss(
            details["global_gate"], self.global_topk
        )
        total = (
            fit_loss
            + self.area_weight * area_loss
            + self.selector_balance_weight * coverage_loss
        )
        return total, {
            "fit": fit_loss,
            "local_fit": local_fit,
            "global_fit": global_fit,
            "gap": details["score_gap"].mean(),
            "area": area_loss,
            "coverage": coverage_loss,
            "sigma": details["sigma"].mean(),
        }

    def _save_checkpoint(self) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": {
                    "dataset": self.dataset,
                    "input_c": self.input_c,
                    "local_candidate_lags": self.local_candidate_lags,
                    "global_candidate_lags": self.global_candidate_lags,
                    "local_topk": self.local_topk,
                    "global_topk": self.global_topk,
                    "selector_hidden": self.selector_hidden,
                    "fitter_hidden": self.fitter_hidden,
                    "gather_backend": getattr(self.model, "gather_backend", "unknown"),
                },
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"未找到检查点：{self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint: {self.checkpoint_path}")

    @torch.no_grad()
    def _anchor_occupancy(self) -> Tuple[np.ndarray, np.ndarray]:
        local_count = np.zeros(len(self.local_candidate_lags), dtype=np.int64)
        global_count = np.zeros(len(self.global_candidate_lags), dtype=np.int64)
        self.model.eval()
        for input_data, _ in self.train_loader:
            _, details = self._forward_batch(input_data)
            local_index = details["local_selected_index"].detach().cpu().numpy()
            global_index = details["global_selected_index"].detach().cpu().numpy()
            local_count += np.bincount(local_index.reshape(-1), minlength=local_count.size)
            global_count += np.bincount(global_index.reshape(-1), minlength=global_count.size)
        return (
            local_count / max(local_count.sum(), 1),
            global_count / max(global_count.sum(), 1),
        )

    def train(self) -> None:
        print(
            "Adaptive sparse-anchor V4 training: "
            f"local {self.local_topk}/{len(self.local_candidate_lags)}, "
            f"global {self.global_topk}/{len(self.global_candidate_lags)}"
        )
        for epoch in range(self.num_epochs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            self.model.train()
            keys = ["total", "fit", "local_fit", "global_fit", "gap", "area", "coverage", "sigma"]
            totals = {key: 0.0 for key in keys}
            batches = 0

            for input_data, _ in self.train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                _, details = self._forward_batch(input_data)
                loss, parts = self._training_loss(details)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"检测到非有限损失：{loss.item()}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                totals["total"] += float(loss.detach().item())
                for key, value in parts.items():
                    totals[key] += float(value.detach().item())
                batches += 1

            if adjust_learning_rate is not None:
                adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            denom = max(batches, 1)
            print(
                f"Epoch [{epoch + 1:02d}/{self.num_epochs:02d}] "
                f"loss={totals['total']/denom:.6f}, fit={totals['fit']/denom:.6f}, "
                f"local_fit={totals['local_fit']/denom:.6f}, "
                f"global_fit={totals['global_fit']/denom:.6f}, "
                f"gap={totals['gap']/denom:.6f}, area={totals['area']/denom:.6f}, "
                f"coverage={totals['coverage']/denom:.6f}, "
                f"sigma={totals['sigma']/denom:.6f}, time={elapsed:.3f}s"
            )

        self._save_checkpoint()
        local_occ, global_occ = self._anchor_occupancy()
        print(f"Local anchor occupancy : {np.round(local_occ, 3).tolist()}")
        print(f"Global anchor occupancy: {np.round(global_occ, 3).tolist()}")
        print(f"Saved checkpoint: {self.checkpoint_path}")

    def _score_dict(self, details: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scores = {
            "gap": details["score_gap"],
            "total": details["score_total"],
            "combined": details["score_combined"],
        }
        if self.score_normalization == "official":
            normalized = {}
            for name, score in scores.items():
                minimum = score.min(dim=-1, keepdim=True).values
                maximum = score.max(dim=-1, keepdim=True).values
                scaled = (score - minimum) / (maximum - minimum + 1e-5)
                normalized[name] = torch.softmax(scaled, dim=-1)
            return normalized
        if self.score_normalization == "raw":
            return scores
        raise ValueError(f"未知 score_normalization：{self.score_normalization}")

    @torch.no_grad()
    def _stream_thresholds(self, loaders: Iterable) -> Dict[str, float]:
        """
        Compute threshold percentiles using the same rule as the original PPLAD:

            threshold = np.percentile(
                concatenate(train_scores, test_scores),
                100 - anormly_ratio,
            )

        ``exact`` retains every score and therefore matches the original
        percentile procedure. ``reservoir`` is kept only for very large
        low-memory experiments and must not be used for strict comparisons.
        """
        self.model.eval()
        percentile = 100.0 - float(self.anormly_ratio)

        if self.quantile_method == "exact":
            collected = {mode: [] for mode in self.score_modes}
            for loader in loaders:
                for input_data, _ in loader:
                    _, details = self._forward_batch(input_data)
                    score_dict = self._score_dict(details)
                    for mode in self.score_modes:
                        values = (
                            score_dict[mode]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )
                        if not np.isfinite(values).all():
                            raise RuntimeError(
                                f"Score mode {mode!r} contains NaN or Inf; "
                                "refusing to compute an exact percentile."
                            )
                        collected[mode].append(values)

            thresholds: Dict[str, float] = {}
            self._threshold_sample_counts = {}
            for mode in self.score_modes:
                if not collected[mode]:
                    raise RuntimeError(
                        f"No scores were collected for score mode {mode!r}."
                    )
                combined_energy = np.concatenate(collected[mode], axis=0)
                self._threshold_sample_counts[mode] = int(
                    combined_energy.size
                )
                thresholds[mode] = float(
                    np.percentile(combined_energy, percentile)
                )
            return thresholds

        if self.quantile_method == "reservoir":
            estimators = {
                mode: ReservoirQuantile(
                    self.quantile_buffer, self.seed + index
                )
                for index, mode in enumerate(self.score_modes)
            }
            for loader in loaders:
                for input_data, _ in loader:
                    _, details = self._forward_batch(input_data)
                    score_dict = self._score_dict(details)
                    for mode in self.score_modes:
                        estimators[mode].update(
                            score_dict[mode].detach().cpu().numpy()
                        )
            self._threshold_sample_counts = {
                mode: int(estimator.count)
                for mode, estimator in estimators.items()
            }
            return {
                mode: estimator.percentile(percentile)
                for mode, estimator in estimators.items()
            }

        raise ValueError(
            f"Unknown quantile_method: {self.quantile_method!r}"
        )

    @staticmethod
    def _point_adjust(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred = pred.copy()
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
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
        return pred

    @torch.no_grad()
    def test(self) -> Tuple[float, float, float, float]:
        if combine_all_evaluation_scores is None:
            raise RuntimeError("缺少 metrics.metrics。")
        self.model.eval()
        threshold_loaders = (
            [self.train_loader]
            if self.threshold_source == "train"
            else [self.train_loader, self.thre_loader]
        )
        thresholds = self._stream_thresholds(threshold_loaders)

        collected = {mode: [] for mode in self.score_modes}
        label_parts = []
        for input_data, labels in self.thre_loader:
            _, details = self._forward_batch(input_data)
            score_dict = self._score_dict(details)
            for mode in self.score_modes:
                collected[mode].append(score_dict[mode].detach().cpu().numpy())
            label_parts.append(labels.detach().cpu().numpy())
        gt = np.concatenate(label_parts, axis=0).reshape(-1).astype(int)

        primary_result = None
        for mode in self.score_modes:
            test_energy = np.concatenate(collected[mode], axis=0).reshape(-1)
            threshold = thresholds[mode]
            pred = (test_energy > threshold).astype(int)
            print("\n" + "=" * 56)
            print(f"Score mode          : {mode}")
            print(f"Score normalization : {self.score_normalization}")
            print(f"Threshold source    : {self.threshold_source}")
            print(f"Threshold estimator : {self.quantile_method}")
            print(
                f"Threshold samples   : "
                f"{self._threshold_sample_counts.get(mode, 'unknown')}"
            )
            print(f"anormly_ratio       : {self.anormly_ratio}")
            print(f"Threshold           : {threshold}")

            raw_accuracy = accuracy_score(gt, pred)
            raw_precision, raw_recall, raw_f1, _ = precision_recall_fscore_support(
                gt, pred, average="binary", zero_division=0
            )
            raw_mcc = float(matthews_corrcoef(gt, pred))
            print(
                "RAW  Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, "
                "F1={:.4f}, MCC={:.4f}".format(
                    raw_accuracy, raw_precision, raw_recall, raw_f1, raw_mcc
                )
            )

            project_scores = combine_all_evaluation_scores(pred, gt, test_energy)
            project_scores["MCC_score"] = raw_mcc
            for key, value in project_scores.items():
                print(f"{key:21} : {value:0.4f}")

            adjusted_pred = self._point_adjust(pred, gt)
            pa_accuracy = accuracy_score(gt, adjusted_pred)
            pa_precision, pa_recall, pa_f1, _ = precision_recall_fscore_support(
                gt, adjusted_pred, average="binary", zero_division=0
            )
            pa_mcc = float(matthews_corrcoef(gt, adjusted_pred))
            print(
                "PA   Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, "
                "F1={:.4f}, MCC={:.4f}".format(
                    pa_accuracy, pa_precision, pa_recall, pa_f1, pa_mcc
                )
            )

            prefix = os.path.join(
                self.result_path, f"{self.dataset}_adaptive_anchor_v4_{mode}"
            )
            np.savetxt(prefix + "_score.txt", test_energy, fmt="%.10f")
            np.savetxt(prefix + "_pred_raw.txt", pred, fmt="%d")
            np.savetxt(prefix + "_pred_pa.txt", adjusted_pred, fmt="%d")
            np.savetxt(prefix + "_label.txt", gt, fmt="%d")
            if mode == self.primary_score:
                primary_result = (pa_accuracy, pa_precision, pa_recall, pa_f1)

        if primary_result is None:
            raise RuntimeError("未得到 primary_score 的测试结果。")
        return primary_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive Sparse Temporal Anchor Competitive Fitting V4"
    )
    parser.add_argument("--dataset", type=str, default="SMAP")
    parser.add_argument("--data_path", type=str, default="SMAP")
    parser.add_argument("--input_c", type=int, default=25)
    parser.add_argument("--output_c", type=int, default=25)
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anormly_ratio", type=float, default=2.0)
    parser.add_argument("--index", type=int, default=137)
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--local_candidate_lags", nargs="+", type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    parser.add_argument(
        "--global_candidate_lags", nargs="+", type=int,
        default=[12, 16, 20, 24, 28, 32, 40, 48]
    )
    parser.add_argument("--local_topk", type=int, default=2)
    parser.add_argument("--global_topk", type=int, default=4)
    parser.add_argument("--selector_hidden", type=int, default=8)
    parser.add_argument("--fitter_hidden", type=int, default=8)
    parser.add_argument("--selector_temperature", type=float, default=0.5)
    parser.add_argument("--similarity_tau", type=float, default=1.0)
    parser.add_argument("--sigma_min", type=float, default=0.03)
    parser.add_argument("--sigma_max", type=float, default=1.50)
    parser.add_argument("--area_weight", type=float, default=0.1)
    parser.add_argument("--selector_balance_weight", type=float, default=0.05)
    parser.add_argument("--gap_weight", type=float, default=1.0)
    parser.add_argument(
        "--gather_backend",
        choices=["advanced_index", "cached_index", "shifted_stack"],
        default="shifted_stack",
        help="V4 symmetric gather backend used for inference/training; shifted_stack is the optimized default.",
    )
    parser.add_argument(
        "--relation_input", choices=["standardized", "instance"], default="instance"
    )
    parser.add_argument(
        "--score_modes", nargs="+", choices=["gap", "total", "combined"],
        default=["gap", "total", "combined"]
    )
    parser.add_argument(
        "--primary_score", choices=["gap", "total", "combined"], default="combined"
    )
    parser.add_argument(
        "--score_normalization", choices=["raw", "official"], default="official"
    )
    parser.add_argument(
        "--threshold_source", choices=["train", "original"], default="original"
    )
    parser.add_argument(
        "--quantile_method",
        choices=["exact", "reservoir"],
        default="exact",
        help=(
            "exact reproduces PPLAD's full np.percentile procedure; "
            "reservoir is an approximate low-memory alternative"
        ),
    )
    parser.add_argument("--quantile_buffer", type=int, default=50000)

    # 兼容原 Solver.__init__ 所需字段。
    parser.add_argument("--local_size", type=int, default=3)
    parser.add_argument("--global_size", nargs="+", type=int, default=[20])
    parser.add_argument("--d_model", type=int, default=8)
    parser.add_argument("--loss_fuc", type=str, default="MSE")
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--similar", type=str, default="MSE")
    parser.add_argument("--rec_timeseries", action="store_true", default=True)
    parser.add_argument("--model_save_path", type=str, default="checkpoints")
    parser.add_argument("--result_path", type=str, default="result/adaptive_anchor_v4")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--devices", type=str, default="0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    print("========== Adaptive Sparse Temporal Anchor Competitive Fitting V4 ==========")
    runner = AdaptiveSparseAnchorSolverV4(vars(args))
    trainable = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
    print(f"Trainable parameters       : {trainable:,}")
    print(
        "Candidate/selected anchors : "
        f"local {len(args.local_candidate_lags)}/{args.local_topk}, "
        f"global {len(args.global_candidate_lags)}/{args.global_topk}"
    )
    print("Temporal relation          : symmetric (t-lag, t+lag), boundary replication")
    print("Selector gradient          : straight-through Top-k")
    print("Competition                : joint normalization + one shared Gaussian")
    print("Training data              : normal same-timestamp relations only")
    print("Valid score positions      : all positions in every window")
    print(f"Score modes                : {args.score_modes}")

    if args.mode == "train":
        runner.train()
        runner.test()
    else:
        runner.load_checkpoint()
        runner.test()


if __name__ == "__main__":
    main()
