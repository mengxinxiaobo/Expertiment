"""Checkpoint-compatible, total-score-only inference implementation of ASCA V4.

The training-compatible ``forward`` intentionally preserves the original V4
mathematics.  Formal optimized inference must use ``forward_total_score_only``.
No class or function in :mod:`asca_ad` is patched or replaced.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


class SharedAnchorSelector(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class SharedGaussianFitter(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 8,
        sigma_min: float = 0.03,
        sigma_max: float = 1.50,
    ) -> None:
        super().__init__()
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("sigma bounds must satisfy 0 < sigma_min < sigma_max")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features).squeeze(-1)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)


class ASCAInferenceOptimized(nn.Module):
    """ASCA-AD V4-IO with an independent total-only inference path.

    Persistent parameters and buffers exactly match the original V4.  Cached
    window-dependent tensors are non-persistent and therefore never enter the
    state dict or a checkpoint.
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
        window_size: int = 100,
    ) -> None:
        super().__init__()
        local_lags = sorted(set(int(v) for v in local_candidate_lags))
        global_lags = sorted(set(int(v) for v in global_candidate_lags))
        if not local_lags or not global_lags:
            raise ValueError("local and global lag sets must be non-empty")
        if min(local_lags + global_lags) <= 0:
            raise ValueError("all lags must be positive")
        if set(local_lags).intersection(global_lags):
            raise ValueError("local and global lag sets must not overlap")
        if not 1 <= local_topk <= len(local_lags):
            raise ValueError("invalid local_topk")
        if not 1 <= global_topk <= len(global_lags):
            raise ValueError("invalid global_topk")
        if selector_temperature <= 0 or similarity_tau <= 0:
            raise ValueError("temperatures must be positive")
        if gap_weight < 0:
            raise ValueError("gap_weight cannot be negative")
        if window_size <= max(local_lags + global_lags):
            raise ValueError("window_size must exceed every lag")

        self.local_topk = int(local_topk)
        self.global_topk = int(global_topk)
        self.selector_temperature = float(selector_temperature)
        self.similarity_tau = float(similarity_tau)
        self.gap_weight = float(gap_weight)
        self.max_lag = float(max(local_lags + global_lags))
        self.local_edge = float(max(local_lags)) / self.max_lag
        self.window_size = int(window_size)

        # These two persistent buffers intentionally retain the original keys.
        self.register_buffer("local_lags", torch.tensor(local_lags, dtype=torch.long))
        self.register_buffer("global_lags", torch.tensor(global_lags, dtype=torch.long))
        self.selector = SharedAnchorSelector(input_dim=6, hidden_dim=selector_hidden)
        self.fitter = SharedGaussianFitter(
            input_dim=8,
            hidden_dim=fitter_hidden,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        self._register_index_cache("local", self.local_lags)
        self._register_index_cache("global", self.global_lags)

    def _register_index_cache(self, name: str, lags: torch.Tensor) -> None:
        time_index = torch.arange(self.window_size, dtype=torch.long).view(1, self.window_size, 1)
        lag_view = lags.detach().cpu().view(1, 1, -1)
        self.register_buffer(
            f"_{name}_left_index",
            (time_index - lag_view).clamp(0, self.window_size - 1),
            persistent=False,
        )
        self.register_buffer(
            f"_{name}_right_index",
            (time_index + lag_view).clamp(0, self.window_size - 1),
            persistent=False,
        )
        self.register_buffer(
            f"_{name}_lag_norm",
            lags.detach().cpu().to(torch.float32).view(1, 1, -1) / self.max_lag,
            persistent=False,
        )

    def _apply(self, function):
        """Move the module, then rebuild normalized lags on the target device.

        The original V4 performs integer-lag -> float conversion and division on
        the inference device. Pre-dividing on CPU changes a few float32 low bits;
        near a Top-k tie that can change the selected anchor. Recomputing these
        non-persistent caches after ``to/cuda`` preserves the exact original
        device-side arithmetic while retaining the cache during inference.
        """
        super()._apply(function)
        dtype = self.selector.network[0].weight.dtype
        self._local_lag_norm = (
            self.local_lags.to(dtype=dtype).view(1, 1, -1) / self.max_lag
        )
        self._global_lag_norm = (
            self.global_lags.to(dtype=dtype).view(1, 1, -1) / self.max_lag
        )
        return self

    @staticmethod
    def _sequence_sketch(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        rms = torch.sqrt(x.square().mean(dim=-1) + 1e-8)
        mean_abs = x.abs().mean(dim=-1)
        return torch.stack([mean, std, rms, mean_abs], dim=-1)

    def _symmetric_gather_cached(
        self, x: torch.Tensor, group: str, lags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length = x.shape[:2]
        if length == self.window_size:
            left_index = getattr(self, f"_{group}_left_index").expand(batch, -1, -1)
            right_index = getattr(self, f"_{group}_right_index").expand(batch, -1, -1)
        else:
            time_index = torch.arange(length, device=x.device).view(1, length, 1)
            lag_view = lags.view(1, 1, -1)
            left_index = (time_index - lag_view).clamp(0, length - 1).expand(batch, -1, -1)
            right_index = (time_index + lag_view).clamp(0, length - 1).expand(batch, -1, -1)
        batch_index = torch.arange(batch, device=x.device).view(batch, 1, 1).expand_as(left_index)
        return x[batch_index, left_index], x[batch_index, right_index]

    def _candidate_group(
        self,
        x: torch.Tensor,
        sketch: torch.Tensor,
        lags: torch.Tensor,
        group_value: float,
        group: str,
    ) -> Dict[str, torch.Tensor]:
        left_x, right_x = self._symmetric_gather_cached(x, group, lags)
        left_sketch, right_sketch = self._symmetric_gather_cached(sketch, group, lags)
        current_x = x.unsqueeze(2)
        left_distance = (current_x - left_x).square().mean(dim=-1)
        right_distance = (current_x - right_x).square().mean(dim=-1)
        affinity = torch.exp(-left_distance / self.similarity_tau) + torch.exp(
            -right_distance / self.similarity_tau
        )
        current_sketch = sketch.unsqueeze(2)
        delta = 0.5 * (
            torch.abs(current_sketch - left_sketch) + torch.abs(current_sketch - right_sketch)
        )
        batch, length, count, _ = delta.shape
        if length == self.window_size:
            lag_norm = getattr(self, f"_{group}_lag_norm").to(dtype=x.dtype).expand(batch, length, -1)
        else:
            lag_norm = (lags.to(x.dtype).view(1, 1, count) / self.max_lag).expand(batch, length, -1)
        group_feature = x.new_full((batch, length, count, 1), float(group_value))
        selector_features = torch.cat([delta, lag_norm.unsqueeze(-1), group_feature], dim=-1)
        return {"affinity": affinity, "logits": self.selector(selector_features), "lag_norm": lag_norm}

    def _straight_through_topk(
        self, logits: torch.Tensor, topk: int
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
    def _hard_topk_eval(logits: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_index = torch.topk(logits, k=topk, dim=-1).indices
        gate = torch.zeros_like(logits).scatter(-1, selected_index, 1.0)
        weights = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return gate, weights

    @staticmethod
    def _weighted_summary(
        target: torch.Tensor, lag_norm: torch.Tensor, gate: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        mean = torch.sum(weights * target, dim=-1)
        variance = torch.sum(weights * (target - mean.unsqueeze(-1)).square(), dim=-1)
        std = torch.sqrt(variance + 1e-8)
        mean_lag = torch.sum(weights * lag_norm, dim=-1)
        selected_mass = torch.sum(gate * target, dim=-1)
        return torch.stack([mean, std, mean_lag, selected_mass], dim=-1)

    @staticmethod
    def _weighted_fit_error(
        target: torch.Tensor, prediction: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return torch.sum(weights * (target - prediction).square(), dim=-1)

    def _fit_terms(self, x: torch.Tensor, optimized_eval: bool) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("input must have shape [B,L,C]")
        sketch = self._sequence_sketch(x)
        local = self._candidate_group(x, sketch, self.local_lags, 0.0, "local")
        global_ = self._candidate_group(x, sketch, self.global_lags, 1.0, "global")
        if optimized_eval:
            if self.training:
                raise RuntimeError("optimized total-only path is inference-only; call model.eval()")
            local_gate, local_weights = self._hard_topk_eval(local["logits"], self.local_topk)
            global_gate, global_weights = self._hard_topk_eval(global_["logits"], self.global_topk)
            local_index = global_index = None
        else:
            local_gate, local_weights, local_index = self._straight_through_topk(local["logits"], self.local_topk)
            global_gate, global_weights, global_index = self._straight_through_topk(global_["logits"], self.global_topk)

        center_affinity = torch.ones_like(local["affinity"][..., :1])
        joint_affinity = torch.cat([center_affinity, local["affinity"], global_["affinity"]], dim=-1)
        joint_target = joint_affinity / joint_affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        local_count = self.local_lags.numel()
        target_center = joint_target[..., :1]
        target_local = joint_target[..., 1 : 1 + local_count]
        target_global = joint_target[..., 1 + local_count :]
        local_summary = self._weighted_summary(target_local, local["lag_norm"], local_gate, local_weights)
        global_summary = self._weighted_summary(target_global, global_["lag_norm"], global_gate, global_weights)
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
        center_weight = local_weights.new_full(
            (*local_weights.shape[:-1], 1), 1.0 / float(self.local_topk + 1)
        )
        local_pair_weights = local_weights * (float(self.local_topk) / float(self.local_topk + 1))
        local_fit = (
            center_weight.squeeze(-1)
            * (target_center.squeeze(-1) - pred_center.squeeze(-1)).square()
            + self._weighted_fit_error(target_local, pred_local, local_pair_weights)
        )
        global_fit = self._weighted_fit_error(target_global, pred_global, global_weights)
        if optimized_eval:
            # Do not retain affinity/logit/target/summary tensors in an inference
            # details object. They can be released before the final addition.
            return {"local_fit": local_fit, "global_fit": global_fit}
        return {
            "local_fit": local_fit,
            "global_fit": global_fit,
            "sigma": sigma,
            "local": local,
            "global": global_,
            "local_gate": local_gate,
            "global_gate": global_gate,
            "local_index": local_index,
            "global_index": global_index,
            "target_center": target_center,
            "target_local": target_local,
            "target_global": target_global,
        }

    def forward_total_score_only(self, x: torch.Tensor) -> torch.Tensor:
        terms = self._fit_terms(x, optimized_eval=True)
        return terms["local_fit"] + terms["global_fit"]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training-compatible path preserving the original V4 outputs."""
        terms = self._fit_terms(x, optimized_eval=False)
        local_fit, global_fit, sigma = terms["local_fit"], terms["global_fit"], terms["sigma"]
        standard_normal = torch.distributions.Normal(torch.zeros_like(sigma), torch.ones_like(sigma))
        edge_z = sigma.new_tensor(self.local_edge) / sigma
        pred_local_area = standard_normal.cdf(edge_z) - standard_normal.cdf(-edge_z)
        pred_global_area = 1.0 - pred_local_area
        target_local_area = terms["target_center"].squeeze(-1) + terms["target_local"].sum(dim=-1)
        target_global_area = terms["target_global"].sum(dim=-1)
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
                terms["local"]["logits"] / self.selector_temperature, dim=-1
            ),
            "global_probabilities": torch.softmax(
                terms["global"]["logits"] / self.selector_temperature, dim=-1
            ),
            "local_gate": terms["local_gate"],
            "global_gate": terms["global_gate"],
            "local_selected_index": terms["local_index"],
            "global_selected_index": terms["global_index"],
        }
        return score_combined, details
