"""Minimal inference-only solver for the independent ASCA V4-IO model."""

from __future__ import annotations

import torch

from .model_inference_optimized import ASCAInferenceOptimized


class ASCASolverInferenceOptimized:
    score_mode = "total"

    def __init__(
        self,
        model: ASCAInferenceOptimized,
        device: torch.device,
        window_size: int = 100,
        relation_input: str = "instance",
        score_normalization: str = "official",
    ) -> None:
        self.model = model
        self.device = device
        self.win_size = int(window_size)
        self.relation_input = str(relation_input)
        self.score_normalization = str(score_normalization)
        if self.score_normalization not in {"official", "raw"}:
            raise ValueError("score_normalization must be 'official' or 'raw'")

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
        raise ValueError(f"unknown relation_input: {self.relation_input}")

    def _score_total_only(self, score: torch.Tensor) -> torch.Tensor:
        if self.score_normalization == "raw":
            return score
        minimum = score.min(dim=-1, keepdim=True).values
        maximum = score.max(dim=-1, keepdim=True).values
        scaled = (score - minimum) / (maximum - minimum + 1e-5)
        return torch.softmax(scaled, dim=-1)

    def forward_total_score(self, input_data: torch.Tensor) -> torch.Tensor:
        total = self.model.forward_total_score_only(self._prepare_input(input_data))
        return self._score_total_only(total)
