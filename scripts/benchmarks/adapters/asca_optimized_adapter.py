"""Score-only adapter for the independent ASCA-AD V4-IO implementation."""

from __future__ import annotations

import torch


class ASCAOptimizedScoreAdapter:
    model_name = "ASCA-AD V4-IO"
    score_mode = "total"

    def __init__(self, solver) -> None:
        self.solver = solver
        self.model = solver.model
        self.window_size = int(solver.win_size)
        if solver.score_normalization != "official":
            raise ValueError("optimized formal inference requires official normalization")

    def eval(self) -> "ASCAOptimizedScoreAdapter":
        self.model.eval()
        return self

    def window_scores(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[1] != self.window_size:
            raise ValueError(
                f"ASCA V4-IO expected [B,{self.window_size},C], got {tuple(windows.shape)}"
            )
        if self.model.training:
            raise RuntimeError("ASCA V4-IO score adapter requires model.eval()")
        with torch.inference_mode():
            return self.solver.forward_total_score(windows)
