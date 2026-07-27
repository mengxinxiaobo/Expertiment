"""Thin score-only adapter for the frozen ASCA-AD V4 checkpoint protocol."""

from __future__ import annotations

import torch


class ASCAV4ScoreAdapter:
    """Expose ASCA V4's pre-declared ``total`` score as `[B,T]` energy."""

    model_name = "ASCA-AD V4"
    score_mode = "total"

    def __init__(self, solver) -> None:
        self.solver = solver
        self.model = solver.model
        self.window_size = int(solver.win_size)

        configured_modes = tuple(str(mode) for mode in solver.score_modes)
        if self.score_mode not in configured_modes:
            raise ValueError(
                f"ASCA score_modes={configured_modes} does not contain the "
                f"frozen score mode {self.score_mode!r}"
            )
        if str(solver.score_normalization) != "official":
            raise ValueError(
                "ASCA unified benchmark requires score_normalization='official'"
            )

    def eval(self) -> "ASCAV4ScoreAdapter":
        self.model.eval()
        return self

    def window_scores(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[1] != self.window_size:
            raise ValueError(
                f"ASCA expected [B,{self.window_size},C], got {tuple(windows.shape)}"
            )
        _combined, details = self.solver._forward_batch(windows)
        scores = self.solver._score_dict(details)[self.score_mode]
        return scores
