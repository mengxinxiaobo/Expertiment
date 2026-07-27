"""V4-IO2 score adapter and bounded reusable CPU window staging."""

from __future__ import annotations

import numpy as np
import torch


class ReusableCPUWindowBatcher:
    """Yield current-batch windows without caching the full window matrix."""

    def __init__(self, data, starts, batch_size, window_size, *, pinned=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.window_size = int(window_size)
        self.channels = int(self.data.shape[1])
        self.pinned = bool(pinned)
        if self.pinned:
            self._torch_buffer = torch.empty(
                (self.batch_size, self.window_size, self.channels),
                dtype=torch.float32, pin_memory=True,
            )
            self._buffer = self._torch_buffer.numpy()
        else:
            self._buffer = np.empty(
                (self.batch_size, self.window_size, self.channels), dtype=np.float32
            )
            self._torch_buffer = torch.from_numpy(self._buffer)

    def __iter__(self):
        for offset in range(0, len(self.starts), self.batch_size):
            current = self.starts[offset : offset + self.batch_size]
            for row, start in enumerate(current):
                self._buffer[row] = self.data[start : start + self.window_size]
            yield current, self._torch_buffer[: len(current)]


class ASCAOptimizedV2ScoreAdapter:
    model_name = "ASCA-AD V4-IO2"
    score_mode = "total"
    reuse_cpu_batch_buffer = True
    use_pinned_staging_buffer = False

    def __init__(self, solver) -> None:
        self.solver = solver
        self.model = solver.model
        self.window_size = int(solver.win_size)
        if solver.score_normalization != "official":
            raise ValueError("V4-IO2 formal inference requires official normalization")

    def eval(self):
        self.model.eval()
        return self

    def window_scores(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[1] != self.window_size:
            raise ValueError(
                f"ASCA V4-IO2 expected [B,{self.window_size},C], got {tuple(windows.shape)}"
            )
        if self.model.training:
            raise RuntimeError("ASCA V4-IO2 adapter requires model.eval()")
        with torch.inference_mode():
            return self.solver.forward_total_score(windows)
