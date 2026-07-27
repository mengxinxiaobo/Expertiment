"""ASCA-AD V4-IO2: checkpoint-compatible chunked-gather inference.

Only scheduling of the inference calculation changes.  Persistent parameters,
score mathematics and checkpoint keys are inherited unchanged from V4-IO.
No full ``[B,L,K,C]`` lag tensor is ever assembled in this implementation.
"""

from __future__ import annotations

from typing import Dict

import torch

from asca_ad_optimized.model_inference_optimized import ASCAInferenceOptimized


class ASCAInferenceOptimizedV2(ASCAInferenceOptimized):
    """Memory-oriented V4-IO implementation with lag-chunk scheduling."""

    use_total_only = True
    use_cached_indices = True
    use_chunked_gather = True

    def __init__(self, *args, chunk_k: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if int(chunk_k) not in {1, 2, 4, 8}:
            raise ValueError("chunk_k must be one of 1, 2, 4, 8")
        self.chunk_k = int(chunk_k)

    def _chunk_indices(
        self, group: str, lags: torch.Tensor, start: int, end: int,
        batch: int, length: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if length == self.window_size:
            left = getattr(self, f"_{group}_left_index")[:, :, start:end]
            right = getattr(self, f"_{group}_right_index")[:, :, start:end]
            left = left.expand(batch, -1, -1)
            right = right.expand(batch, -1, -1)
        else:
            time = torch.arange(length, device=device).view(1, length, 1)
            lag = lags[start:end].view(1, 1, -1)
            left = (time - lag).clamp(0, length - 1).expand(batch, -1, -1)
            right = (time + lag).clamp(0, length - 1).expand(batch, -1, -1)
        batch_index = torch.arange(batch, device=device).view(batch, 1, 1).expand_as(left)
        return batch_index, left, right

    def _candidate_group(
        self,
        x: torch.Tensor,
        sketch: torch.Tensor,
        lags: torch.Tensor,
        group_value: float,
        group: str,
    ) -> Dict[str, torch.Tensor]:
        batch, length, _ = x.shape
        count = int(lags.numel())
        affinity = x.new_empty((batch, length, count))
        logits = x.new_empty((batch, length, count))
        if length == self.window_size:
            lag_norm = getattr(self, f"_{group}_lag_norm").to(dtype=x.dtype).expand(batch, length, -1)
        else:
            lag_norm = (lags.to(x.dtype).view(1, 1, count) / self.max_lag).expand(batch, length, -1)

        current_x = x.unsqueeze(2)
        current_sketch = sketch.unsqueeze(2)
        for start in range(0, count, self.chunk_k):
            end = min(start + self.chunk_k, count)
            batch_index, left_index, right_index = self._chunk_indices(
                group, lags, start, end, batch, length, x.device
            )

            # Channel-heavy tensors exist for the current lag chunk only.
            left_chunk = x[batch_index, left_index]
            right_chunk = x[batch_index, right_index]
            left_distance = (current_x - left_chunk).square().mean(dim=-1)
            right_distance = (current_x - right_chunk).square().mean(dim=-1)
            affinity[:, :, start:end] = torch.exp(
                -left_distance / self.similarity_tau
            ) + torch.exp(-right_distance / self.similarity_tau)
            del left_chunk, right_chunk, left_distance, right_distance

            left_sketch = sketch[batch_index, left_index]
            right_sketch = sketch[batch_index, right_index]
            delta = 0.5 * (
                torch.abs(current_sketch - left_sketch)
                + torch.abs(current_sketch - right_sketch)
            )
            group_feature = x.new_full((batch, length, end - start, 1), float(group_value))
            selector_features = torch.cat(
                [delta, lag_norm[:, :, start:end].unsqueeze(-1), group_feature], dim=-1
            )

            # Padding preserves the V4-IO linear-kernel reduction shape, which
            # avoids a chunk-size-dependent roundoff changing a near-tied Top-k.
            padded = x.new_zeros((batch, length, count, 6))
            padded[:, :, start:end] = selector_features
            logits[:, :, start:end] = self.selector(padded)[:, :, start:end]
            del left_sketch, right_sketch, delta, group_feature, selector_features, padded
            del batch_index, left_index, right_index

        return {"affinity": affinity, "logits": logits, "lag_norm": lag_norm}

    def feature_flags(self) -> dict:
        return {
            "use_total_only": True,
            "use_cached_indices": True,
            "use_chunked_gather": True,
            "chunk_k": self.chunk_k,
            "reuse_cpu_batch_buffer": True,
            "use_pinned_staging_buffer": False,
        }
