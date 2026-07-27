"""Score-only adapter around the unmodified official LTFAD Solver.

This retains LTFAD's official RevIN, neighborhood construction, six error
terms, max reduction, weighting by ``r``, and temporal softmax.  It performs no
label access, thresholding, or evaluation.
"""

from __future__ import annotations

from types import ModuleType

import torch


class LTFADScoreAdapter:
    model_name = "LTFAD"

    def __init__(self, solver, solver_module: ModuleType) -> None:
        self.solver = solver
        self.solver_module = solver_module
        self.model = solver.model
        self.window_size = int(solver.win_size)

    def eval(self) -> "LTFADScoreAdapter":
        self.model.eval()
        return self

    def _relations(self, x: torch.Tensor):
        batch, length, channels = x.shape
        local_targets = []
        global_targets = []

        for index, local_size in enumerate(self.solver.local_size):
            local_size = int(local_size)
            global_size = int(self.solver.global_size[index])
            local_front = local_size // 2
            local_back = local_size - local_front
            local_boundary = length - local_back
            local_values = []

            for position in range(self.window_size):
                if position < local_front:
                    prefix = x[:, 0, :].unsqueeze(1).repeat(
                        1, local_front - position, 1
                    )
                    values = torch.cat((prefix, x[:, :position, :]), dim=1)
                    values = torch.cat(
                        (values, x[:, position:position + local_back, :]), dim=1
                    )
                elif position > local_boundary:
                    suffix = x[:, length - 1, :].unsqueeze(1).repeat(
                        1, local_back + position - length, 1
                    )
                    values = torch.cat(
                        (x[:, position - local_front:self.window_size, :], suffix),
                        dim=1,
                    )
                else:
                    values = x[
                        :, position - local_front:position + local_back, :
                    ].reshape(batch, -1, channels)
                local_values.append(values)

            local_relation = torch.cat(local_values, dim=0).reshape(
                length, batch, local_size, channels
            ).permute(1, 0, 3, 2)

            total_size = local_size + global_size
            total_front = total_size // 2
            total_back = total_size - total_front
            total_boundary = length - total_back
            global_values = []
            for position in range(self.window_size):
                if position < total_front:
                    prefix = x[:, 0, :].unsqueeze(1).repeat(
                        1, total_front - position, 1
                    )
                    values = torch.cat((prefix, x[:, :position, :]), dim=1)
                    values = torch.cat(
                        (values, x[:, position:position + total_back, :]), dim=1
                    )
                    values = torch.cat(
                        (
                            values[:, :total_front - local_front, :],
                            values[
                                :,
                                total_front + local_back:total_front + total_back,
                                :,
                            ],
                        ),
                        dim=1,
                    )
                elif position > total_boundary:
                    suffix = x[:, length - 1, :].unsqueeze(1).repeat(
                        1, total_back + position - length, 1
                    )
                    values = torch.cat(
                        (x[:, position - total_front:self.window_size, :], suffix),
                        dim=1,
                    )
                    values = torch.cat(
                        (
                            values[:, :total_front - local_front, :],
                            values[
                                :,
                                total_front + local_back:total_front + total_back,
                                :,
                            ],
                        ),
                        dim=1,
                    )
                else:
                    values = torch.cat(
                        (
                            x[:, position - total_front:position - local_front, :],
                            x[:, position + local_back:position + total_back, :],
                        ),
                        dim=1,
                    )
                global_values.append(values)

            global_relation = torch.cat(global_values, dim=0).reshape(
                length, batch, global_size, channels
            ).permute(1, 0, 3, 2)
            local_targets.append(local_relation)
            global_targets.append(global_relation)

        if not local_targets:
            raise RuntimeError("LTFAD local_size is empty")
        return local_targets, global_targets

    def window_scores(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[1] != self.window_size:
            raise ValueError(
                f"LTFAD expected [B,{self.window_size},C], got {tuple(windows.shape)}"
            )

        inputs = windows.float().to(self.solver.device)
        revin = self.solver_module.RevIN(num_features=self.solver.input_c)
        x = revin(inputs, "norm")
        local_targets, global_targets = self._relations(x)
        series, prior, series_frequency, prior_frequency = self.model(
            local_targets, global_targets
        )

        losses = [0.0] * 6
        for index in range(len(prior)):
            losses[0] = losses[0] + torch.sum(
                self.solver.criterion_keep(series[index], global_targets[index]),
                dim=-1,
            )
            losses[1] = losses[1] + torch.sum(
                self.solver.criterion_keep(
                    series_frequency[index], global_targets[index]
                ),
                dim=-1,
            )
            losses[2] = losses[2] + torch.sum(
                self.solver.criterion_keep(prior[index], local_targets[index]),
                dim=-1,
            )
            losses[3] = losses[3] + torch.sum(
                self.solver.criterion_keep(
                    prior_frequency[index], local_targets[index]
                ),
                dim=-1,
            )
            losses[4] = losses[4] + torch.sum(
                self.solver.criterion_keep(series[index], series_frequency[index]),
                dim=-1,
            )
            losses[5] = losses[5] + torch.sum(
                self.solver.criterion_keep(prior[index], prior_frequency[index]),
                dim=-1,
            )

        losses = [torch.max(value, dim=-1).values for value in losses]
        metric = (
            (losses[0] + losses[1] + losses[4]) * float(self.solver.r)
            + (losses[2] + losses[3] + losses[5])
            * (1.0 - float(self.solver.r))
        )
        return torch.softmax(metric, dim=-1)
