"""Score-only adapter around the unmodified official PPLAD Solver.

The equations below are the score-producing section of PPLAD's official
``Solver.test`` method.  Thresholding, labels, point adjustment, and metrics are
intentionally absent.  The adapter receives an already constructed official
Solver instance so its model, RevIN class, similarity function, criterion, and
device remain authoritative.
"""

from __future__ import annotations

from types import ModuleType

import torch


class PPLADScoreAdapter:
    model_name = "PPLAD"

    def __init__(self, solver, solver_module: ModuleType) -> None:
        self.solver = solver
        self.solver_module = solver_module
        self.model = solver.model
        self.window_size = int(solver.win_size)

    def eval(self) -> "PPLADScoreAdapter":
        self.model.eval()
        return self

    def _relations(self, x: torch.Tensor):
        batch, length, channels = x.shape
        local_targets = []
        global_targets = []
        last_relation = None

        for index, local_size in enumerate(self.solver.local_size):
            total_size = int(local_size) + int(self.solver.global_size[index])
            front = total_size // 2
            back = total_size - front
            boundary = length - back
            neighborhoods = []
            for position in range(self.window_size):
                if position < front:
                    prefix = x[:, 0, :].unsqueeze(1).repeat(1, front - position, 1)
                    values = torch.cat((prefix, x[:, 0:position, :]), dim=1)
                    values = torch.cat((values, x[:, position:position + back, :]), dim=1)
                elif position > boundary:
                    suffix = x[:, length - 1, :].unsqueeze(1).repeat(
                        1, back + position - length, 1
                    )
                    values = torch.cat(
                        (x[:, position - front:self.window_size, :], suffix), dim=1
                    )
                else:
                    values = x[:, position - front:position + back, :].reshape(
                        batch, -1, channels
                    )
                neighborhoods.append(values)

            relation = torch.cat(neighborhoods, dim=0).reshape(
                length, batch, total_size, channels
            ).permute(1, 0, 3, 2)
            relation = self.solver_module.cal_similar(
                self.solver.similar, relation, x, total_size
            )
            relation = torch.softmax(relation, dim=-1)

            center = total_size // 2
            local_front = int(local_size) // 2
            local_back = int(local_size) - local_front
            local_targets.append(
                relation[:, :, center - local_front:center + local_back]
            )
            global_targets.append(
                torch.cat(
                    (
                        relation[:, :, :center - local_front],
                        relation[:, :, center + local_back:total_size],
                    ),
                    dim=-1,
                )
            )
            last_relation = relation

        if last_relation is None:
            raise RuntimeError("PPLAD local_size is empty")
        return local_targets, global_targets, last_relation

    def window_scores(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[1] != self.window_size:
            raise ValueError(
                f"PPLAD expected [B,{self.window_size},C], got {tuple(windows.shape)}"
            )

        inputs = windows.float().to(self.solver.device)
        revin = self.solver_module.RevIN(num_features=self.solver.input_c)
        x = revin(inputs, "norm")
        local_targets, global_targets, relation = self._relations(x)
        outputs = self.model(
            x,
            local_targets,
            global_targets,
            "test",
            0,
            relation,
        )
        series, prior = outputs[0], outputs[1]
        series_loss = 0.0
        prior_loss = 0.0
        for index in range(len(prior)):
            series_loss = series_loss + torch.sum(
                self.solver.criterion_keep(series[index], local_targets[index]), dim=-1
            )
            prior_loss = prior_loss + torch.sum(
                self.solver.criterion_keep(prior[index], global_targets[index]), dim=-1
            )

        # Official PPLAD test protocol: per-window min-max, then temporal softmax.
        metric = self.solver_module.minmax_norm(series_loss + prior_loss)
        return torch.softmax(metric, dim=-1)
