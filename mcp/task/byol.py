from typing import Optional

import torch
from torch import nn

from mcp.model.mlp import MLP
from mcp.task.base import Task, TaskOutput
from mcp.task.compute import TaskCompute


class BYOLTask(Task):
    def __init__(
        self,
        embedding_size: int,
        compute: TaskCompute,
        head_size: int,
        head_n_hiddens: int,
        dropout: float,
    ):
        super().__init__()
        self.compute = compute
        self.loss = nn.MSELoss()
        self.projection_head = MLP(
            embedding_size, head_size, head_size, head_n_hiddens, dropout
        )
        self.norm = nn.BatchNorm1d(head_size)
        self.predictor = MLP(head_size, head_size, head_size, head_n_hiddens, dropout)

        self._training = True

    @property
    def name(self):
        return "BYOL"

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> TaskOutput:
        x_original = x

        x = self.compute.cache_transform(x_original, self._training)
        x = self.compute.cache_forward(x, encoder)
        # x = self.projection_head(x)
        # x = self.norm(x)
        # x = self.predictor(x)

        x_prime = self.compute.transform(x_original, self._training)
        x_prime = encoder(x_prime)
        # x_prime = self.projection_head(x_prime)
        # x_prime = self.norm(x_prime)

        loss = 100 * self._loss(x, x_prime)
        metric = loss.cpu().detach().item()

        return TaskOutput(loss=loss, metric=metric, metric_name="MSE-norm")

    def train(self, mode: bool = True):
        self._training = mode
        return super().train(mode)

    def _loss(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        x_prime = x_prime / torch.norm(x_prime, dim=-1, keepdim=True)
        return self.loss(x, x_prime)
