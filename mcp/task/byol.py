from copy import deepcopy
from typing import Optional

import torch
from torch import nn

from mcp.model.base import freeze_weights
from mcp.task.base import Task, TaskOutput
from mcp.task.compute import TaskCompute


class Head(nn.Module):
    def __init__(self, size_input: int, size_hidden: int, size_output: int):
        super().__init__()
        self.input = nn.Linear(size_input, size_hidden)
        self.output = nn.Linear(size_hidden, size_output)
        self.batch_norm = nn.BatchNorm1d(size_hidden)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.output(x)

        return x


class BYOLTask(Task):
    def __init__(
        self,
        embedding_size: int,
        compute: TaskCompute,
        head_size: int,
        head_n_hiddens: int,
        tau: float,
    ):
        super().__init__()
        self.compute = compute
        self.tau = tau
        self.loss = nn.MSELoss()
        self.head_projection = Head(embedding_size, head_size, head_size)
        self.head_prediction = Head(head_size, head_size, head_size)

        self._momentum_encoder: Optional[nn.Module] = None
        self._momentum_head_projection: Optional[nn.Module] = None

        self._training = True

    @property
    def name(self):
        return "BYOL"

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> TaskOutput:

        self._update_momentum_model(encoder, self.head_projection)

        x_original = x

        x = self.compute.cache_transform(x_original, self._training)
        x = self.compute.cache_forward(x, encoder)
        x = self.head_projection(x)
        x = self.head_prediction(x)

        x_prime = self.compute.transform(x_original, self._training)
        x_prime = self._momentum_encoder(x_prime)  # type: ignore
        x_prime = self._momentum_head_projection(x_prime)  # type: ignore

        loss = 100 * self._loss(x, x_prime)
        metric = loss.cpu().detach().item()

        return TaskOutput(loss=loss, metric=metric, metric_name="MSE-norm")

    def train(self, mode: bool = True):
        self._training = mode
        return super().train(mode)

    def _update_momentum_model(self, encoder: nn.Module, head_projection: nn.Module):
        if self._momentum_encoder is None:
            self._momentum_encoder = _initialize_momentum_module(encoder)

        if self._momentum_head_projection is None:
            self._momentum_head_projection = _initialize_momentum_module(
                head_projection
            )

        _update_momentum_module(encoder, self._momentum_encoder, self.tau)
        _update_momentum_module(
            head_projection, self._momentum_head_projection, self.tau
        )

    def _loss(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        x_prime = x_prime / torch.norm(x_prime, dim=-1, keepdim=True)
        return self.loss(x, x_prime)


def _update_momentum_module(module: nn.Module, module_momentum: nn.Module, tau: float):
    tmp = deepcopy(module_momentum)
    for param, param_momentum in zip(module.parameters(), module_momentum.parameters()):
        param_momentum = torch.tensor(param_momentum * tau + param * (1 - tau))

    for p, p1 in zip(module_momentum.parameters(), tmp.parameters()):
        assert (p == p1).sum() != (p == p).sum()


def _initialize_momentum_module(module: nn.Module) -> nn.Module:
    momentum_module = deepcopy(module)
    freeze_weights(momentum_module)
    return momentum_module
