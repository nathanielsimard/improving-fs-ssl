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


class TrainableModule(nn.Module):
    def __init__(self, head_projection: nn.Module, head_prediction: nn.Module):
        super().__init__()
        self.head_projection = head_projection
        self.head_prediction = head_prediction


class BYOLTask(Task):
    def __init__(
        self, embedding_size: int, compute: TaskCompute, head_size: int, tau: float,
    ):
        super().__init__()
        self.compute = compute
        self.tau = tau
        self.loss = nn.MSELoss()
        head_projection = Head(embedding_size, head_size, head_size)
        head_prediction = Head(head_size, head_size, head_size)

        self.trainable = TrainableModule(head_projection, head_prediction)

        self._momentum_encoder: Optional[nn.Module] = None
        self._momentum_head_projection: Optional[nn.Module] = None
        self._initial_state_dict = self.state_dict()

        self._training = True

    @property
    def name(self):
        return "BYOL"

    @property
    def initial_state_dict(self):
        return self._initial_state_dict

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> TaskOutput:
        self._update_momentum_model(encoder, self.trainable.head_projection)

        x_original = x

        x = self.compute.cache_transform(x_original, self._training)
        x = self.compute.cache_forward(x, encoder)
        x = self.trainable.head_projection(x)
        x = self.trainable.head_prediction(x)

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
            print("Module encoder is None")
            self._momentum_encoder = _initialize_momentum_module(encoder)

        if self._momentum_head_projection is None:
            print("Module head projection is None")
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

    def state_dict(self):
        value = {}
        value["trainable"] = self.trainable.state_dict()
        value["momentum_encoder"] = _state_dict_or_none(self._momentum_encoder)
        value["momentum_head_projection"] = _state_dict_or_none(
            self._momentum_head_projection
        )

        return value

    def load_state_dict(self, value):
        self.trainable.load_state_dict(value["trainable"])
        self._momentum_encoder = value["momentum_encoder"]
        self._momentum_head_projection = value["momentum_head_projection"]


def _state_dict_or_none(module: Optional[nn.Module]):
    if module is None:
        return None

    return module.state_dict()


def _update_momentum_module(module: nn.Module, module_momentum: nn.Module, tau: float):
    state_module = module.state_dict()
    state_momentum = module_momentum.state_dict()
    state_momentum_updated = {}

    for key in state_module.keys():
        state_momentum_updated[key] = state_momentum[key] * tau + state_module[key] * (
            1 - tau
        )

    module_momentum.load_state_dict(state_momentum_updated)


def _initialize_momentum_module(module: nn.Module) -> nn.Module:
    momentum_module = deepcopy(module)
    freeze_weights(momentum_module)
    return momentum_module
