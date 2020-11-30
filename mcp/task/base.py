import abc
from time import time as time_since_epoch
from typing import NamedTuple, Optional

import torch
from torch import nn

from mcp.model.base import Model


class TaskOutput(NamedTuple):
    loss: torch.Tensor
    metric: float
    metric_name: str
    time: float


class Task(Model):
    @abc.abstractproperty
    def initial_state_dict(self):
        pass

    @abc.abstractproperty
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None,
    ) -> TaskOutput:
        """Abstract task definition, receives encoder and input data with optional targets. Returns computed loss."""
        pass

    def reset(self):
        self.load_state_dict(self.initial_state_dict)


class WeightedTask(Task):
    def __init__(self, task: Task, weight: float):
        super().__init__()
        self.task = task
        self.weight = weight

    @property
    def initial_state_dict(self):
        return self.task.initial_state_dict

    @property
    def name(self) -> str:
        return "Weighted-" + self.task.name

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None,
    ) -> TaskOutput:
        output = self.task.run(encoder, x, y=y)
        return TaskOutput(
            loss=self.weight * output.loss,
            metric=output.metric,
            metric_name=output.metric_name,
            time=time_since_epoch(),
        )

    def state_dict(self):
        return self.task.state_dict()

    def load_state_dict(self, value):
        self.task.load_state_dict(value)
