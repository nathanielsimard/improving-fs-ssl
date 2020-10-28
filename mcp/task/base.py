import abc
from typing import NamedTuple, Optional

import torch
from torch import nn

from mcp.model.base import Model


class TaskOutput(NamedTuple):
    loss: torch.Tensor
    metric: float
    metric_name: str


class Task(Model):
    @abc.abstractproperty
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None,
    ) -> TaskOutput:
        """Abstract task definition, receives encoder and input data with optional targets. Returns computed loss."""
        pass
