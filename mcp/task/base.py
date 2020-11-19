import abc
from copy import deepcopy
from typing import NamedTuple, Optional

import torch
from torch import nn

from mcp.model.base import Model


class TaskOutput(NamedTuple):
    loss: torch.Tensor
    metric: float
    metric_name: str


class Task(Model):
    def __init__(self):
        super().__init__()
        self._init_state = deepcopy(self.state_dict())

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
        self.load_state_dict(self._init_state)
