import abc
from typing import Optional

import torch
from torch import nn


class Metric(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x, y) -> float:
        pass


class Accuracy(Metric):
    def __call__(self, x, y) -> float:
        return 100.0 * (x == y)


class Task(abc.ABC):
    @abc.abstractproperty
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Abstract task definition, receives encoder and input data with optional targets. Returns computed loss."""
        pass

    def eval(self, metric: Metric) -> float:
        pass
