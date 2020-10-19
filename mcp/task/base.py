import abc
from typing import Optional

import torch
from torch import nn


class Task(abc.ABC):
    @abc.abstractproperty
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def run(
        self, model: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Abstract task definition, receives model and input data with optional targets. Returns computed loss."""
        pass
