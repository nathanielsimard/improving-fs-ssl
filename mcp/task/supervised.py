from typing import Optional

import torch
from torch import nn

from mcp.task.base import Task


class SupervisedTask(Task):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    @property
    def name(self):
        return "Supervised"

    def run(
        self, model: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("Labels are required for supervised task")
        x = model(x)
        return self.loss(x, y)
