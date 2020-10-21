from typing import Optional

import torch
from torch import nn

from mcp.task.base import Task


class SupervisedTask(Task, nn.Module):
    def __init__(self, embedding_size: int, num_classes: int):
        super().__init__()
        self.output = nn.Linear(embedding_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    @property
    def name(self):
        return "Supervised"

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("Labels are required for supervised task")

        x = encoder(x)
        x = self.output(x)

        return self.loss(x, y)
