from typing import Optional

import torch
from torch import nn

from mcp.metric import Accuracy
from mcp.task.base import Task, TaskOutput


class SupervisedTask(Task):
    def __init__(self, embedding_size: int, num_classes: int):
        super().__init__()
        self.metric = Accuracy()
        self.output = nn.Linear(embedding_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    @property
    def name(self):
        return "Supervised"

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> TaskOutput:
        if y is None:
            raise ValueError("Labels are required for supervised task")

        x = encoder(x)
        x = self.output(x)

        metric = self.metric(x, y)
        loss = self.loss(x, y)

        return TaskOutput(loss=loss, metric=metric, metric_name="acc")
