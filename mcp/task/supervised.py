from typing import Optional

import torch
from torch import nn

from mcp.data.dataset.transforms import KorniaTransforms
from mcp.metric import Accuracy
from mcp.task.base import Task, TaskOutput


class SupervisedTask(Task):
    def __init__(
        self, embedding_size: int, num_classes: int, transforms: KorniaTransforms
    ):
        super().__init__()
        self.metric = Accuracy()
        self.output = nn.Linear(embedding_size, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.transforms_train = [
            transforms.random_crop(),
            transforms.color_jitter(),
            transforms.random_flip(),
            transforms.normalize(),
        ]
        self.transforms_eval = [transforms.normalize()]
        self._training = True

    @property
    def name(self):
        return "Supervised"

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> TaskOutput:
        if y is None:
            raise ValueError("Labels are required for supervised task")

        # x = self._transform(x)
        x = encoder(x)
        x = self.output(x)

        metric = self.metric(x, y)
        loss = self.loss(x, y)

        return TaskOutput(loss=loss, metric=metric, metric_name="acc")

    def _transform(self, x: torch.Tensor):
        transforms = self.transforms_train if self._training else self.transforms_eval
        for t in transforms:
            x = t(x)
        return x

    def train(self, mode: bool = True):
        self._training = mode
        return super().train(mode)
