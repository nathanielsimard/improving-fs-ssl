from time import time as time
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from mcp.metric import Accuracy
from mcp.task.base import Task, TaskOutput
from mcp.task.compute import TaskCompute


class SupervisedTask(Task):
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        compute: TaskCompute,
        key_transform: str = "default-transform",
        key_forward: str = "default-forward",
    ):
        super().__init__()
        self.compute = compute
        self.key_forward = key_forward
        self.key_transform = key_transform

        self.metric = Accuracy()
        self.output = nn.Linear(embedding_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

        self._initial_state_dict = self.state_dict()
        self._training = True

    @property
    def name(self):
        return "Supervised"

    @property
    def initial_state_dict(self):
        return self._initial_state_dict

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> TaskOutput:
        if y is None:
            raise ValueError("Labels are required for supervised task")

        x = self.compute.cache_transform(x, self._training, key=self.key_transform)
        x = self.compute.cache_forward(x, encoder, key=self.key_forward)

        x = self.output(x)

        metric = self.metric(x, y)
        loss = self.loss(x, y)

        return TaskOutput(loss=loss, metric=metric, metric_name="acc", time=time())

    def train(self, mode: bool = True):
        self._training = mode
        return super().train(mode)


class MultipleSupervisedTasks(Task):
    def __init__(self, tasks: List[SupervisedTask]):
        super().__init__()
        self.tasks = nn.ModuleList(tasks)

        self._initial_state_dict = self.state_dict()
        self._training = True

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> TaskOutput:
        outputs = [t.run(encoder, x, y=y) for t in self.tasks]

        metric = np.asarray([o.metric for o in outputs]).mean()
        loss: torch.Tensor = sum([o.loss for o in outputs]) / len(outputs)  # type: ignore
        metric_name = outputs[0].metric_name
        time = outputs[1].time

        return TaskOutput(loss=loss, metric=metric, metric_name=metric_name, time=time)

    @property
    def name(self):
        return "MultiSupervised"

    @property
    def initial_state_dict(self):
        return self._initial_state_dict

    def train(self, mode: bool = True):
        self._training = mode
        return super().train(mode)
