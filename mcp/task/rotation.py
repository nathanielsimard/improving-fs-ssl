from typing import Optional, NamedTuple, List

import torch
import random
from collections import defaultdict
from torch import nn

from mcp.data.dataset.transforms import KorniaTransforms
from mcp.metric import Accuracy
from mcp.task.base import Task, TaskOutput

class RotationOutput(NamedTuple):
    batch: torch.Tensor
    labels: torch.Tensor

class BatchRotation(object):
    def __init__(self, transforms: KorniaTransforms, degrees: List[int] = [0, 90, 180, 270]):
        self.rotations = [transforms.rotate(d) for d in degrees]

    def rotate(self, x: torch.Tensor) -> RotationOutput:
        tfm_ids = list(range(len(self.rotations)))
        sample_ids = list(range(x.size(0)))
        random.shuffle(sample_ids)
        batch_ids = defaultdict(lambda: [])
        for _id in sample_ids:
            tfm_id = random.sample(tfm_ids, 1)
            batch_ids[tfm_id].append(_id)

        out = torch.empty_like(x)
        labels = torch.empty(x.size(0), dtype=torch.long)
        for tfm_id, ids in batch_ids.items():
            out[ids] = self.rotations[tfm_id](x[ids])
            labels[ids] = tfm_id
        return out, labels


class RotationTask(Task):
    def __init__(
        self, embedding_size: int, transforms: KorniaTransforms, batch_rotation: BatchRotation
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
        self.batch_rotation = batch_rotation
        self._training = True

    @property
    def name(self):
        return "Rotation"

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> TaskOutput:

        x = self._transform(x)
        x, y = self.batch_rotation(x)
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
