import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from mcp.data.dataset.transforms import KorniaTransforms
from mcp.metric import Accuracy
from mcp.task.base import Task, TaskOutput


class BatchRotation(object):
    def __init__(
        self, transforms: KorniaTransforms, degrees: List[int] = [0, 90, 180, 270]
    ):
        self.rotations = [transforms.rotate(d) for d in degrees]
        self.num_classes = len(degrees)

    def rotate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tfm_ids = list(range(len(self.rotations)))
        sample_ids = list(range(x.size(0)))
        random.shuffle(sample_ids)
        batch_ids: Dict[int, List[int]] = defaultdict(lambda: [])

        for _id in sample_ids:
            tfm_id = random.sample(tfm_ids, 1)[0]
            batch_ids[tfm_id].append(_id)

        out = torch.empty_like(x)
        labels = torch.empty(x.size(0), dtype=torch.long, device=x.device)

        for tfm_id, ids in batch_ids.items():
            out[ids] = self.rotations[tfm_id](x[ids])
            labels[ids] = torch.tensor(tfm_id, dtype=labels.dtype, device=labels.device)

        return out, labels


class RotationTask(Task):
    def __init__(
        self,
        embedding_size: int,
        transforms: KorniaTransforms,
        batch_rotation: BatchRotation,
    ):
        super().__init__()
        self.metric = Accuracy()
        self.output = nn.Linear(embedding_size, batch_rotation.num_classes)
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
        x, y = self.batch_rotation.rotate(x)
        x = encoder(x)
        x = self.output(x)

        metric = self.metric(x, y)
        loss = self.loss(x, y)

        return TaskOutput(loss=loss, metric=metric, metric_name="acc")

    def _plot_and_exit(self, x):
        import torchvision.utils as tvu

        tvu.save_image(x[0], "/tmp/test-ori.png")
        x, y = self.batch_rotation.rotate(x)
        tvu.save_image(x[0], "/tmp/test-rot.png")
        print(f"Rotate the thing {y[0]}")
        raise Exception("Exit")

    def _transform(self, x: torch.Tensor):
        transforms = self.transforms_train if self._training else self.transforms_eval
        for t in transforms:
            x = t(x)
        return x

    def train(self, mode: bool = True):
        self._training = mode
        return super().train(mode)
