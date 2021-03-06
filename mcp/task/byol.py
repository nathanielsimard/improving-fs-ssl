from copy import deepcopy
from time import time as time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mcp.data.dataset.transforms import KorniaTransforms, TransformType
from mcp.model.base import freeze_weights
from mcp.model.utils import BatchNormHead
from mcp.task.base import Task, TaskOutput
from mcp.task.compute import TaskCompute


class TrainableModule(nn.Module):
    def __init__(self, head_projection: nn.Module, head_prediction: nn.Module):
        super().__init__()
        self.head_projection = head_projection
        self.head_prediction = head_prediction


class BYOLTask(Task):
    def __init__(
        self,
        embedding_size: int,
        transforms: KorniaTransforms,
        head_size: int,
        hidden_size: int,
        tau: float,
        scale: Tuple[float, float],
        key_transforms: Optional[Tuple[str, str]],
        key_forwards: Optional[Tuple[str, str]],
        compute: TaskCompute,
    ):
        super().__init__()
        self.tau = tau
        self.compute = compute
        self.key_forwards = key_forwards
        self.key_transforms = key_transforms
        head_projection = BatchNormHead(embedding_size, hidden_size, head_size)
        head_prediction = BatchNormHead(head_size, hidden_size, head_size)

        self.trainable = TrainableModule(head_projection, head_prediction)

        self._momentum_encoder: Optional[nn.Module] = None
        self._momentum_head_projection: Optional[nn.Module] = None
        self._initial_state_dict = self.state_dict()

        self._training = True
        self.transforms: List[TransformType] = [
            transforms.resize(),
            transforms.color_jitter(hue=0.1, p=0.8),
            transforms.grayscale(p=0.2),
            transforms.random_flip(),
            transforms.gaussian_blur(p=0.1),
            transforms.random_resized_crop(scale=scale),
            transforms.normalize(),
        ]

    @property
    def name(self):
        return "BYOL"

    @property
    def initial_state_dict(self):
        return self._initial_state_dict

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def run(
        self, encoder: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> TaskOutput:
        self._update_momentum_model(encoder, self.trainable.head_projection)

        x1_tfm, x2_tfm = self._compute_transform(x)
        x1, x2 = self._compute_forward(encoder, x1_tfm, x2_tfm)

        online_proj_one = self.trainable.head_projection(x1)
        online_proj_two = self.trainable.head_projection(x2)

        online_pred_one = self.trainable.head_prediction(online_proj_one)
        online_pred_two = self.trainable.head_prediction(online_proj_two)

        with torch.no_grad():
            x1 = self._momentum_encoder(x1_tfm)  # type: ignore
            x2 = self._momentum_encoder(x2_tfm)  # type: ignore

            target_proj_one = self._momentum_head_projection(x1)  # type: ignore
            target_proj_two = self._momentum_head_projection(x2)  # type: ignore

        loss_one = self.loss(online_pred_one, target_proj_two.detach())
        loss_two = self.loss(online_pred_two, target_proj_one.detach())

        loss = (loss_one + loss_two).mean()
        metric = loss.cpu().detach().item()

        return TaskOutput(loss=loss, metric=metric, metric_name="MSE-norm", time=time())

    def _compute_transform(self, x):
        if self.key_transforms is None:
            x1_tfm = self.transform(x)
            x2_tfm = self.transform(x)
        else:
            x1_tfm = self.compute.cache_transform(
                x, training=True, key=self.key_transforms[0]
            )
            x2_tfm = self.compute.cache_transform(
                x, training=True, key=self.key_transforms[1]
            )

        return x1_tfm, x2_tfm

    def _compute_forward(self, encoder, x1_tfm, x2_tfm):
        if self.key_forwards is None:
            x1 = encoder(x1_tfm)
            x2 = encoder(x2_tfm)
        else:
            x1 = self.compute.cache_forward(x1_tfm, encoder, key=self.key_forwards[0])
            x2 = self.compute.cache_forward(x2_tfm, encoder, key=self.key_forwards[1])

        return x1, x2

    def train(self, mode: bool = True):
        self._training = mode
        return super().train(mode)

    def _update_momentum_model(self, encoder: nn.Module, head_projection: nn.Module):
        if self._momentum_encoder is None:
            self._momentum_encoder = _initialize_momentum_module(encoder)

        if self._momentum_head_projection is None:
            self._momentum_head_projection = _initialize_momentum_module(
                head_projection
            )

        if self._training:
            with torch.no_grad():
                _update_momentum_module(encoder, self._momentum_encoder, self.tau)
                _update_momentum_module(
                    head_projection, self._momentum_head_projection, self.tau
                )

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def state_dict(self):
        value = {}
        value["trainable"] = self.trainable.state_dict()
        value["momentum_encoder"] = _state_dict_or_none(self._momentum_encoder)
        value["momentum_head_projection"] = _state_dict_or_none(
            self._momentum_head_projection
        )

        return value

    def load_state_dict(self, value):
        self.trainable.load_state_dict(value["trainable"])
        self._momentum_encoder = value["momentum_encoder"]
        self._momentum_head_projection = value["momentum_head_projection"]


def _state_dict_or_none(module: Optional[nn.Module]):
    if module is None:
        return None

    return module.state_dict()


def _update_momentum_module(module: nn.Module, module_momentum: nn.Module, tau: float):
    for param_q, param_k in zip(module.parameters(), module_momentum.parameters()):
        param_k.data = param_k.data * tau + param_q.data * (1.0 - tau)


def _initialize_momentum_module(module: nn.Module) -> nn.Module:
    momentum_module = deepcopy(module)
    freeze_weights(momentum_module)
    return momentum_module
