from typing import Callable, Dict, List

import torch
from torch import nn

from mcp.data.dataset.transforms import KorniaTransforms, TransformType


class TaskCompute(object):
    def __init__(self, transforms: KorniaTransforms):
        self.transforms_train: List[TransformType] = [
            transforms.resize(),
            transforms.random_crop(),
            transforms.color_jitter(),
            transforms.random_flip(),
            transforms.normalize(),
        ]

        self.transforms_eval = [transforms.random_crop(), transforms.normalize()]
        self._cache: Dict[str, torch.Tensor] = {}

    def cache_transform(self, x: torch.Tensor, training: bool, key="default-transform"):
        return self.cache(lambda: self.transform(x, training), key)

    def cache_forward(self, x: torch.Tensor, module: nn.Module, key="default-key"):
        return self.cache(lambda: module(x), key)

    def cache(
        self, func: Callable[[], torch.Tensor], key: str,
    ):
        try:
            return self._cache[key]
        except KeyError:
            x = func()
            self._cache[key] = x
            return x

    def cache_clear(self):
        self._cache.clear()

    def transform(self, x: torch.Tensor, training: bool):
        transforms = self.transforms_train if training else self.transforms_eval
        for t in transforms:
            x = t(x)
        return x
