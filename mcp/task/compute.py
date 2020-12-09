from typing import Callable, Dict, List, Tuple

import torch
from torch import nn

from mcp.config.transform import Difficulty
from mcp.data.dataset.transforms import KorniaTransforms, TransformType


class TaskCompute(object):
    def __init__(
        self,
        transforms: KorniaTransforms,
        difficulty: Difficulty,
        scale: Tuple[int, int],
    ):

        self.transforms_train = _create_transformations(transforms, difficulty, scale)
        self.transforms_eval = [transforms.resize(), transforms.normalize()]
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


def _create_transformations(
    transforms: KorniaTransforms, difficulty: Difficulty, scale: Tuple[int, int],
) -> List[TransformType]:
    if difficulty == Difficulty.DEFAULT:
        return [
            transforms.resize(),
            transforms.random_crop(),
            transforms.color_jitter(),
            transforms.random_flip(),
            transforms.normalize(),
        ]
    elif difficulty == Difficulty.NONE:
        return [transforms.resize(), transforms.normalize()]
    elif difficulty == Difficulty.HARD:
        return [
            transforms.resize(),
            transforms.color_jitter(hue=0.1, p=0.8),
            transforms.grayscale(p=0.2),
            transforms.random_flip(),
            transforms.gaussian_blur(p=0.1),
            transforms.random_resized_crop(scale=scale),
            transforms.normalize(),
        ]
    else:
        raise ValueError(f"Difficulty not yet supported {difficulty}.")
