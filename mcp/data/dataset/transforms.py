from typing import Callable, Tuple, Union
from PIL import Image

import torch
import torchvision.transforms as transforms

# Transforms can be a composition, a callable function or object
TransformType = Union[transforms.Compose, Callable]
FloatSequenceType = Tuple[float, float, float]


# Default transform
class DefaultTransform(object):
    def __init__(self, mean: FloatSequenceType, std: FloatSequenceType):
        super().__init__()
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, image: Image) -> torch.Tensor:
        return self.transform(image)
