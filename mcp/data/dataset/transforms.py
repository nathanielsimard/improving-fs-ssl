from typing import Callable, Tuple, Union
from PIL import Image

import torch
import numpy as np
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


# CIFAR transformations
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)
# Train transform
cifar_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ]
)
# Test transform
cifar_test_transform = DefaultTransform(CIFAR_MEAN, CIFAR_STD)
