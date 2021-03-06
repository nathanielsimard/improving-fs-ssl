import random
from typing import Callable, Tuple

import torch
import torchvision.transforms as transforms
from kornia import Resize
from kornia.augmentation import (
    ColorJitter,
    Normalize,
    RandomAffine,
    RandomCrop,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
)
from kornia.enhance.adjust import solarize as k_solarize
from kornia.filters import GaussianBlur2d
from PIL import Image

TransformType = Callable[[torch.Tensor], torch.Tensor]
FloatSequenceType = Tuple[float, float, float]


# Default transform
class DefaultTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __call__(self, image: Image) -> torch.Tensor:
        return self.transform(image)


class RandomApply(object):
    def __init__(self, fn, p):
        self.fn = fn
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class KorniaTransforms(object):
    def __init__(
        self,
        mean: FloatSequenceType,
        std: FloatSequenceType,
        image_size: Tuple[int, int],
        random_crop_size: Tuple[int, int],
        random_crop_padding: int,
    ):
        self.mean = mean
        self.std = std
        self.image_size = image_size
        self.random_crop_size = random_crop_size
        self.random_crop_padding = random_crop_padding

    def normalize(self) -> TransformType:
        return Normalize(torch.tensor(self.mean), torch.tensor(self.std))

    def resize(self) -> TransformType:
        return Resize(self.image_size)

    def random_crop(self) -> TransformType:
        return RandomCrop(self.random_crop_size, padding=self.random_crop_padding)

    def random_resized_crop(self, p=1.0, scale=(0.08, 1.0)) -> TransformType:
        return RandomResizedCrop(self.random_crop_size, scale=scale, p=p)

    def random_flip(self, p: float = 0.5) -> TransformType:
        return RandomHorizontalFlip(p=p)

    def color_jitter(self, hue: float = 0.0, p: float = 1.0) -> TransformType:
        return ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=hue, p=p)

    def rotate(self, degree: float, p: float = 1.0) -> TransformType:
        return RandomAffine(degrees=(degree, degree), p=p)

    def grayscale(self, p: float = 1.0) -> TransformType:
        return RandomGrayscale(p=p)

    def gaussian_blur(self, p: float = 1.0) -> TransformType:
        return RandomApply(GaussianBlur2d((3, 3), (1.5, 1.5)), p)

    def solarize(self, threshold: float, p: float = 1.0) -> TransformType:
        return RandomApply(
            lambda x: k_solarize(x, thresholds=threshold, additions=None), p
        )
