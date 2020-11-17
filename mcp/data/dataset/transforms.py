from typing import Callable, Tuple, Union

import torch
import torchvision.transforms as transforms
from kornia.augmentation import (
    ColorJitter,
    Normalize,
    RandomAffine,
    RandomCrop,
    RandomHorizontalFlip,
)
from PIL import Image

# Transforms can be a composition, a callable function or object
TransformType = Union[transforms.Compose, Callable]
FloatSequenceType = Tuple[float, float, float]


# Default transform
class DefaultTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __call__(self, image: Image) -> torch.Tensor:
        return self.transform(image)


class KorniaTransforms(object):
    def __init__(
        self,
        mean: FloatSequenceType,
        std: FloatSequenceType,
        random_crop_size: Tuple[int, int],
        random_crop_padding: int,
    ):
        self.mean = mean
        self.std = std
        self.random_crop_size = random_crop_size
        self.random_crop_padding = random_crop_padding

    def normalize(self) -> Normalize:
        return Normalize(torch.tensor(self.mean), torch.tensor(self.std))

    def random_crop(self) -> RandomCrop:
        return RandomCrop(self.random_crop_size, padding=self.random_crop_padding)

    def random_flip(self, p: float = 0.5) -> RandomHorizontalFlip:
        return RandomHorizontalFlip(p=p)

    def color_jitter(self) -> ColorJitter:
        return ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

    def rotate(self, degree: float, p: float = 1.0):
        return RandomAffine(degrees=(degree, degree))
