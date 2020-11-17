import unittest

import numpy as np
import torch

from mcp.data.dataset.transforms import KorniaTransforms
from tests.helpers.dataset import create_random_dataset

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


class KorniaTransformTest(unittest.TestCase):
    def setUp(self):
        self.dataset = create_random_dataset(100, 3, (3, 12, 12))
        self.transforms = KorniaTransforms(MEAN, STD, (10, 10), 2)

    def test_whenRotateWith0Degree_shouldNotRotateImage(self):
        for i in range(len(self.dataset)):
            image, label = self.dataset[i]
            transform = self.transforms.rotate(0)
            image_rot = transform(torch.tensor(image)).numpy()[0]  # Remove batch size
            self.assertTrue(np.allclose(image, image_rot))

    def test_whenRotate_allRotationsShouldBeDifferent(self):
        for i in range(len(self.dataset)):
            rotations = [self.transforms.rotate(d) for d in [0, 90, 180, 270]]

            image, label = self.dataset[i]
            images = set()

            for rotation in rotations:
                images.add(str(rotation(torch.tensor(image)).tolist()))

            self.assertEqual(len(images), len(rotations))
