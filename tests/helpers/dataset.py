from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

from mcp.data.dataset.composed import ListDataset


def create_random_dataset(num_items: int, num_classes: int, shape: Tuple) -> Dataset:
    samples = [
        (np.random.random(shape), np.random.randint(0, num_classes - 1))
        for _ in range(num_items)
    ]
    return ListDataset(samples)


def item_equal(item1: Tuple[np.ndarray, int], item2: Tuple[np.ndarray, int]) -> bool:
    sample_equal = np.array_equal(item1[0], item2[0])
    label_equal = item1[1] == item2[1]

    return sample_equal and label_equal
