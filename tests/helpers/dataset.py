from typing import Any, List, Set, Tuple

import numpy as np
from torch.utils.data import Dataset

from mcp.data.dataset.dataset import ListDataset


def create_random_dataset(num_items: int, num_classes: int, shape: Tuple) -> Dataset:
    samples = [
        (np.random.random(shape), np.random.randint(0, num_classes))
        for _ in range(num_items)
    ]
    return ListDataset(samples)


def item_equal(item1: Tuple[np.ndarray, int], item2: Tuple[np.ndarray, int]) -> bool:
    sample_equal = np.array_equal(item1[0], item2[0])
    label_equal = item1[1] == item2[1]

    return sample_equal and label_equal


def unique_classes(dataset: Dataset) -> Set[Any]:
    classes = set()
    for i in range(len(dataset)):
        classes.add(dataset[i][1])

    return classes


def unique_samples(datasets: List[Dataset]) -> Set[Any]:
    samples = set()
    for dataset in datasets:
        for i in range(len(dataset)):
            item = dataset[i]
            samples.add((str(item[0]), item[1]))
    return samples
