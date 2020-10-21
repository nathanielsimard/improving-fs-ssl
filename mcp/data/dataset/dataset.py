import abc
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple

from torch.utils.data import DataLoader, Dataset


class FewShotDataset(NamedTuple):
    support: Dataset
    query: Dataset


class FewShotDataLoader(NamedTuple):
    support: DataLoader
    query: DataLoader


class FewShotDataLoaderSplits(NamedTuple):
    train: DataLoader
    valid: FewShotDataLoader
    test: FewShotDataLoader


class FewShotDatasetSplits(NamedTuple):
    train: Dataset
    valid: FewShotDataset
    test: FewShotDataset


class DatasetSplits(NamedTuple):
    train: Dataset
    valid: Dataset
    test: Dataset


class DatasetMetadata(NamedTuple):
    train_num_class: int
    valid_num_class: int
    test_num_class: int


class DatasetLoader(abc.ABC):
    @abc.abstractproperty
    def metadata(self) -> DatasetMetadata:
        pass

    @abc.abstractmethod
    def load(self, output_dir: str) -> DatasetSplits:
        """Load the dataset correctly splitted.

        The splits are (train, valid, test).
        """
        pass


class ListDataset(Dataset):
    def __init__(self, samples: List):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class ComposedDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self._sizes = [len(dataset) for dataset in self.datasets]

    def __len__(self) -> int:
        return sum(self._sizes)

    def __getitem__(self, index: int):
        is_negative = index < 0

        if not is_negative:
            if index >= len(self):
                raise ValueError(f"Index {index} exceed dataset size {len(self)}.")
        else:
            if abs(index) > len(self):
                raise ValueError(f"Index {index} exceed dataset size {len(self)}.")

        if is_negative:
            return self._backward(index)

        return self._forward(index)

    def _forward(self, index: int):
        limit_current = 0
        for i, size in enumerate(self._sizes):
            limit_next = limit_current + size
            if index < limit_next:
                idx = index - limit_current
                return self.datasets[i][idx]
            limit_current = limit_next

        raise Exception(f"Internal error, index {index}")

    def _backward(self, index: int):
        limit_current = len(self)
        index = limit_current + index
        for i, size in enumerate(reversed(self._sizes)):
            limit_next = limit_current - size
            if index >= limit_next:
                idx = -1 * (limit_current - index)
                return self.datasets[-(i + 1)][idx]
            limit_current = limit_next

        raise Exception(f"Internal error, index {index}")


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indexes: List[int]):
        self.dataset = dataset
        self.indexes = indexes

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int):
        index = self.indexes[index]
        item = self.dataset[index]

        return item


def create_few_shot_datasets(dataset: Dataset, num_samples: int) -> FewShotDataset:
    """Split the dataset into a support and query dataset.

    The first dataset (support-set) contains 'num_samples' per classes in the original dataset, best used for training.
    The second dataset (query-set) contains all the remaining samples, best used for testing.
    """
    indexes_query = []
    indexes_support = []

    classes: Dict[Any, int] = defaultdict(lambda: 0)

    for index in range(len(dataset)):
        _, clazz = dataset[index]

        if classes[clazz] >= num_samples:
            indexes_query.append(index)
        else:
            classes[clazz] += 1
            indexes_support.append(index)

    return FewShotDataset(
        support=IndexedDataset(dataset, indexes_support),
        query=IndexedDataset(dataset, indexes_query),
    )
