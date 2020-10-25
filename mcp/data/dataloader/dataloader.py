import random
from typing import List, NamedTuple

from torch.utils.data import DataLoader

from mcp.data.dataset.dataset import Dataset, FewShotDataset, IndexedDataset


class FewShotDataLoader(NamedTuple):
    support: DataLoader
    query: DataLoader


class FewShotDataLoaderSplits(NamedTuple):
    train: DataLoader
    valid: FewShotDataLoader
    test: FewShotDataLoader


class DataLoaderFactory(object):
    def __init__(self, batch_size: int, shuffle: bool, pin_memory: bool):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def create(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
        )


class FewShotDataLoaderFactory(object):
    def __init__(
        self, num_classes: int, n_ways: int, dataloader_factory: DataLoaderFactory,
    ):
        self.num_classes = num_classes
        self.n_ways = n_ways
        self.dataloader_factory = dataloader_factory

    def create(self, dataset: FewShotDataset) -> FewShotDataLoader:
        classes = random.sample(list(range(self.num_classes)), self.n_ways)
        support = self._filter_classes(dataset.support, classes)
        query = self._filter_classes(dataset.query, classes)

        return FewShotDataLoader(
            support=self.dataloader_factory.create(support),
            query=self.dataloader_factory.create(query),
        )

    def _filter_classes(self, dataset: Dataset, classes: List[int]) -> Dataset:
        indexes = []

        for index in range(len(dataset)):
            _, clazz = dataset[index]
            if clazz in classes:
                indexes.append(index)

        return IndexedDataset(dataset, indexes)
