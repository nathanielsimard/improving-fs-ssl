import abc
from typing import List, Tuple

from torch.utils.data import Dataset


class MergedDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets


class DatasetLoader(abc.ABC):
    def load(self, output_dir: str) -> Tuple[Dataset, Dataset, Dataset]:
        """Load the dataset correctly splitted.

        The splits are (train, valid, test).
        """
        pass
