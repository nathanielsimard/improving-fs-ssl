import os
import pickle
from typing import List, NamedTuple, Tuple
from zipfile import ZipFile

import gdown
import numpy as np
import torch

from mcp.data.dataset.dataset import (
    ComposedDataset,
    Dataset,
    DatasetLoader,
    DatasetMetadata,
    DatasetSplits,
)

from mcp.utils import logging

logger = logging.create_logger(__name__)

IMAGE_MEAN = (120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0)
IMAGES_STD = (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0)


CLASSES_TRAIN = list(range(0, 64))
CLASSES_VALID = list(range(64, 80))
CLASSES_TEST = list(range(80, 100))


class _Dataset(NamedTuple):
    images: np.ndarray
    labels: np.ndarray


class MiniImageNetDataset(Dataset):
    def __init__(self, dataset: _Dataset, first_class_index: int):
        self.dataset = dataset
        self.first_class_index = first_class_index

    def __getitem__(self, index: int):
        image = self.dataset.images[index]
        label = self.dataset.labels[index]

        # To make sure the first class index
        # of the dataset is 0.
        label = label - self.first_class_index

        return torch.tensor(image).transpose(0, -1), int(label)

    def __len__(self):
        return len(self.dataset.labels)


class MiniImageNetDatasetLoader(DatasetLoader):
    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            len(CLASSES_TRAIN), len(CLASSES_VALID), len(CLASSES_TEST)
        )

    def load(self, output_dir: str) -> DatasetSplits:
        download_dir = self._download(output_dir)
        dataset_trains, dataset_valid, dataset_test = self._load(download_dir)

        return DatasetSplits(
            train=ComposedDataset(
                [
                    self._convert(dataset_train, CLASSES_TRAIN)
                    for dataset_train in dataset_trains
                ]
            ),
            valid=self._convert(dataset_valid, CLASSES_VALID),
            test=self._convert(dataset_test, CLASSES_TEST),
        )

    def _download(self, output_dir: str) -> str:
        url = "https://drive.google.com/u/0/uc?export=download&confirm=juh4&id=1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7"
        output = os.path.join(output_dir, "miniImageNet")
        output_zip = output + ".zip"

        if not os.path.exists(output_zip):
            logger.info("Downloading Mini Image Net...")
            gdown.download(url, output_zip, quiet=False)

        if not os.path.exists(output):
            logger.info("Extracting Mini Image Net...")
            self._extract(output_zip, output)

        return output

    def _extract(self, output_zip: str, output: str):
        os.makedirs(output, exist_ok=True)

        with ZipFile(output_zip, "r") as zipObj:
            zipObj.extractall(output)

    def _load(self, download_dir) -> Tuple[List[_Dataset], _Dataset, _Dataset]:
        file_trains = [
            os.path.join(
                download_dir, "miniImageNet_category_split_train_phase_train.pickle"
            ),
            os.path.join(
                download_dir, "miniImageNet_category_split_train_phase_val.pickle"
            ),
            os.path.join(
                download_dir, "miniImageNet_category_split_train_phase_test.pickle"
            ),
        ]
        file_valid = os.path.join(
            download_dir, "miniImageNet_category_split_val.pickle"
        )
        file_test = os.path.join(
            download_dir, "miniImageNet_category_split_test.pickle"
        )

        return (
            [self._load_dataset(file_train) for file_train in file_trains],
            self._load_dataset(file_valid),
            self._load_dataset(file_test),
        )

    def _load_dataset(self, file: str) -> _Dataset:
        logger.info(f"Loading {file}...")
        try:
            with open(file, "rb") as fo:
                data = pickle.load(fo)
        except Exception:
            with open(file, "rb") as f:
                u = pickle._Unpickler(f)  # type: ignore
                u.encoding = "latin1"
                data = u.load()
        return _Dataset(images=data["data"], labels=data["labels"])

    def _convert(self, dataset: _Dataset, classes: List[int]) -> MiniImageNetDataset:
        return MiniImageNetDataset(dataset, classes[0])
