from collections import defaultdict
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from mcp.data.dataset.dataset import ComposedDataset, DatasetLoader, IndexedDataset
from mcp.data.dataset.transforms import TransformType, cifar_test_transform

CLASSES_TRAIN = [
    "train",
    "skyscraper",
    "turtle",
    "raccoon",
    "spider",
    "orange",
    "castle",
    "keyboard",
    "clock",
    "pear",
    "girl",
    "seal",
    "elephant",
    "apple",
    "aquarium_fish",
    "bus",
    "mushroom",
    "possum",
    "squirrel",
    "chair",
    "tank",
    "plate",
    "wolf",
    "road",
    "mouse",
    "boy",
    "shrew",
    "couch",
    "sunflower",
    "tiger",
    "caterpillar",
    "lion",
    "streetcar",
    "lawn_mower",
    "tulip",
    "forest",
    "dolphin",
    "cockroach",
    "bear",
    "porcupine",
    "bee",
    "hamster",
    "lobster",
    "bowl",
    "can",
    "bottle",
    "trout",
    "snake",
    "bridge",
    "pine_tree",
    "skunk",
    "lizard",
    "cup",
    "kangaroo",
    "oak_tree",
    "dinosaur",
    "rabbit",
    "orchid",
    "willow_tree",
    "ray",
    "palm_tree",
    "mountain",
    "house",
    "cloud",
]
CLASSES_VALID = [
    "otter",
    "motorcycle",
    "television",
    "lamp",
    "crocodile",
    "shark",
    "butterfly",
    "beaver",
    "beetle",
    "tractor",
    "flatfish",
    "maple_tree",
    "camel",
    "crab",
    "sea",
    "cattle",
]
CLASSES_TEST = [
    "baby",
    "bed",
    "bicycle",
    "chimpanzee",
    "fox",
    "leopard",
    "man",
    "pickup_truck",
    "plain",
    "poppy",
    "rocket",
    "rose",
    "snail",
    "sweet_pepper",
    "table",
    "telephone",
    "wardrobe",
    "whale",
    "woman",
    "worm",
]


class CifarFsDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        classes_mapping: Dict[int, int],
        transform: TransformType = cifar_test_transform,
    ):
        self.dataset = dataset
        self.classes_mapping = classes_mapping
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, label = self.dataset[index]

        label = self.classes_mapping[label]
        img = self.transform(img)

        return img, label


class CifarFsDatasetLoader(DatasetLoader):
    """Create the CIFAR-FS Dataset from the CIFAR-100 Dataset.

    The data will automatically be downloaded if not present in the `output_dir`.
    """

    def __init__(self, convert_labels: bool):
        """Create the loader.

        Args:
            convert_labels: Convert the labels from the original one (100 classes)
                            to the new one depending of the dataset split.
                                - Train: 64 labels
                                - Valid: 16 labels
                                - Test: 20 labels
        """
        self.convert_labels = convert_labels

    def load(self, output_dir: str) -> Tuple[Dataset, Dataset, Dataset]:
        cifar100_train = self._download(output_dir, train=True)
        cifar100_test = self._download(output_dir, train=False)

        classes = cifar100_train.classes  # type: ignore
        # We redo the splitting, so we include all data from the original train and test set.
        cifar100_full = ComposedDataset([cifar100_train, cifar100_test])

        return self._split(cifar100_full, classes)

    def _download(self, output_dir: str, train=True) -> Dataset:
        try:
            return CIFAR100(output_dir, train=train)
        except Exception:
            return CIFAR100(output_dir, download=True, train=train)

    def _split(
        self, cifar100: Dataset, classes: List[str]
    ) -> Tuple[Dataset, Dataset, Dataset]:
        class_to_index: Dict[str, List[int]] = defaultdict(lambda: [])

        for i in range(len(cifar100)):
            _, clazz_index = cifar100[i]
            clazz = classes[clazz_index]  # type: ignore
            class_to_index[clazz].append(i)

        dataset_train = self._create_dataset(
            classes, CLASSES_TRAIN, cifar100, class_to_index
        )
        dataset_valid = self._create_dataset(
            classes, CLASSES_VALID, cifar100, class_to_index
        )
        dataset_test = self._create_dataset(
            classes, CLASSES_TEST, cifar100, class_to_index
        )

        return dataset_train, dataset_valid, dataset_test

    def _create_dataset(
        self,
        classes_total: List[str],
        classes_split: List[str],
        dataset: Dataset,
        class_to_index: Dict[str, List[int]],
    ) -> Dataset:
        dataset = self._indexed_dataset(classes_split, class_to_index, dataset)
        dataset = self._cifar_fs_dataset(classes_total, classes_split, dataset)

        return dataset

    def _cifar_fs_dataset(
        self, classes_total: List[str], classes_split: List[str], dataset: Dataset
    ) -> Dataset:
        mapping = {}
        current_index = 0

        for i, clazz in enumerate(classes_total):
            if not self.convert_labels:
                mapping[i] = i
            elif clazz in classes_split:
                mapping[i] = current_index
                current_index += 1

        return CifarFsDataset(dataset, mapping)

    def _indexed_dataset(
        self,
        classes_split: List[str],
        class_to_index: Dict[str, List[int]],
        dataset: Dataset,
    ) -> Dataset:
        indexes = []
        for clazz in classes_split:
            indexes += class_to_index[clazz]

        return IndexedDataset(dataset, indexes)
