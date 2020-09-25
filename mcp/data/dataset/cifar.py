from collections import defaultdict
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from mcp.data.dataset.dataset import ComposedDataset, DatasetLoader, IndexedDataset

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


class CifarFsDatasetLoader(DatasetLoader):
    """Create the CIFAR-FS Dataset from the CIFAR-100 Dataset.

    The data will automatically be downloaded if not present in the `output_dir`.
    """

    def load(self, output_dir: str) -> Tuple[Dataset, Dataset, Dataset]:
        cifar100_train = self.download(output_dir, train=True)
        cifar100_test = self.download(output_dir, train=False)

        # We redo the splitting, so we include all data from the original train and test set.
        classes = cifar100_train.classes
        cifar100_full = ComposedDataset([cifar100_train, cifar100_test])

        return self.split(cifar100_full, classes)

    def download(self, output_dir: str, train=True):
        try:
            return CIFAR100(output_dir, train=train)
        except Exception:
            return CIFAR100(output_dir, download=True, train=train)

    def split(self, cifar100, classes) -> Tuple[Dataset, Dataset, Dataset]:
        class_to_index: Dict[str, List[int]] = defaultdict(lambda: [])

        for i in range(len(cifar100)):
            _, clazz_index = cifar100[i]
            clazz = classes[clazz_index]  # type: ignore
            class_to_index[clazz].append(i)

        def create_dataset(classes):
            indexes = []
            for clazz in classes:
                indexes += class_to_index[clazz]
            return IndexedDataset(cifar100, indexes)

        return (
            create_dataset(CLASSES_TRAIN),
            create_dataset(CLASSES_VALID),
            create_dataset(CLASSES_TEST),
        )
