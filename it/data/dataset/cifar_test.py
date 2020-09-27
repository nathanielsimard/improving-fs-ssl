import os
import unittest

import torch

from mcp.data.dataset.cifar import CifarFsDatasetLoader
from mcp.data.dataset.dataset import ComposedDataset
from tests.helpers.dataset import unique_classes

OUTPUT_DIR = "/tmp/MCP-FS/cifar-fs-test"
CONVERT_LABELS = False


class CifarFsTest(unittest.TestCase):
    def setUp(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.loader = CifarFsDatasetLoader(CONVERT_LABELS)

    def load_datasets(self):
        self.dataset_train, self.dataset_valid, self.dataset_test = self.loader.load(
            OUTPUT_DIR
        )

    def test_shouldHaveCorrectTypes(self):
        self.load_datasets()

        image, label = self.dataset_train[0]

        self.assertTrue(isinstance(image, torch.Tensor))
        self.assertTrue(isinstance(label, int))

    def test_shouldContainAllItems(self):
        self.load_datasets()

        self.assertEqual(
            60_000,
            len(self.dataset_train) + len(self.dataset_valid) + len(self.dataset_test),
        )

    def test_shouldHave100TotalClasses(self):
        self.load_datasets()

        dataset_full = ComposedDataset(
            [self.dataset_train, self.dataset_valid, self.dataset_test]
        )
        self._test_num_classes(dataset_full, 100)

    def test_shouldHave64TrainingClasses(self):
        self.load_datasets()

        self._test_num_classes(self.dataset_train, 64)

    def test_shouldHave16ValidClasses(self):
        self.load_datasets()

        self._test_num_classes(self.dataset_valid, 16)

    def test_shouldHave20TestClasses(self):
        self.load_datasets()

        self._test_num_classes(self.dataset_test, 20)

    def test_whenConvertLabels_shouldHaveOverlappingClasses(self):
        self.loader = CifarFsDatasetLoader(True)
        self.load_datasets()

        classes_train = sorted(unique_classes(self.dataset_train))
        classes_valid = sorted(unique_classes(self.dataset_valid))
        classes_test = sorted(unique_classes(self.dataset_test))

        self.assertEqual(list(range(64)), classes_train)
        self.assertEqual(list(range(16)), classes_valid)
        self.assertEqual(list(range(20)), classes_test)

    def _test_num_classes(self, dataset, num_classes):
        classes = unique_classes(dataset)
        self.assertEqual(num_classes, len(classes))
