import os
import unittest

from mcp.data.dataset.cifar import CifarFsDatasetLoader
from mcp.data.dataset.dataset import ComposedDataset

OUTPUT_DIR = "/tmp/MCP-FS/cifar-fs-test"


class CifarFsTest(unittest.TestCase):
    def setUp(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        loader = CifarFsDatasetLoader()
        self.dataset_train, self.dataset_valid, self.dataset_test = loader.load(
            OUTPUT_DIR
        )

    def test_shouldContainAllItems(self):
        self.assertEqual(
            60_000,
            len(self.dataset_train) + len(self.dataset_valid) + len(self.dataset_test),
        )

    def test_shouldHave100TotalClasses(self):
        dataset_full = ComposedDataset(
            [self.dataset_train, self.dataset_valid, self.dataset_test]
        )
        self._test_num_classes(dataset_full, 100)

    def test_shouldHave64TrainingClasses(self):
        self._test_num_classes(self.dataset_train, 64)

    def test_shouldHave16ValidClasses(self):
        self._test_num_classes(self.dataset_valid, 16)

    def test_shouldHave20TestClasses(self):
        self._test_num_classes(self.dataset_test, 20)

    def _test_num_classes(self, dataset, num_classes):
        classes = set()
        for i in range(len(dataset)):
            classes.add(dataset[i][1])

        self.assertEqual(num_classes, len(classes))
