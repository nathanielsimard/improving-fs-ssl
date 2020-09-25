import unittest

from mcp.data.dataset.dataset import ComposedDataset, ListDataset
from tests.helpers.dataset import create_random_dataset, item_equal

NUM_ITEMS = 100
NUM_CLASSES = 10
SHAPE = (24, 24)


class ComposedDatasetTest(unittest.TestCase):
    def setUp(self):
        samples = create_random_dataset(NUM_ITEMS, NUM_CLASSES, SHAPE).samples  # type: ignore

        self.dataset_1 = ListDataset(samples[0:20])
        self.dataset_2 = ListDataset(samples[20:40])
        self.dataset_3 = ListDataset(samples[70:100])
        self.composed_dataset = ComposedDataset(
            [self.dataset_1, self.dataset_2, self.dataset_3]
        )

    def test_shouldContainsAllItems(self):
        dataset = ListDataset(
            self.dataset_1.samples + self.dataset_2.samples + self.dataset_3.samples
        )

        for index in range(len(dataset)):
            self.assertTrue(item_equal(self.composed_dataset[index], dataset[index]))

    def test_shouldSupportNegativeIndex(self):
        dataset = ListDataset(
            self.dataset_1.samples + self.dataset_2.samples + self.dataset_3.samples
        )

        for i in range(len(dataset)):
            index = -(i + 1)
            self.assertTrue(item_equal(self.composed_dataset[index], dataset[index]))

    def test_whenIndexToBig_shouldRaise(self):
        datasets = [
            create_random_dataset(NUM_ITEMS, NUM_CLASSES, SHAPE),
            create_random_dataset(NUM_ITEMS, NUM_CLASSES, SHAPE),
        ]
        dataset = ComposedDataset(datasets)

        self.assertRaises(ValueError, lambda: dataset[200])
        self.assertRaises(ValueError, lambda: dataset[201])
        self.assertRaises(ValueError, lambda: dataset[-201])
