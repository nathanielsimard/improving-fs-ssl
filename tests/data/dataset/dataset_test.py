import unittest

from mcp.data.dataset.dataset import (
    ComposedDataset,
    ListDataset,
    create_few_shot_datasets,
)
from tests.helpers.dataset import (
    create_random_dataset,
    item_equal,
    unique_classes,
    unique_samples,
)

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


class FewShotDatasetTest(unittest.TestCase):
    def test_shouldNotHaveDupplicates(self):
        dataset = create_random_dataset(NUM_ITEMS, NUM_CLASSES, SHAPE)

        train_dataset, test_dataset = create_few_shot_datasets(dataset, 5)

        samples = unique_samples([train_dataset, test_dataset])
        self.assertEqual(len(samples), len(train_dataset) + len(test_dataset))

    def test_givenOneNumSample_trainDatasetShouldHaveOneSamplePerClass(self):
        dataset = create_random_dataset(NUM_ITEMS, NUM_CLASSES, SHAPE)

        train_dataset, test_dataset = create_few_shot_datasets(dataset, 1)

        classes_train = unique_classes(train_dataset)
        self.assertEqual(len(train_dataset), NUM_CLASSES)
        self.assertEqual(len(classes_train), NUM_CLASSES)

    def test_given5NumSamples_trainDatasetShouldHave5SamplesPerClass(self):
        num_samples = 5
        # We add more items to make sure all classes are included
        dataset = create_random_dataset(NUM_ITEMS * num_samples, NUM_CLASSES, SHAPE)

        train_dataset, test_dataset = create_few_shot_datasets(dataset, num_samples)

        classes_train = unique_classes(train_dataset)
        self.assertEqual(len(train_dataset), num_samples * NUM_CLASSES)
        self.assertEqual(len(classes_train), NUM_CLASSES)
