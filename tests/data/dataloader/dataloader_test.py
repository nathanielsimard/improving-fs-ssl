import unittest

from mcp.data.dataloader.dataloader import DataLoaderFactory, FewShotDataLoaderFactory
from mcp.data.dataset.dataset import ListDataset, create_few_shot_datasets
from tests.helpers.dataset import create_random_dataset

NUM_ITEMS = 250
NUM_CLASSES = 10
SHAPE = (24, 24)

BATCH_SIZE = 16
SHUFFLE = False
PIN_MEMORY = False


K_SHOT = 1
N_WAY = 5


class FewShotDataLoaderFactoryTest(unittest.TestCase):
    def setUp(self):
        samples = create_random_dataset(NUM_ITEMS, NUM_CLASSES, SHAPE).samples  # type: ignore
        self.dataset = create_few_shot_datasets(ListDataset(samples), N_WAY)
        self.dataloader_factory = DataLoaderFactory(BATCH_SIZE, SHUFFLE, PIN_MEMORY)

    def test_whenCreate_shouldOnlyContainsNWayClasses(self):
        factory = FewShotDataLoaderFactory(NUM_CLASSES, N_WAY, self.dataloader_factory)
        dataloader = factory.create(self.dataset)

        classes_support = set()
        classes_query = set()

        for _, y in dataloader.support:
            for yi in y:
                classes_support.add(yi.item())

        for _, y in dataloader.query:
            for yi in y:
                classes_query.add(yi.item())

        self.assertEqual(N_WAY, len(classes_query))
        self.assertEqual(N_WAY, len(classes_support))
        self.assertEqual(classes_support, classes_query)
