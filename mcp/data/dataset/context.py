from mcp.data.dataset.cifar import CifarFsDatasetLoader
from mcp.data.dataset.selector import DatasetLoader


def create(config) -> DatasetLoader:
    return CifarFsDatasetLoader()
