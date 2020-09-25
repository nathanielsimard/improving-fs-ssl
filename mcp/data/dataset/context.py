from mcp.data.dataset.cifar import CifarFsDatasetLoader
from mcp.data.dataset.selector import DatasetLoader


def create(config={"convert_labels": True}) -> DatasetLoader:
    return CifarFsDatasetLoader(config["convert_labels"])
