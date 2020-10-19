from typing import NamedTuple

from mcp.config.loader import ConfigType


class DataLoaderConfig(NamedTuple):
    batch_size: int
    num_workers: int
    shuffle: bool


def parse(config: ConfigType) -> DataLoaderConfig:
    config = config["dataloader"]

    return DataLoaderConfig(
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
    )
