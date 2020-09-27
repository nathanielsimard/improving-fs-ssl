from enum import Enum
from typing import NamedTuple

from mcp.config.loader import ConfigType


class Source(Enum):
    CIFAR_FS = "cifar_fs"


class CifarFsConfig(NamedTuple):
    convert_labels: bool


def _parse_cifar_fs(config: ConfigType) -> CifarFsConfig:
    config = config["cifar_fs"]
    return CifarFsConfig(convert_labels=config["convert_labels"])


class DatasetConfig(NamedTuple):
    source: Source
    cifar_fs: CifarFsConfig


def parse(config: ConfigType) -> DatasetConfig:
    config = config["dataset"]
    return DatasetConfig(
        source=Source(config["source"]), cifar_fs=_parse_cifar_fs(config)
    )