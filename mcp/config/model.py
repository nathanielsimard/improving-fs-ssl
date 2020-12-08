from enum import Enum
from typing import NamedTuple

from mcp.config.loader import ConfigType


class ModelArchitecture(Enum):
    RESNET_50 = "resnet-50"
    RESNET_18 = "resnet-18"


class ModelConfig(NamedTuple):
    architecture: ModelArchitecture
    embedding_size: int


def parse(config: ConfigType) -> ModelConfig:
    config = config["model"]

    return ModelConfig(
        architecture=ModelArchitecture(config["architecture"]),
        embedding_size=config["embedding_size"],
    )
