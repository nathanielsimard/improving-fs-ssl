from typing import NamedTuple

from mcp.config.loader import ConfigType


class ModelConfig(NamedTuple):
    embedding_size: int


def parse(config: ConfigType) -> ModelConfig:
    config = config["model"]

    return ModelConfig(embedding_size=config["embedding_size"])
