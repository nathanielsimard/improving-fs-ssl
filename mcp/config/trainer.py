from typing import NamedTuple

from mcp.config.loader import ConfigType


class SupportTrainingConfig(NamedTuple):
    max_epochs: int
    min_loss: int


class TrainerConfig(NamedTuple):
    epochs: int
    support_training: SupportTrainingConfig


def parse(config: ConfigType) -> TrainerConfig:
    config = config["trainer"]
    return TrainerConfig(
        epochs=config["epochs"], support_training=_parse_support_training(config),
    )


def _parse_support_training(config: ConfigType) -> SupportTrainingConfig:
    config = config["support_training"]
    return SupportTrainingConfig(
        max_epochs=config["max_epochs"], min_loss=config["min_loss"]
    )
