from enum import Enum
from typing import NamedTuple

from mcp.config.loader import ConfigType


class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"


class SGDConfig(NamedTuple):
    momentum: float


class _OptimizerConfig(NamedTuple):
    type: OptimizerType
    sgd: SGDConfig
    learning_rate: float
    weight_decay: float


class OptimizerConfig(NamedTuple):
    train: _OptimizerConfig
    support: _OptimizerConfig


def _parse_sgd(config: ConfigType) -> SGDConfig:
    config = config["sgd"]
    return SGDConfig(momentum=config["momentum"])


def parse(config: ConfigType) -> OptimizerConfig:
    config = config["optimizer"]
    return OptimizerConfig(
        train=_parse(config, "train"), support=_parse(config, "support"),
    )


def _parse(config: ConfigType, tag: str) -> _OptimizerConfig:
    config = config[tag]
    return _OptimizerConfig(
        type=OptimizerType(config["type"]),
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        sgd=_parse_sgd(config),
    )
