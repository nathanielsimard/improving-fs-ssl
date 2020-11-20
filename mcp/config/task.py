from enum import Enum
from typing import List, NamedTuple

from mcp.config.loader import ConfigType


class TaskType(Enum):
    SUPERVISED = "supervised"
    ROTATION = "rotation"
    BYOL = "byol"


class BYOLConfig(NamedTuple):
    head_size: int
    tau: float


class TaskConfig(NamedTuple):
    train: List[TaskType]
    valid: List[TaskType]
    byol: BYOLConfig


def parse(config: ConfigType) -> TaskConfig:
    config = config["task"]

    return TaskConfig(
        train=[TaskType(t) for t in config["train"]],
        valid=[TaskType(t) for t in config["valid"]],
        byol=_parse_byol(config),
    )


def _parse_byol(config: ConfigType) -> BYOLConfig:
    config = config["byol"]

    return BYOLConfig(head_size=config["head_size"], tau=config["tau"],)
