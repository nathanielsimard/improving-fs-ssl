from enum import Enum
from typing import List, NamedTuple

from mcp.config.loader import ConfigType


class TaskType(Enum):
    SUPERVISED = "supervised"


class TrainerConfig(NamedTuple):
    epochs: int
    tasks: List[TaskType]


def parse(config: ConfigType) -> TrainerConfig:
    config = config["trainer"]
    return TrainerConfig(
        epochs=config["epochs"], tasks=[TaskType(t) for t in config["tasks"]]
    )
